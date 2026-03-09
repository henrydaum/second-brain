import logging
import os
import time
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from Stage_1.registry import get_modality, get_supported_extensions

logger = logging.getLogger("Watcher")

"""
Watcher.

Monitors directories for file changes and keeps the files table in sync.
The watcher does NOT know about tasks. It does two things:

	1. Maintain the files table (upsert on create/modify, remove on delete)
	2. Notify the orchestrator (on_file_discovered / on_file_deleted)

The orchestrator decides what tasks to queue. The watcher just reports
what happened on disk.

Hard problems solved here:
	- Debouncing: Windows/watchdog fire multiple events for one change.
	  We use a timer per path — only fire after 1s of silence.
	- Ghosts: Files in the DB that no longer exist on disk. Caught during
	  initial scan by diffing DB state against disk state.
	- Folder moves: Moving a folder triggers delete+create for every file
	  inside. We handle this by walking the new location and treating
	  each file as a discovery. Deletes are handled by ghost cleanup.
	- False alarms: Viewing an image on Windows updates its access time,
	  triggering a modify event. We track mtimes and ignore events where
	  the mtime hasn't actually changed.
"""


class Watcher:
	def __init__(self, orchestrator, db, config: dict):
		self.orchestrator = orchestrator
		self.db = db
		self.config = config
		self.observer = Observer()

		# Directories to watch
		raw_dirs = config.get("sync_directories", [])
		if isinstance(raw_dirs, str):
			raw_dirs = [raw_dirs]
		self.watch_dirs = raw_dirs

		# Known extensions from the parser registry
		self.supported_extensions = get_supported_extensions()

		# Ignored extensions from config
		self.ignored_extensions = set(config.get("ignored_extensions", []))

		# Mtime cache — detects false alarms from Windows
		self._known_mtimes: dict[str, float] = {}

		# Ignored folders
		self.ignored_folders = set(config.get("ignored_folders", []))
		self.skip_hidden = config.get("skip_hidden_folders", True)

	# =================================================================
	# START / STOP
	# =================================================================

	def start(self):
		valid_dirs = [d for d in self.watch_dirs if d and os.path.exists(d)]
		if not valid_dirs:
			logger.error("No valid sync directories found. Watcher not starting.")
			return

		logger.info("Running initial scan...")
		self._initial_scan(valid_dirs)

		handler = DebouncedHandler(self)
		for d in valid_dirs:
			self.observer.schedule(handler, d, recursive=True)
			logger.info(f"Watching: {d}")

		self.observer.start()

	def stop(self):
		if self.observer.is_alive():
			self.observer.stop()
			self.observer.join()
		logger.info("Watcher stopped.")

	# =================================================================
	# INITIAL SCAN
	# =================================================================

	def _initial_scan(self, valid_dirs):
		"""
		Walk all watched directories. Compare disk state to DB state.
		New files -> upsert + notify orchestrator.
		Modified files -> upsert + notify orchestrator.
		Deleted files (ghosts) -> remove from DB + notify orchestrator.
		"""
		db_state = self.db.get_all_files()  # {path: mtime}
		disk_files = set()

		for watch_dir in valid_dirs:
			if self._is_ignored(watch_dir):
				continue

			for root, dirs, files in os.walk(watch_dir):
				if self._is_ignored(root):
					continue
				# Prune ignored dirs in-place so os.walk skips them
				dirs[:] = [d for d in dirs
						   if not self._is_ignored(os.path.join(root, d))]

				for name in files:
					path = str(Path(os.path.join(root, name)))

					if not self._is_valid_file(path):
						continue

					disk_files.add(path)
					mtime = os.path.getmtime(path)
					self._known_mtimes[path] = mtime

					if path not in db_state:
						# New file
						self._register_file(path, mtime)

					elif abs(mtime - db_state[path]) > 1.0:
						# Modified file
						self._register_file(path, mtime)

		# Ghost cleanup — files in DB but not on disk
		for db_path in db_state:
			if db_path not in disk_files:
				logger.info(f"[Scan] Removed: {Path(db_path).name}")
				self.orchestrator.on_file_deleted(db_path)

	# =================================================================
	# FILE REGISTRATION
	# =================================================================

	def _register_file(self, path: str, mtime: float):
		"""Upsert a file into the DB and notify the orchestrator."""
		p = Path(path)
		ext = p.suffix.lower()
		modality = get_modality(ext)

		self.db.upsert_file(
			path=path,
			file_name=p.name,
			extension=ext,
			modality=modality,
			mtime=mtime,
		)
		self.orchestrator.on_file_discovered(path, ext, modality)

	# =================================================================
	# LIVE EVENT HANDLING
	# =================================================================

	def handle_create_or_modify(self, path: str):
		"""Called by the debounced handler after events settle."""
		if not os.path.exists(path):
			return

		# Folder pasted/moved in — walk it
		if os.path.isdir(path):
			if self._is_ignored(path):
				return
			logger.info(f"[Live] Scanning directory: {Path(path).name}")
			for root, dirs, files in os.walk(path):
				if self._is_ignored(root):
					continue
				dirs[:] = [d for d in dirs if not self._is_ignored(os.path.join(root, d))]
				for name in files:
					file_path = str(Path(os.path.join(root, name)))
					if self._is_valid_file(file_path):
						mtime = os.path.getmtime(file_path)
						self._known_mtimes[file_path] = mtime
						self._register_file(file_path, mtime)
			return

		# Single file
		if not self._is_valid_file(path):
			return

		try:
			current_mtime = os.path.getmtime(path)
			last_mtime = self._known_mtimes.get(path)

			# False alarm — mtime hasn't actually changed
			if last_mtime and abs(current_mtime - last_mtime) < 0.1:
				return

			self._known_mtimes[path] = current_mtime
			logger.info(f"[Live] Changed: {Path(path).name}")
			self._register_file(path, current_mtime)
		except OSError:
			logger.debug(f"Could not stat {Path(path).name}: {e}")

	def handle_delete(self, path: str):
		"""Called immediately (no debounce) when a file or folder is deleted."""
		db_state = self.db.get_all_files()
		deleted_path = str(Path(path))

		# Find all DB paths that match this path or are inside this folder
		targets = [
			db_path for db_path in db_state
			if db_path == deleted_path
			or db_path.startswith(deleted_path + os.sep)
		]

		for target in targets:
			logger.info(f"[Live] Deleted: {Path(target).name}")
			self._known_mtimes.pop(target, None)
			self.orchestrator.on_file_deleted(target)

	# =================================================================
	# HELPERS
	# =================================================================

	def _is_valid_file(self, path: str) -> bool:
		"""Uselses crap that you don't want, but have to deal with anyway."""
		p = Path(path)
		name = p.name

		# Hidden files
		if name.startswith("."):
			return False

		# Office lock files
		if name.startswith("~$"):
			return False

		# System junk
		if name.lower() in ("thumbs.db", "desktop.ini", "ds_store", ".ds_store"):
			return False

		# SQLite sidecar files
		if any(name.endswith(suffix) for suffix in ("-wal", "-shm", "-journal")):
			return False

		# Temp files (common patterns)
		if name.endswith(".tmp") or name.endswith(".temp"):
			return False

		# Ignored extensions from config
		if p.suffix.lower() in self.ignored_extensions:
			return False

		return p.suffix.lower() in self.supported_extensions

	def _is_ignored(self, path: str) -> bool:
		parts = Path(path).parts
		if any(part in self.ignored_folders for part in parts):
			return True
		if self.skip_hidden and any(part.startswith(".") for part in parts):
			return True
		return False


class DebouncedHandler(FileSystemEventHandler):
	"""
	Watchdog fires multiple events per file change. This handler
	debounces them: it waits for 1 second of silence on a path
	before forwarding to the watcher.

	Deletes are NOT debounced — they fire immediately.
	"""

	def __init__(self, watcher: Watcher):
		self.watcher = watcher
		self.debounce_interval = 1.0
		self.pending: dict[str, threading.Timer] = {}
		self.lock = threading.Lock()

	def _debounce(self, path: str):
		with self.lock:
			if path in self.pending:
				self.pending[path].cancel()
			timer = threading.Timer(
				self.debounce_interval,
				self._fire,
				[path],
			)
			self.pending[path] = timer
			timer.start()

	def _fire(self, path: str):
		with self.lock:
			self.pending.pop(path, None)
		self.watcher.handle_create_or_modify(path)

	# --- Watchdog event callbacks ---

	def on_created(self, event):
		self._debounce(event.src_path)

	def on_modified(self, event):
		if event.is_directory:
			return
		self._debounce(event.src_path)

	def on_moved(self, event):
		# Source is gone, destination is new
		self.watcher.handle_delete(event.src_path)
		self._debounce(event.dest_path)

	def on_deleted(self, event):
		# No debounce — delete immediately
		self.watcher.handle_delete(event.src_path)