import logging
import os
import time
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from Stage_1.registry import get_modality, get_supported_extensions
from paths import SANDBOX_TOOLS, SANDBOX_TASKS

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
	def __init__(self, orchestrator, db, config: dict, on_plugin_changed=None):
		self.orchestrator = orchestrator
		self.db = db
		self.config = config
		self.on_plugin_changed = on_plugin_changed
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

		handler = DebouncedHandler(self)
		for d in valid_dirs:
			self.observer.schedule(handler, d, recursive=True)
			logger.info(f"Watching: {d}")

		# Watch plugin directories for hot-reload
		if self.on_plugin_changed:
			plugin_handler = PluginHandler(self.on_plugin_changed)
			root = Path(self.config.get("_root", ""))
			plugin_dirs = [
				root / "Stage_2" / "tasks",
				root / "Stage_3" / "tools",
				SANDBOX_TOOLS,
				SANDBOX_TASKS,
			]
			for plugin_dir in plugin_dirs:
				if plugin_dir.exists():
					self.observer.schedule(plugin_handler, str(plugin_dir), recursive=False)
					logger.info(f"Watching plugins: {plugin_dir}")

		self.observer.start()

		# Initial scan in background — doesn't block startup
		threading.Thread(target=self._initial_scan, args=(valid_dirs,), daemon=True).start()

	def rescan(self):
		"""Re-read sync_directories from config, update observers, run fresh scan."""
		# Refresh config-driven state
		raw_dirs = self.config.get("sync_directories", [])
		if isinstance(raw_dirs, str):
			raw_dirs = [raw_dirs]
		self.watch_dirs = raw_dirs
		self.ignored_extensions = set(self.config.get("ignored_extensions", []))
		self.ignored_folders = set(self.config.get("ignored_folders", []))
		self.skip_hidden = self.config.get("skip_hidden_folders", True)

		valid_dirs = [d for d in self.watch_dirs if d and os.path.exists(d)]
		if not valid_dirs:
			logger.warning("No valid sync directories after rescan.")
			return

		# Tear down existing observer, create fresh one
		if self.observer.is_alive():
			self.observer.stop()
			self.observer.join()
		self.observer = Observer()

		handler = DebouncedHandler(self)
		for d in valid_dirs:
			self.observer.schedule(handler, d, recursive=True)
			logger.info(f"Watching: {d}")

		if self.on_plugin_changed:
			plugin_handler = PluginHandler(self.on_plugin_changed)
			root = Path(self.config.get("_root", ""))
			plugin_dirs = [
				root / "Stage_2" / "tasks",
				root / "Stage_3" / "tools",
				SANDBOX_TOOLS,
				SANDBOX_TASKS,
			]
			for plugin_dir in plugin_dirs:
				if plugin_dir.exists():
					self.observer.schedule(plugin_handler, str(plugin_dir), recursive=False)

		self.observer.start()
		threading.Thread(target=self._initial_scan, args=(valid_dirs,), daemon=True).start()
		logger.info("Rescan triggered.")

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
		t0 = time.time()
		db_state = self.db.get_watched_files()  # {path: mtime} — watched only
		disk_files = set()
		new_count = 0
		modified_count = 0

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
						self._register_file(path, mtime)
						new_count += 1

					elif abs(mtime - db_state[path]) > 1.0:
						self._register_file(path, mtime)
						modified_count += 1

		# Ghost cleanup — files in DB but not on disk
		ghost_count = 0
		for db_path in db_state:
			if db_path not in disk_files:
				logger.info(f"[Scan] Removed: {Path(db_path).name}")
				self.orchestrator.on_file_deleted(db_path)
				ghost_count += 1

		elapsed = time.time() - t0
		logger.info(
			f"Initial scan complete: {len(disk_files)} files on disk, "
			f"{new_count} new, {modified_count} modified, {ghost_count} ghosts removed "
			f"({elapsed:.2f}s)"
		)

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
		except OSError as e:
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
		"""Filter out junk files that shouldn't be indexed."""
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
		"""Called after the debounce interval expires — events have settled."""
		with self.lock:
			self.pending.pop(path, None)
		logger.debug(f"[Debounce] Firing for: {Path(path).name}")
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


class PluginHandler(FileSystemEventHandler):
	"""
	Watches plugin directories for new/modified task and tool files.
	Debounces changes (2s) then calls the reload callback.

	Includes mtime tracking to avoid false alarms — on Windows, merely
	reading a file can update its access time and trigger on_modified.
	"""

	def __init__(self, on_plugin_changed):
		self.on_plugin_changed = on_plugin_changed
		self._debounce_timer = None
		self._lock = threading.Lock()
		self._known_mtimes: dict[str, float] = {}

	def _is_plugin_file(self, path: str) -> bool:
		name = Path(path).name
		return (name.startswith("task_") or name.startswith("tool_")) and name.endswith(".py")

	def _mtime_changed(self, path: str) -> bool:
		"""Return True only if the file's mtime actually changed."""
		try:
			current = os.path.getmtime(path)
		except OSError:
			return False
		last = self._known_mtimes.get(path)
		if last and abs(current - last) < 0.1:
			return False
		self._known_mtimes[path] = current
		return True

	def _schedule_reload(self):
		with self._lock:
			if self._debounce_timer:
				self._debounce_timer.cancel()
			self._debounce_timer = threading.Timer(2.0, self.on_plugin_changed)
			self._debounce_timer.start()

	def on_created(self, event):
		if not event.is_directory and self._is_plugin_file(event.src_path):
			try:
				self._known_mtimes[event.src_path] = os.path.getmtime(event.src_path)
			except OSError:
				pass
			logger.info(f"[Plugin] New: {Path(event.src_path).name}")
			self._schedule_reload()

	def on_modified(self, event):
		if not event.is_directory and self._is_plugin_file(event.src_path):
			if not self._mtime_changed(event.src_path):
				return  # False alarm — file was read, not written
			logger.info(f"[Plugin] Modified: {Path(event.src_path).name}")
			self._schedule_reload()