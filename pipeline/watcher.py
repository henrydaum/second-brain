"""Pipeline support for watcher."""

import logging
import os
import time
import threading
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pipeline.database import set_thread_priority_low
from pipeline.ignore_rules import IgnoreRules
from parsing import get_modality, get_supported_extensions
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
	"""Watcher."""
	def __init__(self, orchestrator, db, config: dict):
		"""Initialize the watcher."""
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

		# What may be indexed at all. The settings are normalized rather than
		# compared as typed — see pipeline/ignore_rules.py.
		self.rules = IgnoreRules.from_config(config)

		# Mtime cache — detects false alarms from Windows
		self._known_mtimes: dict[str, float] = {}

	# =================================================================
	# START / STOP
	# =================================================================

	def start(self):
		"""Start watcher."""
		valid_dirs = [d for d in self.watch_dirs if d and os.path.exists(d)]
		if not valid_dirs:
			logger.error("No valid sync directories found — watcher not starting. Use /configure sync_directories to add a folder.")
			return

		handler = DebouncedHandler(self)
		for d in valid_dirs:
			self.observer.schedule(handler, d, recursive=True)
			logger.info(f"Watching: {d}")

		self.observer.start()

		# Sync in background — doesn't block startup
		threading.Thread(target=self._sync_worker, args=(valid_dirs,), daemon=True).start()

	def rescan(self):
		"""Re-read sync_directories from config, update observers, run fresh scan."""
		# Refresh config-driven state
		raw_dirs = self.config.get("sync_directories", [])
		if isinstance(raw_dirs, str):
			raw_dirs = [raw_dirs]
		self.watch_dirs = raw_dirs
		self.rules = IgnoreRules.from_config(self.config)

		# And the one piece of state that is not config's. Installing a parser
		# is what makes an extension worth looking at, and _should_process gates
		# every file on this set — so a snapshot taken in __init__ meant a newly
		# installed parser's file types stayed invisible until the next boot.
		# ".zip" is the worked example: it is not in parsing._NATIVE_DEFAULTS,
		# so before parse_container is installed an archive is not merely
		# unparseable, it never enters the files table at all. That is also why
		# the fresh scan below is the point of this call rather than a side
		# effect of it — there is nothing to re-resolve, there are files that
		# were never seen.
		self.supported_extensions = get_supported_extensions()

		valid_dirs = [d for d in self.watch_dirs if d and os.path.exists(d)]

		# The sync runs whether or not there is anywhere left to watch. Pruning
		# what the rules now exclude is a question about config, not about what
		# is mounted, and returning early here meant an unplugged drive quietly
		# cancelled the prune the user had just asked for.
		threading.Thread(target=self._sync_worker, args=(valid_dirs,), daemon=True).start()

		if not valid_dirs:
			logger.warning("No valid sync directories after rescan. Use /configure sync_directories to add a folder.")
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

		self.observer.start()
		logger.info("Rescan triggered.")

	def stop(self):
		"""Stop watcher."""
		if self.observer.is_alive():
			self.observer.stop()
			self.observer.join()
		logger.info("Watcher stopped.")

	# =================================================================
	# INITIAL SCAN
	# =================================================================

	def _sync_worker(self, valid_dirs):
		"""Bring the database back in line with config, then with disk.

		The order is the point. Pruning what the ignore settings exclude is a
		question about *config*, so it runs first and unconditionally; the disk
		walk is a separate question and must not be able to starve it. Ghost
		cleanup used to be the last statement of an unguarded scan on an
		unguarded daemon thread, so one unreadable file killed the whole
		reconciliation, silently, on every boot — and a setting the user had
		just changed appeared to do nothing at all.
		"""
		set_thread_priority_low()

		try:
			self._prune_by_rules()
		except Exception:
			logger.exception("Ignore-rule prune failed")

		try:
			self._sweep_orphans()
		except Exception:
			logger.exception("Orphan sweep failed")

		if not valid_dirs:
			return

		try:
			self._initial_scan(valid_dirs)
		except Exception:
			logger.exception("Initial scan failed")

	def _prune_by_rules(self):
		"""Drop everything the current ignore settings exclude.

		Reads ``get_all_files()`` rather than ``get_watched_file_state()``,
		which ghost cleanup cannot do: container-extracted children live under
		a temp ``extract_dir`` outside every watch dir, so a disk diff would
		call all of them ghosts. This loop is safe over the whole table
		precisely because it only ever deletes on an explicit rule match, and
		that is also what lets it remove a child sitting under a newly ignored
		folder directly rather than only when its parent archive goes.
		"""
		pruned = 0
		for path in self.db.get_all_files():
			if self.rules.excludes(path):
				logger.info(f"[Rules] Excluded: {Path(path).name}")
				self.orchestrator.on_file_deleted(path)
				pruned += 1
		if pruned:
			logger.info(f"Ignore rules pruned {pruned} file(s) from the index.")

	def _sweep_orphans(self):
		"""Delete output rows whose file is no longer in the files table.

		``on_file_deleted`` knows only the tables of *currently registered*
		tasks, and it drops the files row regardless of what it managed to
		clean. So removing a file while its pipeline package was uninstalled
		stranded that package's rows permanently: with no files row the path
		can never be a ghost again, and nothing that reconciles by path can
		reach it. This is the only thing that can.
		"""
		for table, removed in self.db.sweep_orphaned_output_rows().items():
			logger.info(f"[Sweep] Removed {removed} orphaned row(s) from {table}.")

	def _initial_scan(self, valid_dirs):
		"""
		Walk all watched directories. Compare disk state to DB state.
		New files -> upsert + notify orchestrator.
		Modified files -> upsert + notify orchestrator.
		Reclassified files -> upsert + notify orchestrator.
		Deleted files (ghosts) -> remove from DB + notify orchestrator.

		"Reclassified" is the case a parser install creates: the file on disk
		is untouched, but ``get_modality`` now answers differently about it
		than it did when the row was written. Comparing only mtimes meant such
		a file was skipped forever — the row kept saying ``unknown`` and no
		modality-rooted task ever found it.

		Ghost cleanup is the destructive half, and everything below is arranged
		so it only ever runs on a complete picture of the disk: a file that
		cannot be read is claimed as on-disk anyway, and a walk that hit an
		error suppresses the sweep entirely.
		"""
		t0 = time.time()
		db_state = self.db.get_watched_file_state()  # {path: (mtime, modality)}
		disk_files = set()
		new_count = 0
		modified_count = 0
		reclassified_count = 0
		walk_ok = True

		def on_walk_error(error):
			# os.walk swallows directory errors by default, so an unreadable
			# subtree reads as an empty one — and an empty subtree is exactly
			# what a deleted subtree looks like. Left alone, one permission
			# error would have ghost cleanup delete every file beneath it.
			nonlocal walk_ok
			walk_ok = False
			logger.warning(f"[Scan] Could not read {getattr(error, 'filename', '?')}: {error}")

		for watch_dir in valid_dirs:
			if self._is_ignored(watch_dir):
				continue

			for root, dirs, files in os.walk(watch_dir, onerror=on_walk_error):
				if self._is_ignored(root):
					continue
				# Prune ignored dirs in-place so os.walk skips them
				dirs[:] = [d for d in dirs
						   if not self._is_ignored(os.path.join(root, d))]

				for name in files:
					path = str(Path(os.path.join(root, name)))

					if not self._is_valid_file(path):
						continue

					try:
						mtime = os.path.getmtime(path)
					except FileNotFoundError:
						# Vanished between the listing and the stat. Genuinely
						# gone, so leaving it out of disk_files is correct and
						# ghost cleanup is the right answer for it.
						continue
					except OSError as e:
						# Present but unreadable. Claim it as on-disk so the
						# sweep below cannot delete a file that exists; this
						# used to escape and kill the scan thread outright.
						logger.warning(f"[Scan] Could not stat {name}: {e}")
						disk_files.add(path)
						continue

					disk_files.add(path)
					self._known_mtimes[path] = mtime

					if path not in db_state:
						self._register_file(path, mtime)
						new_count += 1
						continue

					known_mtime, known_modality = db_state[path]
					if abs(mtime - known_mtime) > 1.0:
						self._register_file(path, mtime)
						modified_count += 1
					elif known_modality != get_modality(Path(path).suffix.lower()):
						self._register_file(path, mtime)
						reclassified_count += 1

		# Ghost cleanup — files in DB but not on disk
		ghost_count = 0
		if walk_ok:
			for db_path in db_state:
				if db_path not in disk_files:
					logger.info(f"[Scan] Removed: {Path(db_path).name}")
					self.orchestrator.on_file_deleted(db_path)
					ghost_count += 1
		else:
			logger.warning(
				"Skipping ghost cleanup: the walk could not read every "
				"directory, so a deleted file cannot be told from an "
				"unreadable one."
			)

		elapsed = time.time() - t0
		logger.info(
			f"Initial scan complete: {len(disk_files)} files on disk, "
			f"{new_count} new, {modified_count} modified, "
			f"{reclassified_count} reclassified, {ghost_count} ghosts removed "
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
		# Live watcher writes are background — yield the DB lock to the conversation.
		set_thread_priority_low()
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
		set_thread_priority_low()
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
		if self.rules.ignores_extension(p.suffix):
			return False

		return p.suffix.lower() in self.supported_extensions

	def _is_ignored(self, path: str) -> bool:
		"""Return whether a directory is excluded by the current ignore rules."""
		return self.rules.ignores_folder(path)


class DebouncedHandler(FileSystemEventHandler):
	"""
	Watchdog fires multiple events per file change. This handler
	debounces them: it waits for 1 second of silence on a path
	before forwarding to the watcher.

	Deletes are NOT debounced — they fire immediately.
	"""

	def __init__(self, watcher: Watcher):
		"""Initialize the debounced handler."""
		self.watcher = watcher
		self.debounce_interval = 1.0
		self.pending: dict[str, threading.Timer] = {}
		self.lock = threading.Lock()

	def _debounce(self, path: str):
		"""Internal helper to handle debounce."""
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
		"""Handle on created."""
		self._debounce(event.src_path)

	def on_modified(self, event):
		"""Handle on modified."""
		if event.is_directory:
			return
		self._debounce(event.src_path)

	def on_moved(self, event):
		# Source is gone, destination is new
		"""Handle on moved."""
		self.watcher.handle_delete(event.src_path)
		self._debounce(event.dest_path)

	def on_deleted(self, event):
		# No debounce — delete immediately
		"""Handle on deleted."""
		self.watcher.handle_delete(event.src_path)

