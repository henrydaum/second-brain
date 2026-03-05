import logging
import sqlite3
import threading
import time

logger = logging.getLogger("Database")

"""
Database for the task pipeline.

Two fixed tables:
	files       — one row per discovered file (crawler writes here)
	task_queue  — one row per (file, task) pair (orchestrator writes here)

Dynamic output tables:
	Created by tasks via raw SQL. Each task owns its own schema.

Search index:
	FTS5 table fed by tasks. Built out later.
"""


class Database:
	def __init__(self, db_path: str):
		self.db_path = db_path
		self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
		self.conn.row_factory = sqlite3.Row  # dict-like access on rows
		self.lock = threading.Lock()
		self._setup()

	def _setup(self):
		self.conn.execute("PRAGMA journal_mode=WAL")
		self.conn.execute("PRAGMA cache_size=-50000")

		# Master file registry — one row per file on disk
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS files (
				path          TEXT PRIMARY KEY,
				file_name     TEXT,
				extension     TEXT,
				modality      TEXT,
				mtime         REAL,
				discovered_at REAL,
				updated_at    REAL
			)
		""")

		# Task queue — one row per (file, task) pair
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS task_queue (
				path         TEXT,
				task_name    TEXT,
				task_version INTEGER DEFAULT 1,
				status       TEXT DEFAULT 'PENDING',
				created_at   REAL,
				started_at   REAL,
				completed_at REAL,
				error        TEXT,
				PRIMARY KEY (path, task_name)
			)
		""")
		self.conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_queue_dispatch
			ON task_queue (task_name, status)
		""")

		# Task registry — remembers which tasks are registered across restarts
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS registered_tasks (
				task_name    TEXT PRIMARY KEY,
				task_version INTEGER,
				output_table TEXT,
				modalities   TEXT,
				depends_on   TEXT
			)
		""")

		self.conn.commit()

	# =================================================================
	# FILES
	# =================================================================

	def upsert_file(self, path, file_name, extension, modality, mtime):
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT INTO files (path, file_name, extension, modality, mtime, discovered_at, updated_at)
				VALUES (?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(path) DO UPDATE SET
					mtime = excluded.mtime,
					updated_at = excluded.updated_at
			""", (path, file_name, extension, modality, mtime, now, now))
			self.conn.commit()

	def remove_file(self, path):
		"""Remove a file and all its task queue entries. Output table cleanup is caller's job."""
		with self.lock:
			self.conn.execute("DELETE FROM task_queue WHERE path = ?", (path,))
			self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
			self.conn.commit()

	def get_all_files(self):
		"""Returns {path: mtime} for diffing against disk."""
		with self.lock:
			cur = self.conn.execute("SELECT path, mtime FROM files")
			return {row["path"]: row["mtime"] for row in cur.fetchall()}

	def get_files_by_modality(self, modality):
		"""Returns list of paths for a given modality."""
		with self.lock:
			cur = self.conn.execute("SELECT path FROM files WHERE modality = ?", (modality,))
			return [row["path"] for row in cur.fetchall()]

	# =================================================================
	# TASK QUEUE
	# =================================================================

	def enqueue_task(self, path, task_name, task_version=1):
		"""Add a task to the queue. Skips if already exists."""
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT OR IGNORE INTO task_queue (path, task_name, task_version, status, created_at)
				VALUES (?, ?, ?, 'PENDING', ?)
			""", (path, task_name, task_version, now))
			self.conn.commit()

	def re_enqueue_task(self, path, task_name, task_version=1):
		"""Enqueue a task, resetting it to PENDING if it already exists."""
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT INTO task_queue (path, task_name, task_version, status, created_at)
				VALUES (?, ?, ?, 'PENDING', ?)
				ON CONFLICT(path, task_name) DO UPDATE SET
					status = 'PENDING',
					task_version = excluded.task_version,
					started_at = NULL,
					completed_at = NULL,
					error = NULL
			""", (path, task_name, task_version, now))
			self.conn.commit()

	def claim_tasks(self, task_name, batch_size):
		"""Atomically grab up to N PENDING tasks. Returns list of paths. Can be used with batch_size=1 for single tasks."""
		with self.lock:
			cur = self.conn.execute("""
				UPDATE task_queue
				SET status = 'PROCESSING', started_at = ?
				WHERE rowid IN (
					SELECT rowid FROM task_queue
					WHERE task_name = ? AND status = 'PENDING'
					LIMIT ?
				)
				RETURNING path
			""", (time.time(), task_name, batch_size))
			rows = cur.fetchall()
			self.conn.commit()
			return [row["path"] for row in rows]

	def complete_task(self, path, task_name):
		with self.lock:
			self.conn.execute("""
				UPDATE task_queue
				SET status = 'DONE', completed_at = ?
				WHERE path = ? AND task_name = ?
			""", (time.time(), path, task_name))
			self.conn.commit()

	def fail_task(self, path, task_name, error=""):
		with self.lock:
			self.conn.execute("""
				UPDATE task_queue
				SET status = 'FAILED', completed_at = ?, error = ?
				WHERE path = ? AND task_name = ?
			""", (time.time(), error, path, task_name))
			self.conn.commit()

	def is_task_done(self, path, task_name):
		"""Check if a specific task is done for a file. Used for dependency checks."""
		with self.lock:
			cur = self.conn.execute("""
				SELECT status FROM task_queue
				WHERE path = ? AND task_name = ?
			""", (path, task_name))
			row = cur.fetchone()
			return row["status"] == "DONE" if row else False

	def get_pending_tasks(self, task_name=None):
		"""Get all pending tasks, optionally filtered by task name."""
		with self.lock:
			if task_name:
				cur = self.conn.execute(
					"SELECT path, task_name FROM task_queue WHERE status = 'PENDING' AND task_name = ?",
					(task_name,))
			else:
				cur = self.conn.execute(
					"SELECT path, task_name FROM task_queue WHERE status = 'PENDING'")
			return [(row["path"], row["task_name"]) for row in cur.fetchall()]

	def reset_stuck_tasks(self, timeout_seconds=300):
		"""Reset tasks stuck in PROCESSING back to PENDING."""
		cutoff = time.time() - timeout_seconds
		with self.lock:
			self.conn.execute("""
				UPDATE task_queue
				SET status = 'PENDING', started_at = NULL
				WHERE status = 'PROCESSING' AND started_at < ?
			""", (cutoff,))
			self.conn.commit()

	def reset_failed_tasks(self, task_name=None):
		with self.lock:
			if task_name:
				self.conn.execute(
					"UPDATE task_queue SET status = 'PENDING', error = NULL WHERE status = 'FAILED' AND task_name = ?",
					(task_name,))
			else:
				self.conn.execute(
					"UPDATE task_queue SET status = 'PENDING', error = NULL WHERE status = 'FAILED'")
			self.conn.commit()

	def reset_task(self, task_name):
		"""Reset all entries for a task back to PENDING. Used when version changes."""
		with self.lock:
			self.conn.execute("""
				UPDATE task_queue 
				SET status = 'PENDING', started_at = NULL, completed_at = NULL, error = NULL
				WHERE task_name = ?
			""", (task_name,))
			self.conn.commit()

	# =================================================================
	# OUTPUT TABLES
	# =================================================================

	def clean_output_tables(self, path, table_names):
		"""Remove a file's data from multiple output tables."""
		with self.lock:
			for table in table_names:
				try:
					self.conn.execute(f"DELETE FROM {table} WHERE path = ?", (path,))
				except sqlite3.OperationalError:
					pass  # table might not exist yet
			self.conn.commit()

	def ensure_output_table(self, task_name, schema_sql):
		"""
		Execute a task's schema SQL. Only CREATE TABLE and CREATE INDEX allowed.
		The task owns its schema — it provides raw SQL. 
		
		Takes raw SQL that can contain multiple CREATE TABLE and CREATE INDEX statements separated by semicolons.
		"""
		allowed_prefixes = ("create table", "create index", "create unique index")
		statements = [s.strip() for s in schema_sql.split(";") if s.strip()]

		for stmt in statements:
			normalized = " ".join(stmt.lower().split())
			if not any(normalized.startswith(p) for p in allowed_prefixes):
				raise ValueError(
					f"Task '{task_name}' schema contains disallowed SQL: {stmt[:80]}"
				)

		with self.lock:
			try:
				self.conn.executescript(schema_sql)
				self.conn.commit()
			except sqlite3.Error as e:
				logger.error(f"Schema creation failed for '{task_name}': {e}")
				raise

	def write_outputs(self, table_name, rows):
		"""Batch insert. rows is a list of dicts (all same keys)."""
		if not rows:
			return
		columns = ", ".join(rows[0].keys())
		placeholders = ", ".join("?" * len(rows[0]))
		with self.lock:
			self.conn.executemany(
				f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})",
				[list(row.values()) for row in rows])
			self.conn.commit()

	def get_task_output(self, table_name, path):
		"""Retrieve output for a single file from any output table."""
		with self.lock:
			try:
				cur = self.conn.execute(
					f"SELECT * FROM {table_name} WHERE path = ?", (path,))
				rows = cur.fetchall()
				return [dict(row) for row in rows]
			except sqlite3.OperationalError:
				return []

	def drop_task_data(self, table_name):
		"""Nuclear option — drop an entire output table."""
		with self.lock:
			try:
				self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
				self.conn.commit()
			except sqlite3.Error as e:
				logger.error(f"Failed to drop {table_name}: {e}")

	# =================================================================
	# TASK REGISTRATION
	# =================================================================

	def register_task(self, name, version, output_table, modalities, depends_on):
		"""Persist task metadata so we can detect version changes across restarts."""
		with self.lock:
			self.conn.execute("""
				INSERT OR REPLACE INTO registered_tasks
				(task_name, task_version, output_table, modalities, depends_on)
				VALUES (?, ?, ?, ?, ?)
			""", (name, version, output_table,
				  ",".join(modalities) if modalities else "",
				  ",".join(depends_on) if depends_on else ""))
			self.conn.commit()

	def get_registered_tasks(self):
		with self.lock:
			cur = self.conn.execute("SELECT * FROM registered_tasks")
			return [dict(row) for row in cur.fetchall()]

	def has_version_changed(self, name, version):
		with self.lock:
			cur = self.conn.execute(
				"SELECT task_version FROM registered_tasks WHERE task_name = ?", (name,))
			row = cur.fetchone()
			if row is None:
				return True  # new task, never seen before
			return row["task_version"] != version

	# =================================================================
	# STATS
	# =================================================================

	def get_system_stats(self):
		with self.lock:
			# File counts by modality
			cur = self.conn.execute(
				"SELECT modality, COUNT(*) as count FROM files GROUP BY modality")
			file_stats = {row["modality"]: row["count"] for row in cur.fetchall()}

			# Task counts by name and status
			cur = self.conn.execute(
				"SELECT task_name, status, COUNT(*) as count FROM task_queue GROUP BY task_name, status")
			task_stats = {}
			for row in cur.fetchall():
				name = row["task_name"]
				if name not in task_stats:
					task_stats[name] = {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0}
				task_stats[name][row["status"]] = row["count"]

			return {"files": file_stats, "tasks": task_stats}
		
		