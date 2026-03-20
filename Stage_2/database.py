import logging
import sqlite3
import threading
import time

logger = logging.getLogger("Database")

"""
Database for the task pipeline.

Fixed tables:
	files                  — one row per discovered file (crawler writes here)
	task_queue             — one row per (file, task) pair (orchestrator writes here)
	conversations          — one row per chat conversation
	conversation_messages  — one row per message in a conversation

Dynamic output tables:
	Created by tasks via raw SQL. Each task owns its own schema.
	Supports CREATE TABLE, CREATE INDEX, CREATE VIRTUAL TABLE,
	and CREATE TRIGGER (for FTS5 content-sync).
"""


class Database:
	def __init__(self, db_path: str):
		self.db_path = db_path
		self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
		self.conn.row_factory = sqlite3.Row  # dict-like access on rows
		self.lock = threading.Lock()
		self._setup()

	def _setup(self):
		# WAL mode allows concurrent readers while one writer holds the lock —
		# critical for the dispatch loop reading while workers write results.
		self.conn.execute("PRAGMA journal_mode=WAL")
		# Negative value = KB. -50000 ≈ 50 MB page cache (default is ~2 MB).
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
				updated_at    REAL,
				source        TEXT DEFAULT 'watched'
			)
		""")

		# Migration: add source column for existing databases
		try:
			self.conn.execute("ALTER TABLE files ADD COLUMN source TEXT DEFAULT 'watched'")
		except sqlite3.OperationalError:
			pass  # column already exists

		# Task queue — one row per (file, task) pair
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS task_queue (
				path         TEXT,
				task_name    TEXT,
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
		# Drop and recreate to handle schema migrations cleanly
		self.conn.execute("DROP TABLE IF EXISTS registered_tasks")
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS registered_tasks (
				task_name    TEXT PRIMARY KEY,
				writes       TEXT,
				reads        TEXT,
				modalities   TEXT
			)
		""")

		# Conversation history — persists agent chat sessions
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS conversations (
				id          INTEGER PRIMARY KEY AUTOINCREMENT,
				title       TEXT,
				created_at  REAL,
				updated_at  REAL
			)
		""")
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS conversation_messages (
				id              INTEGER PRIMARY KEY AUTOINCREMENT,
				conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
				role            TEXT,
				content         TEXT,
				tool_call_id    TEXT,
				tool_name       TEXT,
				timestamp       REAL
			)
		""")
		self.conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_conv_msg_conv
			ON conversation_messages(conversation_id)
		""")
		# Enable foreign key enforcement (needed for ON DELETE CASCADE)
		self.conn.execute("PRAGMA foreign_keys = ON")

		self.conn.commit()

	# =================================================================
	# FILES
	# =================================================================

	def upsert_file(self, path, file_name, extension, modality, mtime, source="watched"):
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT INTO files (path, file_name, extension, modality, mtime, discovered_at, updated_at, source)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(path) DO UPDATE SET
					mtime = excluded.mtime,
					updated_at = excluded.updated_at
			""", (path, file_name, extension, modality, mtime, now, now, source))
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

	def get_watched_files(self):
		"""Returns {path: mtime} for watched files only (excludes container-extracted)."""
		with self.lock:
			cur = self.conn.execute("SELECT path, mtime FROM files WHERE source = 'watched'")
			return {row["path"]: row["mtime"] for row in cur.fetchall()}

	def get_container_children(self, extract_dir):
		"""Returns list of paths for files extracted from a container (under extract_dir)."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT path FROM files WHERE source = 'container' AND path LIKE ?",
				(extract_dir.rstrip("/\\") + "%",)
			)
			return [row["path"] for row in cur.fetchall()]

	def get_files_by_modality(self, modality):
		"""Returns list of paths for a given modality."""
		with self.lock:
			cur = self.conn.execute("SELECT path FROM files WHERE modality = ?", (modality,))
			return [row["path"] for row in cur.fetchall()]

	def get_paths_with_any_task_done(self, task_names):
		"""Returns distinct paths where any of the given tasks are DONE.
		Used by _backfill_tasks for downstream tasks with no modalities."""
		if not task_names:
			return []
		with self.lock:
			placeholders = ",".join("?" * len(task_names))
			cur = self.conn.execute(f"""
				SELECT DISTINCT path FROM task_queue
				WHERE task_name IN ({placeholders}) AND status = 'DONE'
			""", task_names)
			return [row["path"] for row in cur.fetchall()]

	# =================================================================
	# TASK QUEUE
	# =================================================================

	def enqueue_task(self, path, task_name):
		"""Add a task to the queue. Skips if already exists."""
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT OR IGNORE INTO task_queue (path, task_name, status, created_at)
				VALUES (?, ?, 'PENDING', ?)
			""", (path, task_name, now))
			self.conn.commit()

	def re_enqueue_task(self, path, task_name):
		"""Enqueue a task, resetting it to PENDING if it already exists."""
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT INTO task_queue (path, task_name, status, created_at)
				VALUES (?, ?, 'PENDING', ?)
				ON CONFLICT(path, task_name) DO UPDATE SET
					status = 'PENDING',
					started_at = NULL,
					completed_at = NULL,
					error = NULL
			""", (path, task_name, now))
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
	
	def reset_stuck_tasks_for(self, task_name: str, timeout_seconds: int) -> int:
		"""
		Reset PROCESSING entries for a specific task back to PENDING
		if they've been running longer than timeout_seconds.

		Returns the number of rows reset.
		"""
		cutoff = time.time() - timeout_seconds
		with self.lock:
			cur = self.conn.execute("""
				UPDATE task_queue
				SET status = 'PENDING', started_at = NULL
				WHERE task_name = ? AND status = 'PROCESSING' AND started_at < ?
			""", (task_name, cutoff))
			self.conn.commit()
			return cur.rowcount

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
		"""Reset all entries for a task back to PENDING."""
		with self.lock:
			self.conn.execute("""
				UPDATE task_queue
				SET status = 'PENDING', started_at = NULL, completed_at = NULL, error = NULL
				WHERE task_name = ?
			""", (task_name,))
			self.conn.commit()

	def invalidate_tasks_for_paths(self, task_names: list[str], paths: list[str]):
		"""Reset task_queue entries to PENDING for specific (path, task_name) pairs.
		Used to cascade invalidation when an upstream task fails."""
		if not task_names or not paths:
			return
		with self.lock:
			for task_name in task_names:
				self.conn.executemany(
					"""UPDATE task_queue
					   SET status = 'PENDING', started_at = NULL, completed_at = NULL, error = NULL
					   WHERE path = ? AND task_name = ? AND status != 'PENDING'""",
					[(p, task_name) for p in paths]
				)
			self.conn.commit()

	def invalidate_tasks_bulk(self, task_names: list[str]):
		"""Reset ALL entries for given task names to PENDING.
		Used to cascade invalidation when an upstream task is fully reset."""
		if not task_names:
			return
		with self.lock:
			for task_name in task_names:
				self.conn.execute(
					"""UPDATE task_queue
					   SET status = 'PENDING', started_at = NULL, completed_at = NULL, error = NULL
					   WHERE task_name = ? AND status != 'PENDING'""",
					(task_name,)
				)
			self.conn.commit()

	def get_paths_for_task_status(self, task_name: str, status: str) -> list[str]:
		"""Get all paths for a task with a given status."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT path FROM task_queue WHERE task_name = ? AND status = ?",
				(task_name, status))
			return [row["path"] for row in cur.fetchall()]

	# =================================================================
	# OUTPUT TABLES
	# =================================================================

	def clean_output_tables(self, path, table_names):
		"""Remove a file's data from multiple output tables."""
		with self.lock:
			for table in table_names:
				try:
					self.conn.execute(f"DELETE FROM {table} WHERE path = ?", (path,))
				except sqlite3.OperationalError as e:
					if "no such table" not in str(e):
						raise
			self.conn.commit()

	def create_cascade_trigger(self, upstream_table: str, downstream_table: str):
		"""
		Create a SQL trigger that deletes downstream rows when upstream rows
		are deleted. INSERT OR REPLACE fires DELETE triggers in SQLite, so
		this automatically cascades when an upstream task re-runs for a file.
		"""
		trigger_name = f"cascade_delete_{upstream_table}_to_{downstream_table}"
		sql = f"""
			CREATE TRIGGER IF NOT EXISTS {trigger_name}
			AFTER DELETE ON {upstream_table}
			FOR EACH ROW
			BEGIN
				DELETE FROM {downstream_table} WHERE path = OLD.path;
			END;
		"""
		with self.lock:
			self.conn.execute(sql)
			self.conn.commit()

	def unclaim_tasks(self, task_name: str, paths: list[str]):
		"""Return claimed tasks to PENDING when deps aren't met at dispatch time."""
		with self.lock:
			self.conn.executemany(
				"UPDATE task_queue SET status = 'PENDING', started_at = NULL "
				"WHERE path = ? AND task_name = ?",
				[(p, task_name) for p in paths]
			)
			self.conn.commit()

	def ensure_output_table(self, task_name, schema_sql):
		"""
		Execute a task's schema SQL. Only CREATE statements allowed.
		The task owns its schema — it provides raw SQL.

		Takes raw SQL that can contain multiple statements separated by
		semicolons, including CREATE TRIGGER blocks (which contain internal
		semicolons within BEGIN...END).
		"""
		allowed_prefixes = ("create table", "create index", "create unique index",
						   "create virtual table", "create trigger")

		# Trigger bodies contain semicolons inside BEGIN...END, so we can't
		# naively split on ";". Instead, split and then rejoin trigger blocks.
		raw_parts = [s.strip() for s in schema_sql.split(";") if s.strip()]
		statements = []
		current = None
		in_trigger = False

		for part in raw_parts:
			if in_trigger:
				current += ";" + part
				if part.strip().upper() == "END":
					statements.append(current)
					current = None
					in_trigger = False
			else:
				normalized = " ".join(part.lower().split())
				if normalized.startswith("create trigger"):
					in_trigger = True
					current = part
				else:
					statements.append(part)

		if current:
			statements.append(current)

		for stmt in statements:
			normalized = " ".join(stmt.lower().split())
			if not any(normalized.startswith(p) for p in allowed_prefixes):
				raise ValueError(
					f"Task '{task_name}' schema contains disallowed SQL: {stmt[:80]}"
				)

		t0 = time.time()
		with self.lock:
			try:
				self.conn.executescript(schema_sql)
				self.conn.commit()
				logger.debug(
					f"Schema for '{task_name}' ensured ({len(statements)} statements, "
					f"{time.time() - t0:.3f}s)"
				)
			except sqlite3.Error as e:
				logger.error(f"Schema creation failed for '{task_name}': {e}")
				raise

	def write_outputs(self, table_name, rows):
		"""Batch insert. rows is a list of dicts (all same keys)."""
		if not rows:
			return
		columns = ", ".join(rows[0].keys())
		placeholders = ", ".join("?" * len(rows[0]))
		t0 = time.time()
		with self.lock:
			self.conn.executemany(
				f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})",
				[list(row.values()) for row in rows])
			self.conn.commit()
		elapsed = time.time() - t0
		if elapsed > 0.5:
			logger.debug(f"write_outputs: {len(rows)} rows to '{table_name}' in {elapsed:.2f}s")

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

	def register_task(self, name, writes, reads, modalities):
		"""Persist task metadata across restarts."""
		with self.lock:
			self.conn.execute("""
				INSERT OR REPLACE INTO registered_tasks
				(task_name, writes, reads, modalities)
				VALUES (?, ?, ?, ?)
			""", (name,
				  ",".join(writes) if writes else "",
				  ",".join(reads) if reads else "",
				  ",".join(modalities) if modalities else ""))
			self.conn.commit()

	def get_registered_tasks(self):
		with self.lock:
			cur = self.conn.execute("SELECT * FROM registered_tasks")
			return [dict(row) for row in cur.fetchall()]

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
		
	# =================================================================
	# DIRECT QUERY
	# =================================================================
	def query(self, sql: str, max_rows: int = 25) -> dict:
		"""
		Execute a read-only SQL query and return results.

		Returns:
			{
				"columns":   list of column names,
				"rows":      list of tuples,
				"truncated": bool — True if results were capped at max_rows,
			}

		Raises ValueError for non-SELECT statements.
		Raises sqlite3.Error for invalid SQL.
		"""
		normalized = " ".join(sql.strip().split()).lower()
		if not (normalized.startswith("select") or normalized.startswith("pragma")):
			raise ValueError("Only SELECT and PRAGMA statements are allowed.")

		with self.lock:
			cur = self.conn.execute(sql)
			columns = [desc[0] for desc in cur.description] if cur.description else []
			rows = cur.fetchmany(max_rows + 1)

			truncated = len(rows) > max_rows
			if truncated:
				rows = rows[:max_rows]

			return {
				"columns": columns,
				"rows": [tuple(row) for row in rows],
				"truncated": truncated,
			}

	# =================================================================
	# CONVERSATIONS
	# =================================================================

	def create_conversation(self, title="New conversation") -> int:
		now = time.time()
		with self.lock:
			cur = self.conn.execute(
				"INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
				(title, now, now))
			self.conn.commit()
			return cur.lastrowid

	def save_message(self, conversation_id, role, content,
					 tool_call_id=None, tool_name=None):
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT INTO conversation_messages
				(conversation_id, role, content, tool_call_id, tool_name, timestamp)
				VALUES (?, ?, ?, ?, ?, ?)
			""", (conversation_id, role, content, tool_call_id, tool_name, now))
			self.conn.execute(
				"UPDATE conversations SET updated_at = ? WHERE id = ?",
				(now, conversation_id))
			self.conn.commit()

	def update_conversation_title(self, conversation_id, title):
		with self.lock:
			self.conn.execute(
				"UPDATE conversations SET title = ? WHERE id = ?",
				(title, conversation_id))
			self.conn.commit()

	def list_conversations(self, limit=50):
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
				(limit,))
			return [dict(row) for row in cur.fetchall()]

	def get_conversation_messages(self, conversation_id):
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM conversation_messages WHERE conversation_id = ? ORDER BY timestamp",
				(conversation_id,))
			return [dict(row) for row in cur.fetchall()]

	def delete_conversation(self, conversation_id):
		with self.lock:
			self.conn.execute(
				"DELETE FROM conversations WHERE id = ?", (conversation_id,))
			self.conn.commit()

	def delete_all_conversations(self):
		with self.lock:
			self.conn.execute("DELETE FROM conversation_messages")
			self.conn.execute("DELETE FROM conversations")
			self.conn.commit()

	def conversation_message_count(self, conversation_id) -> int:
		with self.lock:
			cur = self.conn.execute(
				"SELECT COUNT(*) as cnt FROM conversation_messages WHERE conversation_id = ?",
				(conversation_id,))
			return cur.fetchone()["cnt"]