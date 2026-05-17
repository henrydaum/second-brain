"""Pipeline support for database."""

import logging
import re
import sqlite3
import threading
import time

from paths import ATTACHMENT_CACHE

logger = logging.getLogger("Database")

# Paths under these roots are enqueued with elevated priority so the user
# doesn't wait behind a backlog for files they just handed the agent.
_PRIORITY_ROOTS: tuple[str, ...] = (str(ATTACHMENT_CACHE).rstrip("\\/"),)
_HIGH_PRIORITY = 100
_DEFAULT_PRIORITY = 0


def _priority_for(path: str) -> int:
    """Internal helper to handle priority for."""
    norm = str(path).replace("\\", "/").rstrip("/")
    for root in _PRIORITY_ROOTS:
        root_n = root.replace("\\", "/").rstrip("/")
        if norm == root_n or norm.startswith(root_n + "/"):
            return _HIGH_PRIORITY
    return _DEFAULT_PRIORITY

_VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

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
	"""Database."""
	def __init__(self, db_path: str):
		"""Initialize the database."""
		self.db_path = db_path
		self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
		self.conn.row_factory = sqlite3.Row  # dict-like access on rows
		self.lock = threading.Lock()
		self._setup()

	@staticmethod
	def _validate_identifier(name: str):
		"""Ensure a table/column name is a safe SQL identifier."""
		if not _VALID_IDENTIFIER.match(name):
			raise ValueError(f"Invalid SQL identifier: {name!r}")

	def _setup(self):
		# WAL mode allows concurrent readers while one writer holds the lock —
		# critical for the dispatch loop reading while workers write results.
		"""Internal helper to handle setup."""
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

		# Task queue — one row per (file, task) pair
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS task_queue (
				path         TEXT,
				task_name    TEXT,
				status       TEXT DEFAULT 'PENDING',
				priority     INTEGER DEFAULT 0,
				created_at   REAL,
				started_at   REAL,
				completed_at REAL,
				error        TEXT,
				PRIMARY KEY (path, task_name)
			)
		""")
		self.conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_queue_dispatch
			ON task_queue (task_name, status, priority DESC, created_at)
		""")

		# Task registry — remembers which tasks are registered across restarts
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS registered_tasks (
				task_name    TEXT PRIMARY KEY,
				writes       TEXT,
				reads        TEXT,
				modalities   TEXT,
				trigger      TEXT DEFAULT 'path',
				trigger_channels TEXT DEFAULT ''
			)
		""")
		# Event-task runs — one row per triggered run for trigger="event" tasks.
		# Path-keyed tasks use task_queue; event-keyed tasks use this table.
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS task_runs (
				run_id        TEXT PRIMARY KEY,
				task_name     TEXT NOT NULL,
				status        TEXT NOT NULL DEFAULT 'PENDING',
				triggered_by  TEXT,
				parent_run_id TEXT,
				payload_json  TEXT,
				created_at    REAL,
				started_at    REAL,
				finished_at   REAL,
				error         TEXT
			)
		""")
		self.conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_runs_dispatch
			ON task_runs (task_name, status)
		""")

		# Conversation history — persists agent chat sessions
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS conversations (
				id          INTEGER PRIMARY KEY AUTOINCREMENT,
				title       TEXT,
				kind        TEXT DEFAULT 'user',
				category    TEXT,
				created_at  REAL,
				updated_at  REAL
			)
		""")
		# Migration: add last_title_check_message_count if missing
		try:
			self.conn.execute("ALTER TABLE conversations ADD COLUMN last_title_check_message_count INTEGER")
			self.conn.commit()
		except Exception:
			pass
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

		# Web users — anonymous now (user_id = sha256(ip|session_id)[:24]),
		# upgradeable to real auth later. `credits` and `tier` are pre-wired
		# for the monetization work.
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS web_users (
				user_id     TEXT PRIMARY KEY,
				session_id  TEXT,
				ip_hash     TEXT,
				created_at  REAL,
				last_seen   REAL,
				tier        TEXT DEFAULT 'free',
				credits     INTEGER DEFAULT 0
			)
		""")
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS web_usage_events (
				id       INTEGER PRIMARY KEY,
				user_id  TEXT NOT NULL,
				ts       REAL NOT NULL
			)
		""")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_web_usage_user_ts ON web_usage_events(user_id, ts)")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_web_usage_ts ON web_usage_events(ts)")

		# Monetization: account columns on web_users, plus auth tokens,
		# promo codes, and payments. ALTERs are guarded — sqlite raises if the
		# column already exists, so we swallow that specific error.
		for col, decl in (("email", "TEXT"), ("account_id", "TEXT")):
			try:
				self.conn.execute(f"ALTER TABLE web_users ADD COLUMN {col} {decl}")
			except sqlite3.OperationalError as e:
				if "duplicate column" not in str(e).lower():
					raise
		self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_web_users_email ON web_users(email) WHERE email IS NOT NULL")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_web_users_account ON web_users(account_id)")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_web_users_ip ON web_users(ip_hash)")

		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS web_auth_tokens (
				token       TEXT PRIMARY KEY,
				email       TEXT NOT NULL,
				created_at  REAL NOT NULL,
				used_at     REAL
			)
		""")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_web_auth_tokens_email ON web_auth_tokens(email)")

		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS web_promo_codes (
				code         TEXT PRIMARY KEY,
				kind         TEXT NOT NULL,
				credits      INTEGER,
				max_uses     INTEGER DEFAULT 1,
				uses         INTEGER DEFAULT 0,
				created_at   REAL NOT NULL,
				note         TEXT
			)
		""")

		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS web_payments (
				id              INTEGER PRIMARY KEY,
				stripe_event_id TEXT UNIQUE,
				email           TEXT,
				amount_cents    INTEGER,
				credits_granted INTEGER,
				ts              REAL
			)
		""")

		# Skill scoring — implicit signals (share/download/remix) attributed
		# across chain steps. `generations` is a denominator-only counter.
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS skill_events (
				id              INTEGER PRIMARY KEY,
				ts              REAL NOT NULL,
				kind            TEXT NOT NULL,
				slug            TEXT NOT NULL,
				image_path      TEXT,
				chain_position  TEXT,
				weight          REAL NOT NULL
			)
		""")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_events_slug ON skill_events(slug)")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_skill_events_kind ON skill_events(kind)")
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS skill_scores (
				slug         TEXT PRIMARY KEY,
				shares       REAL DEFAULT 0,
				downloads    REAL DEFAULT 0,
				remixes      REAL DEFAULT 0,
				generations  INTEGER DEFAULT 0,
				updated_at   REAL
			)
		""")

		self.conn.commit()

	# =================================================================
	# FILES
	# =================================================================

	def upsert_file(self, path, file_name, extension, modality, mtime, source="watched"):
		"""Handle upsert file."""
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
		prio = _priority_for(path)
		with self.lock:
			self.conn.execute("""
				INSERT OR IGNORE INTO task_queue (path, task_name, status, priority, created_at)
				VALUES (?, ?, 'PENDING', ?, ?)
			""", (path, task_name, prio, now))
			self.conn.commit()

	def re_enqueue_task(self, path, task_name):
		"""Enqueue a task, resetting it to PENDING if it already exists."""
		now = time.time()
		prio = _priority_for(path)
		with self.lock:
			self.conn.execute("""
				INSERT INTO task_queue (path, task_name, status, priority, created_at)
				VALUES (?, ?, 'PENDING', ?, ?)
				ON CONFLICT(path, task_name) DO UPDATE SET
					status = 'PENDING',
					priority = excluded.priority,
					started_at = NULL,
					completed_at = NULL,
					error = NULL
			""", (path, task_name, prio, now))
			self.conn.commit()

	def claim_tasks(self, task_name, batch_size):
		"""Atomically grab up to N PENDING tasks. Returns list of paths. Can be used with batch_size=1 for single tasks.

		Higher-priority paths (e.g. attachments dropped in via the frontend) are
		claimed first. Ties are broken by creation time (FIFO) so normal workload
		still flows in order.
		"""
		with self.lock:
			cur = self.conn.execute("""
				UPDATE task_queue
				SET status = 'PROCESSING', started_at = ?
				WHERE rowid IN (
					SELECT rowid FROM task_queue
					WHERE task_name = ? AND status = 'PENDING'
					ORDER BY priority DESC, created_at ASC
					LIMIT ?
				)
				RETURNING path
			""", (time.time(), task_name, batch_size))
			rows = cur.fetchall()
			self.conn.commit()
			return [row["path"] for row in rows]

	def complete_task(self, path, task_name):
		"""Handle complete task."""
		with self.lock:
			self.conn.execute("""
				UPDATE task_queue
				SET status = 'DONE', completed_at = ?
				WHERE path = ? AND task_name = ?
			""", (time.time(), path, task_name))
			self.conn.commit()

	def fail_task(self, path, task_name, error=""):
		"""Handle fail task."""
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
		"""Handle reset failed tasks."""
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
	# TASK RUNS (event-triggered tasks)
	# =================================================================

	def create_run(self, run_id, task_name, triggered_by=None,
				   payload_json=None, parent_run_id=None):
		"""Enqueue a new event-task run as PENDING."""
		now = time.time()
		with self.lock:
			self.conn.execute("""
				INSERT INTO task_runs
				(run_id, task_name, status, triggered_by, parent_run_id, payload_json, created_at)
				VALUES (?, ?, 'PENDING', ?, ?, ?, ?)
			""", (run_id, task_name, triggered_by, parent_run_id, payload_json, now))
			self.conn.commit()

	def claim_runs(self, task_name, batch_size=1):
		"""Atomically grab up to N PENDING runs. Returns list of (run_id, payload_json)."""
		with self.lock:
			cur = self.conn.execute("""
				UPDATE task_runs
				SET status = 'PROCESSING', started_at = ?
				WHERE run_id IN (
					SELECT run_id FROM task_runs
					WHERE task_name = ? AND status = 'PENDING'
					LIMIT ?
				)
				RETURNING run_id, payload_json
			""", (time.time(), task_name, batch_size))
			rows = cur.fetchall()
			self.conn.commit()
			return [(row["run_id"], row["payload_json"]) for row in rows]

	def complete_run(self, run_id):
		"""Handle complete run."""
		with self.lock:
			self.conn.execute("""
				UPDATE task_runs
				SET status = 'DONE', finished_at = ?
				WHERE run_id = ?
			""", (time.time(), run_id))
			self.conn.commit()

	def fail_run(self, run_id, error=""):
		"""Handle fail run."""
		with self.lock:
			self.conn.execute("""
				UPDATE task_runs
				SET status = 'FAILED', finished_at = ?, error = ?
				WHERE run_id = ?
			""", (time.time(), error, run_id))
			self.conn.commit()

	def unclaim_run(self, run_id):
		"""Return a claimed run to PENDING (e.g. task was paused after claim)."""
		with self.lock:
			self.conn.execute("""
				UPDATE task_runs
				SET status = 'PENDING', started_at = NULL
				WHERE run_id = ?
			""", (run_id,))
			self.conn.commit()

	def reset_stuck_runs_for(self, task_name: str, timeout_seconds: int) -> int:
		"""Reset PROCESSING runs back to PENDING if running longer than timeout."""
		cutoff = time.time() - timeout_seconds
		with self.lock:
			cur = self.conn.execute("""
				UPDATE task_runs
				SET status = 'PENDING', started_at = NULL
				WHERE task_name = ? AND status = 'PROCESSING' AND started_at < ?
			""", (task_name, cutoff))
			self.conn.commit()
			return cur.rowcount

	def get_runs(self, task_name=None, limit=50):
		"""List recent runs, newest first, optionally filtered by task."""
		with self.lock:
			if task_name:
				cur = self.conn.execute("""
					SELECT * FROM task_runs
					WHERE task_name = ?
					ORDER BY created_at DESC LIMIT ?
				""", (task_name, limit))
			else:
				cur = self.conn.execute("""
					SELECT * FROM task_runs
					ORDER BY created_at DESC LIMIT ?
				""", (limit,))
			return [dict(row) for row in cur.fetchall()]

	def get_run_stats(self):
		"""Return per-event-task status counts from task_runs."""
		with self.lock:
			cur = self.conn.execute("""
				SELECT task_name, status, COUNT(*) as count
				FROM task_runs
				GROUP BY task_name, status
			""")
			run_stats = {}
			for row in cur.fetchall():
				name = row["task_name"]
				if name not in run_stats:
					run_stats[name] = {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0}
				run_stats[name][row["status"]] = row["count"]
			return run_stats

	# =================================================================
	# OUTPUT TABLES
	# =================================================================

	def clean_output_tables(self, path, table_names):
		"""Remove a file's data from path-keyed output tables."""
		for table in table_names:
			self._validate_identifier(table)
		with self.lock:
			for table in dict.fromkeys(table_names):
				try:
					if not self._table_has_column_unlocked(table, "path"):
						logger.debug(f"Skipping cleanup for '{table}' (no path column)")
						continue
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
		Returns True when a trigger was created or already existed; False when
		either table is not path-keyed and should be skipped.
		"""
		self._validate_identifier(upstream_table)
		self._validate_identifier(downstream_table)
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
			if (
				not self._table_has_column_unlocked(upstream_table, "path")
				or not self._table_has_column_unlocked(downstream_table, "path")
			):
				return False
			self.conn.execute(sql)
			self.conn.commit()
			return True

	def _table_has_column_unlocked(self, table_name: str, column_name: str) -> bool:
		"""Return whether a table has a column. Caller must hold self.lock."""
		self._validate_identifier(table_name)
		self._validate_identifier(column_name)
		cur = self.conn.execute(f"PRAGMA table_info({table_name})")
		return any(row["name"] == column_name for row in cur.fetchall())

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
		self._validate_identifier(table_name)
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
		self._validate_identifier(table_name)
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
		self._validate_identifier(table_name)
		with self.lock:
			try:
				self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
				self.conn.commit()
			except sqlite3.Error as e:
				logger.error(f"Failed to drop {table_name}: {e}")

	# =================================================================
	# TASK REGISTRATION
	# =================================================================

	def register_task(self, name, writes, reads, modalities, trigger="path", trigger_channels=None):
		"""Persist task metadata across restarts."""
		with self.lock:
			self.conn.execute("""
				INSERT OR REPLACE INTO registered_tasks
				(task_name, writes, reads, modalities, trigger, trigger_channels)
				VALUES (?, ?, ?, ?, ?, ?)
			""", (name,
				  ",".join(writes) if writes else "",
				  ",".join(reads) if reads else "",
				  ",".join(modalities) if modalities else "",
				  trigger or "path",
				  ",".join(trigger_channels) if trigger_channels else ""))
			self.conn.commit()

	def get_registered_tasks(self):
		"""Get registered tasks."""
		with self.lock:
			cur = self.conn.execute("SELECT * FROM registered_tasks")
			return [dict(row) for row in cur.fetchall()]

	# =================================================================
	# STATS
	# =================================================================

	def get_system_stats(self):
		"""Get system stats."""
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

	def create_conversation(self, title="New Conversation", kind="user", category=None) -> int:
		"""Create conversation."""
		now = time.time()
		with self.lock:
			cur = self.conn.execute(
				"INSERT INTO conversations (title, kind, category, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
				(title, kind, category, now, now))
			self.conn.commit()
			return cur.lastrowid

	def save_message(self, conversation_id, role, content,
					 tool_call_id=None, tool_name=None):
		"""Save message."""
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
		"""Update conversation title."""
		with self.lock:
			self.conn.execute(
				"UPDATE conversations SET title = ? WHERE id = ?",
				(title, conversation_id))
			self.conn.commit()

	def update_conversation_title_check_count(self, conversation_id, count: int):
		"""Record the message count seen by the last title-update sweep.

		The title-update task uses this together with the live message count
		to decide whether enough new turns have accumulated to justify a
		re-titling LLM call.
		"""
		with self.lock:
			self.conn.execute(
				"UPDATE conversations SET last_title_check_message_count = ? WHERE id = ?",
				(int(count), conversation_id))
			self.conn.commit()

	def list_conversations_for_title_check(self, threshold: int = 4) -> list[dict]:
		"""Return conversations whose unseen-message delta meets ``threshold``.

		Each row carries ``id``, ``title``, ``message_count``, and
		``last_title_check_message_count`` so the caller can decide which
		ones to re-title and persist the new high-water mark.
		"""
		with self.lock:
			cur = self.conn.execute(
				"""
				SELECT c.id            AS id,
				       c.title         AS title,
				       COALESCE(c.last_title_check_message_count, 0) AS last_title_check_message_count,
				       (SELECT COUNT(*) FROM conversation_messages m
				          WHERE m.conversation_id = c.id) AS message_count
				FROM conversations c
				WHERE (
				    (SELECT COUNT(*) FROM conversation_messages m
				        WHERE m.conversation_id = c.id)
				    - COALESCE(c.last_title_check_message_count, 0)
				) >= ?
				ORDER BY c.updated_at DESC
				""",
				(int(threshold),))
			return [dict(row) for row in cur.fetchall()]

	def set_conversation_category(self, conversation_id, category):
		"""Set/overwrite the category on a conversation row."""
		with self.lock:
			self.conn.execute(
				"UPDATE conversations SET category = ? WHERE id = ?",
				(category, conversation_id))
			self.conn.commit()

	def get_conversation(self, conversation_id):
		"""Get conversation."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM conversations WHERE id = ?",
				(conversation_id,))
			row = cur.fetchone()
			return dict(row) if row else None

	def list_conversations(self, limit=50):
		"""List conversations."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
				(limit,))
			return [dict(row) for row in cur.fetchall()]

	def list_conversations_page(self, offset=0, limit=10, category=None) -> tuple[list[dict], bool]:
		"""Return ``(rows, has_more)`` sorted by most-recent activity.

		``category``:
		    - None → no filter (every conversation).
		    - "" → conversations with NULL/empty category (the "Main" bucket).
		    - any other string → exact match on the category column.
		"""
		params: list = []
		where = ""
		if category is not None:
			if category == "":
				where = "WHERE category IS NULL OR category = ''"
			else:
				where = "WHERE category = ?"
				params.append(category)
		params += [limit + 1, offset]
		with self.lock:
			cur = self.conn.execute(
				f"SELECT * FROM conversations {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
				params)
			rows = [dict(row) for row in cur.fetchall()]
		has_more = len(rows) > limit
		return rows[:limit], has_more

	def list_conversation_categories(self) -> list[str | None]:
		"""Distinct category values present in the conversations table.

		``None`` is included if any row has NULL/empty category.
		"""
		with self.lock:
			cur = self.conn.execute(
				"SELECT DISTINCT category FROM conversations")
			values = [row["category"] for row in cur.fetchall()]
		out: list[str | None] = []
		seen_main = False
		for v in values:
			if v in (None, ""):
				if not seen_main:
					out.append(None)
					seen_main = True
			elif v not in out:
				out.append(v)
		return out

	def list_user_conversations(self, limit=50):
		"""List user conversations."""
		with self.lock:
			cur = self.conn.execute(
				"""
				SELECT *
				FROM conversations c
				ORDER BY c.updated_at DESC
				LIMIT ?
				""",
				(limit,))
			return [dict(row) for row in cur.fetchall()]

	def get_conversation_messages(self, conversation_id):
		"""Get conversation messages."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM conversation_messages WHERE conversation_id = ? ORDER BY timestamp",
				(conversation_id,))
			return [dict(row) for row in cur.fetchall()]

	def replace_conversation_messages(self, conversation_id, history: list[dict]) -> None:
		"""Atomically replace a conversation's persisted messages with `history`.

		`history` is in provider-message shape: list of {role, content, ...} dicts.
		Assistant turns with tool_calls get JSON-packed into the content column.
		"""
		import json as _json
		base = time.time()
		rows = []
		for i, msg in enumerate(history):
			role = msg.get("role")
			if role not in {"user", "assistant", "tool"}:
				continue
			content = msg.get("content") or ""
			if role == "assistant" and msg.get("tool_calls"):
				content = _json.dumps({
					"content": msg.get("content"),
					"tool_calls": msg["tool_calls"],
				})
			# Stagger timestamps so ORDER BY timestamp preserves insertion order.
			ts = base + i * 0.001
			rows.append((
				conversation_id, role, content,
				msg.get("tool_call_id"), msg.get("name"),
				ts,
			))
		with self.lock:
			self.conn.execute(
				"DELETE FROM conversation_messages WHERE conversation_id = ?",
				(conversation_id,))
			if rows:
				self.conn.executemany("""
					INSERT INTO conversation_messages
					(conversation_id, role, content, tool_call_id, tool_name, timestamp)
					VALUES (?, ?, ?, ?, ?, ?)
				""", rows)
			self.conn.execute(
				"UPDATE conversations SET updated_at = ? WHERE id = ?",
				(time.time(), conversation_id))
			self.conn.commit()


	def clear_conversation_messages(self, conversation_id):
		"""Clear conversation messages."""
		with self.lock:
			self.conn.execute(
				"DELETE FROM conversation_messages WHERE conversation_id = ?",
				(conversation_id,))
			self.conn.commit()

	def delete_conversation(self, conversation_id):
		"""Delete conversation."""
		with self.lock:
			self.conn.execute(
				"DELETE FROM conversations WHERE id = ?", (conversation_id,))
			self.conn.commit()

	def delete_all_conversations(self):
		"""Delete all conversations."""
		with self.lock:
			self.conn.execute("DELETE FROM conversation_messages")
			self.conn.execute("DELETE FROM conversations")
			self.conn.commit()

	def conversation_message_count(self, conversation_id) -> int:
		"""Handle conversation message count."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT COUNT(*) as cnt FROM conversation_messages WHERE conversation_id = ?",
				(conversation_id,))
			return cur.fetchone()["cnt"]
