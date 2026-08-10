"""Pipeline support for database."""

import json
import logging
import re
import sqlite3
import threading
import time
from contextlib import contextmanager

import trees

from . import sql_functions

logger = logging.getLogger("Database")

# Paths under these roots are enqueued with elevated priority so the user
# doesn't wait behind a backlog for files they just handed the agent.
_PRIORITY_ROOTS: tuple[str, ...] = (str(trees.attachment_cache()).rstrip("\\/"),)
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

# Authorizer *action* codes, by name, for rendering a refusal in words.
#
# Listed explicitly rather than swept out of the ``sqlite3`` namespace, because
# that namespace overloads the integers three ways: the authorizer's own return
# values (SQLITE_DENY is 1) and the SQLITE_LIMIT_* configuration constants share
# the action space. SQLITE_DELETE and SQLITE_LIMIT_VARIABLE_NUMBER are both 9,
# so a sweep rendered a refused DELETE as "limit variable number" — a message
# that exists to stop people guessing, confidently naming the wrong construct.
_ACTION_NAMES = (
	"SQLITE_CREATE_INDEX", "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX",
	"SQLITE_CREATE_TEMP_TABLE", "SQLITE_CREATE_TEMP_TRIGGER",
	"SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW",
	"SQLITE_DELETE", "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE",
	"SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE",
	"SQLITE_DROP_TEMP_TRIGGER", "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TRIGGER",
	"SQLITE_DROP_VIEW", "SQLITE_INSERT", "SQLITE_PRAGMA", "SQLITE_READ",
	"SQLITE_SELECT", "SQLITE_TRANSACTION", "SQLITE_UPDATE", "SQLITE_ATTACH",
	"SQLITE_DETACH", "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE",
	"SQLITE_CREATE_VTABLE", "SQLITE_DROP_VTABLE", "SQLITE_FUNCTION",
	"SQLITE_SAVEPOINT", "SQLITE_RECURSIVE",
)
_AUTHORIZER_ACTIONS = {
	value: name[len("SQLITE_"):].lower().replace("_", " ")
	for name, value in ((n, getattr(sqlite3, n, None)) for n in _ACTION_NAMES)
	if isinstance(value, int)
}


def _describe_action(action, arg1, arg2) -> str:
	"""One refused authorizer action, in words a person can act on."""
	name = _AUTHORIZER_ACTIONS.get(action, f"perform action {action}")
	target = " ".join(str(arg) for arg in (arg1, arg2) if arg)
	return f"{name} {target}".strip()

# The base user every session falls back to when no frontend has bound an
# identity (REPL, Telegram, background drivers). Seeded as row id 1. Identity
# (whose data) is orthogonal to authorization (frontend_profile, what's allowed):
# there is deliberately no privileged "admin" user.
DEFAULT_USER_ID = 1

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


# ---------------------------------------------------------------------------
# Priority lock
# ---------------------------------------------------------------------------
# Every DB call serializes on one connection + one mutex, so a churning
# background scan (or pipeline workers) can starve the conversation thread:
# threading.Lock isn't fair, so a thread re-acquiring in a tight loop keeps
# winning. _PriorityLock fixes that deterministically — background work
# acquires at LOW priority and always yields to a thread waiting at HIGH
# priority. A high-priority (interactive/conversation) acquirer therefore waits
# at most for the single critical section currently executing — a short
# statement plus commit — never behind a queue of background ones, regardless
# of how many files churn or how long their tasks run. Priority is read from a
# thread-local, so the existing `with db.lock:` sites need no changes; the
# calling thread's role decides.

_HIGH_PRIORITY, _LOW_PRIORITY = 0, 1
_priority_tls = threading.local()


def set_thread_priority_low():
	"""Mark the current thread as background — its DB ops yield to interactive
	(high-priority) ones. Use as a ThreadPoolExecutor ``initializer`` or at the
	top of a dedicated background thread."""
	_priority_tls.priority = _LOW_PRIORITY


class _PriorityLock:
	"""Mutex that always prefers high-priority acquirers. Non-reentrant, matching
	the threading.Lock it replaces."""

	def __init__(self):
		self._cond = threading.Condition()
		self._held = False
		self._high_waiters = 0

	def __enter__(self):
		high = getattr(_priority_tls, "priority", _HIGH_PRIORITY) == _HIGH_PRIORITY
		with self._cond:
			if high:
				# A high-priority waiter blocks only on the current holder.
				self._high_waiters += 1
				try:
					while self._held:
						self._cond.wait()
				finally:
					self._high_waiters -= 1
			else:
				# Low priority also defers to anyone waiting at high priority,
				# so background work can never jump ahead of the conversation.
				while self._held or self._high_waiters:
					self._cond.wait()
			self._held = True
		return self

	def __exit__(self, *exc):
		with self._cond:
			self._held = False
			self._cond.notify_all()
		return False


class Database:
	"""Database."""
	def __init__(self, db_path: str):
		"""Initialize the database."""
		self.db_path = db_path
		self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
		self.conn.row_factory = sqlite3.Row  # dict-like access on rows
		# Scalar functions any query may use, plugin queries included — see
		# pipeline/sql_functions.py for why an operator rather than a Request.
		sql_functions.register(self.conn)
		self.lock = _PriorityLock()
		# The single retention knob (``data_retention_days`` config, days;
		# 0 = keep everything). Set from config at bootstrap; a full prune
		# runs at startup and ledger-only pruning happens opportunistically
		# inside record_action.
		self.retention_days = 0
		self._ledger_inserts = 0
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
		# Every commit fsyncs under the default (FULL), and this database is
		# committed on a great many small writes — the action ledger above all.
		# Under WAL, NORMAL syncs at checkpoints instead: a power cut can cost
		# the last transactions and can never corrupt the file, which is the
		# right trade for a local-first app whose most frequent writer is a
		# flight recorder.
		self.conn.execute("PRAGMA synchronous=NORMAL")
		# Negative value = KB. -50000 ≈ 50 MB page cache (default is ~2 MB).
		self.conn.execute("PRAGMA cache_size=-50000")
		# Enable foreign key enforcement (needed for ON DELETE CASCADE). Must run
		# before any DML below — a pending implicit transaction makes SQLite
		# silently ignore this PRAGMA, which would disable cascades.
		self.conn.execute("PRAGMA foreign_keys = ON")

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
		# Migration: conversations become user-owned. Add the column then backfill
		# pre-existing rows to the base user so they stay visible to the operator.
		try:
			self.conn.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")
			self.conn.commit()
		except Exception:
			pass
		self.conn.execute(
			"UPDATE conversations SET user_id = ? WHERE user_id IS NULL",
			(DEFAULT_USER_ID,))

		# Users — the "user dimension". One row per identity. ``config`` is a JSON
		# blob (email, credits, per-user settings); ``username``/``password_hash``
		# are first-class columns because login looks them up directly. Identity
		# only — authorization lives in frontend_profile, not here.
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS users (
				id            INTEGER PRIMARY KEY AUTOINCREMENT,
				frontend      TEXT,
				external_id   TEXT,
				user_type     TEXT DEFAULT 'user',
				username      TEXT UNIQUE,
				password_hash TEXT,
				config        TEXT DEFAULT '{}',
				created_at    REAL,
				updated_at    REAL,
				UNIQUE(frontend, external_id)
			)
		""")
		try:
			self.conn.execute("ALTER TABLE users ADD COLUMN user_type TEXT DEFAULT 'user'")
			self.conn.commit()
		except Exception:
			pass
		self.conn.execute("UPDATE users SET user_type = 'user' WHERE user_type IS NULL OR user_type = ''")
		# Seed the base user (id 1). No credentials — it isn't a login account.
		# 'base' is a sentinel in the transport-identity columns: this user belongs
		# to no frontend (every transport falls back to it), so it is not 'local'
		# (Telegram is remote) nor 'admin' (it holds no privilege — authorization
		# lives in frontend_profile).
		now = time.time()
		self.conn.execute("""
			INSERT OR IGNORE INTO users (id, frontend, external_id, user_type, config, created_at, updated_at)
			VALUES (?, 'base', 'base', 'base', '{}', ?, ?)
		""", (DEFAULT_USER_ID, now, now))
		self.conn.execute("""
			UPDATE users SET user_type = 'base'
			WHERE id = ? AND frontend = 'base' AND external_id = 'base'
		""", (DEFAULT_USER_ID,))

		# Action ledger — append-only record of every action the system takes:
		# the two labeled enact() chokepoints (origin 'user_enact'/'agent_enact')
		# plus system-level acts such as package installs, config saves, and
		# conversation lifecycle ops (origin 'system'). Deliberately no foreign
		# keys: audit rows must outlive the conversations/users they describe.
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS action_ledger (
				id              INTEGER PRIMARY KEY AUTOINCREMENT,
				ts              REAL NOT NULL,
				origin          TEXT NOT NULL,
				session_key     TEXT,
				conversation_id INTEGER,
				user_id         INTEGER,
				actor_id        TEXT,
				action_type     TEXT NOT NULL,
				name            TEXT,
				args_json       TEXT,
				ok              INTEGER NOT NULL,
				error_code      TEXT,
				error_message   TEXT,
				call_id         TEXT,
				duration_ms     INTEGER,
				data_json       TEXT
			)
		""")
		self.conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_ledger_conv
			ON action_ledger(conversation_id, id)
		""")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_ts ON action_ledger(ts)")

		# What the system told the user about, as opposed to what it did (the
		# ledger above) or what was said in conversation (conversation_messages).
		# A notification is delivered over the bus the moment it happens, but a
		# surface that draws them in a panel needs to fill that panel on a fresh
		# load — the bus only ever answers "what happened since you connected".
		# Same no-foreign-keys rule as the ledger: a notification about a
		# conversation must outlive that conversation being deleted, or the row
		# vanishes exactly when its explanation is most wanted.
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS notifications (
				id              INTEGER PRIMARY KEY AUTOINCREMENT,
				ts              REAL NOT NULL,
				title           TEXT,
				body            TEXT,
				source          TEXT,
				source_id       TEXT,
				level           TEXT,
				session_key     TEXT,
				conversation_id INTEGER,
				user_id         INTEGER,
				read_at         REAL
			)
		""")
		# The backfill seek: one user's notifications, newest first, optionally
		# from an id they already have.
		self.conn.execute("""
			CREATE INDEX IF NOT EXISTS idx_notifications_user
			ON notifications(user_id, id)
		""")
		self.conn.execute("CREATE INDEX IF NOT EXISTS idx_notifications_ts ON notifications(ts)")

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

	def get_task_status(self, path, task_name):
		"""Return the queue status for one (path, task) pair, or None if absent."""
		with self.lock:
			cur = self.conn.execute("""
				SELECT status FROM task_queue
				WHERE path = ? AND task_name = ?
			""", (path, task_name))
			row = cur.fetchone()
			return row["status"] if row else None

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
					ORDER BY created_at ASC
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
	# SQLite's PRAGMA namespace mixes introspection with mutations
	# (foreign_keys, writable_schema, WAL checkpoints, ...).  The one read-only
	# form exposed here is the schema inspection plugins already use.
	_READ_PRAGMA = re.compile(
		r"^\s*pragma\s+(?:main\.)?table_info\s*\(\s*"
		r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\)\s*;?\s*$",
		re.IGNORECASE,
	)
	READ_PREFIXES = ("select", "explain", "with")

	@classmethod
	def _validate_read_sql(cls, sql: str) -> None:
		"""Refuse anything outside the deliberately read-only SQL surface."""
		normalized = " ".join((sql or "").strip().split()).lower()
		if normalized.startswith(cls.READ_PREFIXES):
			return
		if cls._READ_PRAGMA.fullmatch(sql or ""):
			return
		# Both messages name the route that *works*, because the refusal an
		# agent cannot act on is the one it retries verbatim. Schema questions
		# are the common case and have two ordinary-SQL answers, neither of
		# which is obvious from a message that only says what is forbidden.
		if normalized.startswith("pragma"):
			raise ValueError(
				"Only PRAGMA table_info(<table>) is available through a read; "
				"other PRAGMAs can change database state. For schema "
				"questions use ordinary SQL: SELECT name, sql FROM "
				"sqlite_master, or SELECT * FROM pragma_table_info('<table>').")
		raise ValueError(
			"Only SELECT / EXPLAIN / WITH statements and read-only "
			"PRAGMA table_info(<table>) are allowed; use db.write for a "
			"mutation. SELECT against sqlite_master and "
			"pragma_table_info('<table>') both work here.")

	#: PRAGMAs the authorizer lets through, and why each is safe to.
	#:
	#: ``table_info`` is the schema inspection plugins already use, and is
	#: additionally checked against a valid identifier below.
	#:
	#: ``data_version`` is not asked for by any plugin — **SQLite asks for it
	#: itself**, from inside a virtual table. It reports whether another
	#: connection has committed since last time, it takes no argument, and it
	#: cannot change anything at all. Denying it broke every FTS5 ``MATCH``:
	#: a query that never mentions a PRAGMA failed with a bare "authorization
	#: denied", so ``lexical_search`` could not run a single search from a
	#: sandboxed tool — and because the tool reported that as "no results",
	#: the whole keyword half of retrieval looked like an empty corpus rather
	#: than a refusal. Anything else in the namespace mixes reads with
	#: mutations (``foreign_keys``, ``writable_schema``, WAL checkpoints) and
	#: stays refused.
	_ALLOWED_PRAGMAS = frozenset({"table_info", "data_version"})

	@staticmethod
	def _read_authorizer(action, arg1, arg2, database_name, trigger_name):
		"""Let SQLite prove a query has no write or configuration effects."""
		allowed = {
			sqlite3.SQLITE_SELECT,
			sqlite3.SQLITE_READ,
			sqlite3.SQLITE_FUNCTION,
			sqlite3.SQLITE_RECURSIVE,
		}
		if action in allowed:
			return sqlite3.SQLITE_OK
		if action == sqlite3.SQLITE_PRAGMA:
			name = str(arg1 or "").lower()
			if name == "data_version" and not arg2:
				return sqlite3.SQLITE_OK
			if (name == "table_info"
					and _VALID_IDENTIFIER.fullmatch(str(arg2 or ""))):
				return sqlite3.SQLITE_OK
		return sqlite3.SQLITE_DENY

	@contextmanager
	def _read_guard(self):
		"""Install the read authorizer while the caller holds ``self.lock``.

		Records what was refused, and says so. SQLite's own message for this is
		the two words "not authorized" — no statement, no construct, no hint —
		and the construct is often **not in the query at all**: FTS5 asks for
		``PRAGMA data_version`` from inside a virtual table, so a plain
		``MATCH`` failed with a message naming nothing the author had written.
		A rule nobody can act on reads as a broken database, which is exactly
		how this one was received.

		Safe to keep on the instance because every caller holds ``self.lock``
		for the whole of the guard, so two reads can never interleave here.
		"""
		refused: list = []

		def authorize(action, arg1, arg2, database_name, trigger_name):
			"""Answer as ``_read_authorizer`` does, remembering any refusal."""
			verdict = self._read_authorizer(
				action, arg1, arg2, database_name, trigger_name)
			if verdict != sqlite3.SQLITE_OK:
				refused.append(_describe_action(action, arg1, arg2))
			return verdict

		self.conn.set_authorizer(authorize)
		try:
			yield
		except sqlite3.DatabaseError as exc:
			if refused and "not authorized" in str(exc).lower():
				raise sqlite3.DatabaseError(
					f"{exc}: this read tried to {refused[0]}. A read may use "
					f"SELECT, EXPLAIN or WITH — including SELECT against "
					f"sqlite_master and pragma_table_info('<table>') for "
					f"schema questions. Anything that changes data or "
					f"settings needs db.write."
				) from exc
			raise
		finally:
			self.conn.set_authorizer(None)

	def query(self, sql: str, params=(), max_rows: int = 25) -> dict:
		"""
		Execute a read-only SQL query and return results.

		Returns:
			{
				"columns":   list of column names,
				"rows":      list of tuples,
				"truncated": bool — True if results were capped at max_rows,
			}

		Raises ValueError for statements outside the read-only SQL surface.
		Raises sqlite3.Error for invalid SQL.
		"""
		self._validate_read_sql(sql)

		with self.lock:
			with self._read_guard():
				cur = self.conn.execute(sql, tuple(params or ()))
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

	def query_rows(self, sql: str, params=(), max_rows: int = 500) -> list:
		"""
		Execute a read and return ``sqlite3.Row`` objects.

		The counterpart to ``query()`` for callers that want rows rather than
		a display bundle — the ``db.query`` Request handler is the only one
		today. Capped at ``max_rows`` because the answer crosses a process
		boundary as JSON; the caller learns it was capped by getting exactly
		``max_rows`` back.

		Raises ValueError for a statement that is not a read.
		Raises sqlite3.Error for invalid SQL.
		"""
		self._validate_read_sql(sql)

		with self.lock:
			with self._read_guard():
				cur = self.conn.execute(sql, tuple(params or ()))
				return cur.fetchmany(max_rows)

	def execute_write(self, sql: str, params=()) -> dict:
		"""
		Execute a single mutating SQL statement (INSERT/UPDATE/DELETE/DDL) and
		commit it.

		This is the deliberate counterpart to ``query()``: it carries **no**
		read-only guard. The kernel never calls it — it exists solely so an
		explicitly approved agent/operator action can write. Every caller MUST
		gate this behind active user approval (see the sql_query tool).

		``conn.execute`` runs exactly one statement; a multi-statement script
		raises sqlite3's "one statement at a time" error, by design. A
		``RETURNING`` clause surfaces through ``columns``/``rows``.

		Returns:
			{
				"rowcount":  int — rows affected (-1 when sqlite can't report it),
				"lastrowid": int | None — rowid of the inserted row, if any,
				"columns":   list[str] — populated only for RETURNING,
				"rows":      list[tuple] — RETURNING rows, else empty,
			}

		Raises sqlite3.Error for invalid SQL.
		"""
		with self.lock:
			cur = self.conn.execute(sql, tuple(params or ()))
			columns = [desc[0] for desc in cur.description] if cur.description else []
			rows = [tuple(row) for row in cur.fetchall()] if columns else []
			self.conn.commit()
			return {
				"rowcount": cur.rowcount,
				"lastrowid": cur.lastrowid,
				"columns": columns,
				"rows": rows,
			}

	# =================================================================
	# USERS
	# =================================================================

	@staticmethod
	def _user_row(row) -> dict | None:
		"""Decode a users row into a dict with ``config`` parsed to a dict."""
		if row is None:
			return None
		user = dict(row)
		try:
			user["config"] = json.loads(user.get("config") or "{}")
		except (TypeError, json.JSONDecodeError):
			user["config"] = {}
		return user

	def upsert_user(self, frontend, external_id, config=None, user_type="user") -> int:
		"""Create-or-touch a user by transport identity; return its id.

		Stores no credentials — use ``set_user_credentials`` for those. On an
		existing (frontend, external_id) only ``updated_at`` is refreshed; the
		stored config is left intact.
		"""
		now = time.time()
		blob = json.dumps(config or {})
		kind = (str(user_type or "user").strip() or "user")
		with self.lock:
			cur = self.conn.execute("""
				INSERT INTO users (frontend, external_id, user_type, config, created_at, updated_at)
				VALUES (?, ?, ?, ?, ?, ?)
				ON CONFLICT(frontend, external_id) DO UPDATE SET updated_at = excluded.updated_at
				RETURNING id
			""", (frontend, external_id, kind, blob, now, now))
			row = cur.fetchone()
			self.conn.commit()
			return row["id"]

	def get_user(self, user_id) -> dict | None:
		"""Fetch one user by id (config parsed to a dict)."""
		with self.lock:
			cur = self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
			return self._user_row(cur.fetchone())

	def get_user_by_external(self, frontend, external_id) -> dict | None:
		"""Fetch one user by transport identity (frontend, external_id)."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM users WHERE frontend = ? AND external_id = ?",
				(frontend, external_id))
			return self._user_row(cur.fetchone())

	def get_user_by_username(self, username) -> dict | None:
		"""Fetch one user by account login name (the indexed login lookup)."""
		with self.lock:
			cur = self.conn.execute("SELECT * FROM users WHERE username = ?", (username,))
			return self._user_row(cur.fetchone())

	def set_user_credentials(self, user_id, username, password_hash) -> None:
		"""Attach/replace an account login on a user row.

		The kernel stores the hash opaquely — hashing and verification are the
		frontend's responsibility (the kernel ships no crypto).
		"""
		with self.lock:
			self.conn.execute(
				"UPDATE users SET username = ?, password_hash = ?, updated_at = ? WHERE id = ?",
				(username, password_hash, time.time(), user_id))
			self.conn.commit()

	def set_user_type(self, user_id, user_type) -> None:
		"""Set the frontend-defined user type label for a user row.

		The kernel stores and exposes this label but does not grant permissions from
		it. Frontends/policy plugins decide what values such as guest, admin, paid,
		or creator mean.
		"""
		kind = (str(user_type or "user").strip() or "user")
		with self.lock:
			self.conn.execute(
				"UPDATE users SET user_type = ?, updated_at = ? WHERE id = ?",
				(kind, time.time(), user_id))
			self.conn.commit()

	def get_user_config(self, user_id) -> dict:
		"""Return a user's config blob as a dict (empty if missing)."""
		user = self.get_user(user_id)
		return user["config"] if user else {}

	def set_user_config(self, user_id, config: dict) -> None:
		"""Replace a user's config blob."""
		with self.lock:
			self.conn.execute(
				"UPDATE users SET config = ?, updated_at = ? WHERE id = ?",
				(json.dumps(config or {}), time.time(), user_id))
			self.conn.commit()

	def list_users(self, limit=50) -> list[dict]:
		"""List users, newest first."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM users ORDER BY created_at DESC LIMIT ?", (limit,))
			return [self._user_row(row) for row in cur.fetchall()]

	# =================================================================
	# ACTION LEDGER
	# =================================================================

	# Serialized args/data are capped so one huge tool result can't bloat the
	# ledger; oversized payloads are wrapped (still valid JSON) with head+tail.
	LEDGER_JSON_CAP = 4000

	@classmethod
	def _ledger_json(cls, value) -> str | None:
		"""Serialize a ledger payload to capped, always-valid JSON (or None)."""
		if value is None:
			return None
		try:
			text = json.dumps(value, default=str)
		except Exception:
			text = json.dumps(str(value))
		if len(text) <= cls.LEDGER_JSON_CAP:
			return text
		half = cls.LEDGER_JSON_CAP // 2
		return json.dumps({
			"_truncated_chars": len(text),
			"head": text[:half],
			"tail": text[-half:],
		})

	def record_action(self, *, origin, action_type, ok, session_key=None,
					  conversation_id=None, user_id=None, actor_id=None,
					  name=None, args=None, error_code=None, error_message=None,
					  call_id=None, duration_ms=None, data=None) -> None:
		"""Append one row to the action ledger.

		Best-effort by design: the ledger observes the system and must never
		break it. Any failure (serialization, lock, disk) is swallowed and
		logged, so callers do not need their own try/except.
		"""
		try:
			with self.lock:
				self.conn.execute("""
					INSERT INTO action_ledger
					(ts, origin, session_key, conversation_id, user_id, actor_id,
					 action_type, name, args_json, ok, error_code, error_message,
					 call_id, duration_ms, data_json)
					VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
				""", (time.time(), str(origin), session_key, conversation_id,
					  user_id, actor_id, str(action_type), name,
					  self._ledger_json(args), 1 if ok else 0, error_code,
					  str(error_message)[:1000] if error_message else None,
					  call_id, duration_ms, self._ledger_json(data)))
				self.conn.commit()
			self._ledger_inserts += 1
			if self.retention_days and self._ledger_inserts % 256 == 0:
				self.prune_expired(self.retention_days, ledger_only=True)
		except Exception as e:
			logger.warning(f"Action-ledger write failed (ignored): {e}")

	# =================================================================
	# NOTIFICATIONS
	# =================================================================

	def record_notification(self, *, title=None, body=None, source=None,
							source_id=None, level="info", session_key=None,
							conversation_id=None, user_id=None) -> int | None:
		"""Append one notification and return its id, or None if it failed.

		Best-effort for the same reason the ledger is: telling the user
		something must never break the thing that had something to tell them.
		A failure costs the panel one row on the next reload, and the live bus
		delivery still happens — the caller emits regardless of what this
		answered.
		"""
		try:
			with self.lock:
				cur = self.conn.execute("""
					INSERT INTO notifications
					(ts, title, body, source, source_id, level, session_key,
					 conversation_id, user_id)
					VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
				""", (time.time(), title, body, source, source_id, str(level),
					  session_key, conversation_id, user_id))
				self.conn.commit()
				return int(cur.lastrowid)
		except Exception as e:
			logger.warning(f"Notification write failed (ignored): {e}")
			return None

	def get_notifications(self, user_id=None, since_id=None, unread_only=False,
						  limit=100) -> list[dict]:
		"""Read notifications, newest first.

		Every filter is applied in SQL, the same discipline ``get_ledger_rows``
		follows: ``idx_notifications_user`` makes the ``user_id`` + ``since_id``
		pair an index seek, which is what a client reconnecting with rows up to
		N actually asks for.

		**A NULL ``user_id`` means the system, and everybody sees it.** Most
		notifications belong to no user — a plugin registering, a setting
		changing — and they already broadcast on the bus with no session to aim
		at, so the stored form matches the delivered one. Filtering on equality
		alone made every one of those rows unreachable: they were written, they
		were never returned, and an empty panel is indistinguishable from a
		quiet system.
		"""
		clauses, params = [], []
		if user_id is not None:
			clauses.append("(user_id = ? OR user_id IS NULL)")
			params.append(int(user_id))
		if since_id is not None:
			clauses.append("id > ?"); params.append(int(since_id))
		if unread_only:
			clauses.append("read_at IS NULL")
		where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
		with self.lock:
			cur = self.conn.execute(
				f"SELECT * FROM notifications{where} ORDER BY id DESC LIMIT ?",
				(*params, int(limit)))
			return [dict(row) for row in cur.fetchall()]

	def mark_notifications_read(self, ids=None, *, user_id=None,
								before_id=None) -> int:
		"""Stamp ``read_at``. Returns how many rows changed.

		Two spellings because a panel needs both: ``ids`` for dismissing what
		was clicked, ``before_id`` for "mark everything up to here read" after
		a scroll. ``user_id`` narrows either, so one user can never settle
		another's — the handler passes it from the context rather than trusting
		an argument.

		System rows (NULL ``user_id``) are settleable by anyone, matching
		``get_notifications``: they are shown to everyone, so anyone who was
		shown one may dismiss it. Excluding them made every plugin
		registration permanently unread — the row was drawn, clicked, and
		stayed exactly where it was.

		Already-read rows are excluded rather than restamped, so the count is
		what actually changed and a repeated call is idempotent.
		"""
		clauses, params = ["read_at IS NULL"], []
		if ids:
			ids = [int(i) for i in ids]
			clauses.append(f"id IN ({','.join('?' * len(ids))})")
			params.extend(ids)
		if before_id is not None:
			clauses.append("id <= ?"); params.append(int(before_id))
		if user_id is not None:
			clauses.append("(user_id = ? OR user_id IS NULL)")
			params.append(int(user_id))
		if not ids and before_id is None:
			return 0  # refuse to settle the whole table by omission
		with self.lock:
			cur = self.conn.execute(
				f"UPDATE notifications SET read_at = ? WHERE {' AND '.join(clauses)}",
				(time.time(), *params))
			self.conn.commit()
			return int(cur.rowcount)

	def prune_expired(self, days, *, ledger_only: bool = False) -> int:
		"""Delete data older than ``days`` — the single retention knob.

		Covers everything that accumulates without bound: action-ledger rows,
		notifications, finished task runs, and idle conversations (their
		messages cascade).
		A conversation's ``updated_at`` is bumped on every message, so
		anything still in use is never eligible. ``ledger_only`` restricts to
		the cheap ledger sweep (used opportunistically from record_action;
		the full prune runs at bootstrap). Returns total rows deleted;
		``days`` <= 0 keeps everything."""
		if not days or days <= 0:
			return 0
		cutoff = time.time() - float(days) * 86400.0
		deleted: dict[str, int] = {}
		with self.lock:
			deleted["action_ledger"] = self.conn.execute(
				"DELETE FROM action_ledger WHERE ts < ?", (cutoff,)).rowcount
			if not ledger_only:
				deleted["notifications"] = self.conn.execute(
					"DELETE FROM notifications WHERE ts < ?", (cutoff,)).rowcount
				deleted["task_runs"] = self.conn.execute(
					"DELETE FROM task_runs WHERE finished_at IS NOT NULL AND finished_at < ?",
					(cutoff,)).rowcount
				deleted["conversations"] = self.conn.execute(
					"DELETE FROM conversations WHERE COALESCE(updated_at, created_at) < ?",
					(cutoff,)).rowcount
			self.conn.commit()
		total = sum(deleted.values())
		if total:
			logger.info(f"Retention prune ({days}d): {deleted}")
			# Deleting data is itself an auditable act. Recorded outside the
			# lock; the ledger row carries what was removed and why.
			self.record_action(origin="system", action_type="retention_prune",
							   ok=True, args={"days": days},
							   data={k: v for k, v in deleted.items() if v})
		return total

	def get_ledger_rows(self, conversation_id=None, origin=None, session_key=None,
						action_types=None, since_id=None, limit=100) -> list[dict]:
		"""Read recent ledger rows, newest first. For tests and inspection UX.

		Every filter is applied in SQL rather than by the caller, because the
		guidance is to read this table *targeted*: `idx_ledger_conv` makes
		`conversation_id` an index seek, and narrowing there is the difference
		between reading one conversation's rows and reading the whole flight
		recorder to throw most of it away. `since_id` is the incremental form of
		the same idea — a reader that already has rows up to N asks only for
		what followed.
		"""
		clauses, params = [], []
		if conversation_id is not None:
			clauses.append("conversation_id = ?"); params.append(conversation_id)
		if origin is not None:
			clauses.append("origin = ?"); params.append(origin)
		if session_key is not None:
			clauses.append("session_key = ?"); params.append(session_key)
		if action_types:
			# Built from the count rather than interpolated: the values are
			# still bound, so a caller-supplied type can never be SQL.
			clauses.append(f"action_type IN ({','.join('?' * len(action_types))})")
			params.extend(str(t) for t in action_types)
		if since_id is not None:
			clauses.append("id > ?"); params.append(int(since_id))
		where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
		with self.lock:
			cur = self.conn.execute(
				f"SELECT * FROM action_ledger{where} ORDER BY id DESC LIMIT ?",
				(*params, int(limit)))
			return [dict(row) for row in cur.fetchall()]

	# =================================================================
	# CONVERSATIONS
	# =================================================================

	def create_conversation(self, title="New Conversation", kind="user", category=None, user_id=DEFAULT_USER_ID) -> int:
		"""Create conversation."""
		now = time.time()
		with self.lock:
			cur = self.conn.execute(
				"INSERT INTO conversations (title, kind, category, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
				(title, kind, category, user_id, now, now))
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

	def set_conversation_category(self, conversation_id, category, user_id=None):
		"""Set/overwrite the category on a conversation row.

		When ``user_id`` is given the update is scoped to that owner (no-op on a
		mismatch) — defence-in-depth behind the runtime ownership guard.
		"""
		scope = " AND user_id = ?" if user_id is not None else ""
		params = [category, conversation_id] + ([user_id] if user_id is not None else [])
		with self.lock:
			self.conn.execute(
				f"UPDATE conversations SET category = ? WHERE id = ?{scope}",
				params)
			self.conn.commit()

	def get_conversation(self, conversation_id):
		"""Get conversation."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT * FROM conversations WHERE id = ?",
				(conversation_id,))
			row = cur.fetchone()
			return dict(row) if row else None

	def list_conversations(self, limit=50, user_id=None):
		"""List conversations, optionally scoped to one owner."""
		where = "WHERE user_id = ?" if user_id is not None else ""
		params = ([user_id] if user_id is not None else []) + [limit]
		with self.lock:
			cur = self.conn.execute(
				f"SELECT * FROM conversations {where} ORDER BY updated_at DESC LIMIT ?",
				params)
			return [dict(row) for row in cur.fetchall()]

	def list_conversations_page(self, offset=0, limit=10, category=None, user_id=None) -> tuple[list[dict], bool]:
		"""Return ``(rows, has_more)`` sorted by most-recent activity.

		``category``:
		    - None → no filter (every conversation).
		    - "" → conversations with NULL/empty category (the "Main" bucket).
		    - any other string → exact match on the category column.
		``user_id``: when given, only that owner's conversations.
		"""
		params: list = []
		clauses: list[str] = []
		if category is not None:
			if category == "":
				clauses.append("(category IS NULL OR category = '')")
			else:
				clauses.append("category = ?")
				params.append(category)
		if user_id is not None:
			clauses.append("user_id = ?")
			params.append(user_id)
		where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
		params += [limit + 1, offset]
		with self.lock:
			cur = self.conn.execute(
				f"SELECT * FROM conversations {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
				params)
			rows = [dict(row) for row in cur.fetchall()]
		has_more = len(rows) > limit
		return rows[:limit], has_more

	def list_conversation_categories(self, user_id=None) -> list[str | None]:
		"""Distinct category values present in the conversations table.

		``None`` is included if any row has NULL/empty category. Scoped to one
		owner when ``user_id`` is given.
		"""
		where = "WHERE user_id = ?" if user_id is not None else ""
		params = [user_id] if user_id is not None else []
		with self.lock:
			cur = self.conn.execute(
				f"SELECT DISTINCT category FROM conversations {where}", params)
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

	def list_user_conversations(self, limit=50, user_id=None):
		"""List user conversations, optionally scoped to one owner."""
		where = "WHERE c.user_id = ?" if user_id is not None else ""
		params = ([user_id] if user_id is not None else []) + [limit]
		with self.lock:
			cur = self.conn.execute(
				f"""
				SELECT *
				FROM conversations c
				{where}
				ORDER BY c.updated_at DESC
				LIMIT ?
				""",
				params)
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

	def delete_conversation(self, conversation_id, user_id=None):
		"""Delete conversation. When ``user_id`` is given the delete is scoped to
		that owner (no-op on a mismatch) — defence-in-depth behind the runtime
		ownership guard."""
		scope = " AND user_id = ?" if user_id is not None else ""
		params = [conversation_id] + ([user_id] if user_id is not None else [])
		with self.lock:
			self.conn.execute(
				f"DELETE FROM conversations WHERE id = ?{scope}", params)
			self.conn.commit()

	def conversation_message_count(self, conversation_id) -> int:
		"""Handle conversation message count."""
		with self.lock:
			cur = self.conn.execute(
				"SELECT COUNT(*) as cnt FROM conversation_messages WHERE conversation_id = ?",
				(conversation_id,))
			return cur.fetchone()["cnt"]
