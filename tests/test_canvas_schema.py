"""Schema regression for the canvas-domain tables.

After the canvas-rework cleanup, the canvas tables in the central DB are:
  - canvas_states      (per-user canvas editing handle: id -> state JSON)
  - canvas_pools       (pool_hash -> render-determining state JSON)
  - user_canvas_actions (user X did Y on pool Z at ts)
  - technique_scores / technique_events / technique_errors (analytics, with link_opens)

The legacy tables (canvas_shares, canvas_layer_cache, canvas_seed_pool,
and the abandoned Phase 0 ones: techniques, canvases, canvas_layers,
image_generations) are dropped on startup for existing installs.
"""

from pathlib import Path

from pipeline.database import Database


def _columns(db: Database, table: str) -> set[str]:
	"""Internal helper to handle columns."""
	rows = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
	return {row["name"] for row in rows}


def _indexes(db: Database, table: str) -> set[str]:
	"""Internal helper to handle indexes."""
	rows = db.conn.execute(f"PRAGMA index_list({table})").fetchall()
	return {row["name"] for row in rows}


def _fresh_db(name: str) -> tuple[Database, Path]:
	"""Internal helper: open a fresh DB at a project-local path."""
	path = Path(f".canvas_schema_test_{name}.sqlite")
	path.unlink(missing_ok=True)
	return Database(str(path)), path


def _cleanup(db: Database, path: Path) -> None:
	"""Close + remove a test DB."""
	db.conn.close()
	for p in (path, path.with_suffix(".sqlite-wal"), path.with_suffix(".sqlite-shm")):
		p.unlink(missing_ok=True)


def test_canvas_tables_exist():
	"""All current canvas-domain tables have the expected columns."""
	db, path = _fresh_db("tables")
	try:
		assert _columns(db, "canvas_states") == {"canvas_id", "state_json", "updated_at"}
		assert _columns(db, "canvas_pools") == {"pool_hash", "state_json", "created_at"}
		assert _columns(db, "user_canvas_actions") == {
			"id", "user_id", "pool_hash", "action", "ts", "meta_json",
		}
	finally:
		_cleanup(db, path)


def test_technique_scores_has_link_opens_column():
	"""technique_scores keeps the link_opens column added during the rework."""
	db, path = _fresh_db("scores")
	try:
		assert "link_opens" in _columns(db, "technique_scores")
	finally:
		_cleanup(db, path)


def test_legacy_tables_are_dropped_on_startup():
	"""Old canvas tables and Phase 0 dormant tables are gone after _setup."""
	db, path = _fresh_db("dropped")
	try:
		# Pretend an old install: manually create one of the legacy
		# tables, close, reopen, and confirm it's gone.
		db.conn.execute("CREATE TABLE IF NOT EXISTS canvas_shares (share_id TEXT PRIMARY KEY)")
		db.conn.commit()
		db.conn.close()
		db = Database(str(path))
		tables = {
			r["name"] for r in db.conn.execute(
				"SELECT name FROM sqlite_master WHERE type='table'"
			).fetchall()
		}
		for gone in (
			"canvas_shares", "canvas_layer_cache", "canvas_seed_pool",
			"techniques", "canvases", "canvas_layers", "image_generations",
		):
			assert gone not in tables, f"legacy table {gone!r} still present"
	finally:
		_cleanup(db, path)


def test_canvas_domain_indexes_exist():
	"""Indexes that back common queries are in place."""
	db, path = _fresh_db("idx")
	try:
		action_idx = _indexes(db, "user_canvas_actions")
		assert "idx_user_canvas_actions_user" in action_idx
		assert "idx_user_canvas_actions_pool" in action_idx
		assert "idx_canvas_pools_created" in _indexes(db, "canvas_pools")
		assert "idx_canvas_states_updated" in _indexes(db, "canvas_states")
	finally:
		_cleanup(db, path)


def test_phase0_dangling_fk_is_rebuilt_on_startup():
	"""Phase 0 installs had a FK on user_canvas_actions(canvas_id) -> canvases(id).
	After dropping `canvases`, INSERTs failed with 'no such table: canvases'.
	Startup must detect that FK and rebuild the table cleanly so writes work."""
	path = Path(".canvas_schema_test_fk_rebuild.sqlite")
	path.unlink(missing_ok=True)
	try:
		# Hand-craft the Phase 0 shape: a `canvases` table + a
		# `user_canvas_actions` table whose FK points at it. Then drop
		# `canvases` to simulate the post-cleanup state.
		import sqlite3
		raw = sqlite3.connect(str(path))
		raw.execute("CREATE TABLE canvases (id TEXT PRIMARY KEY)")
		raw.execute("""
			CREATE TABLE user_canvas_actions (
				id INTEGER PRIMARY KEY,
				user_id TEXT NOT NULL,
				canvas_id TEXT NOT NULL,
				action TEXT NOT NULL,
				ts REAL NOT NULL,
				FOREIGN KEY (canvas_id) REFERENCES canvases(id) ON DELETE CASCADE
			)
		""")
		raw.commit()
		raw.close()

		# Opening the Database should drop `canvases` and rebuild
		# user_canvas_actions without the dangling FK.
		db = Database(str(path))
		try:
			# An insert that would have triggered the FK lookup must succeed.
			import time as _t
			db.conn.execute(
				"INSERT INTO user_canvas_actions (user_id, pool_hash, action, ts) VALUES (?, ?, ?, ?)",
				("u1", "ph1", "share", _t.time()),
			)
			db.conn.commit()
			row = db.conn.execute("SELECT user_id, pool_hash FROM user_canvas_actions").fetchone()
			assert row["user_id"] == "u1"
			assert row["pool_hash"] == "ph1"
			# No FK to canvases remains on the rebuilt table.
			fks = db.conn.execute("PRAGMA foreign_key_list(user_canvas_actions)").fetchall()
			assert not any((r["table"] if "table" in r.keys() else r[2]) == "canvases" for r in fks)
		finally:
			db.conn.close()
	finally:
		path.unlink(missing_ok=True)


def test_setup_is_idempotent():
	"""Re-opening the same DB doesn't crash and keeps the canvas tables."""
	path = Path(".canvas_schema_test_reopen.sqlite")
	path.unlink(missing_ok=True)
	try:
		Database(str(path)).conn.close()
		db = Database(str(path))
		try:
			tables = {
				r["name"] for r in db.conn.execute(
					"SELECT name FROM sqlite_master WHERE type='table'"
				).fetchall()
			}
			assert "canvas_states" in tables
			assert "canvas_pools" in tables
			assert "user_canvas_actions" in tables
		finally:
			db.conn.close()
	finally:
		path.unlink(missing_ok=True)


def test_existing_auth_tokens_gain_guest_claim_column():
	path = Path(".canvas_schema_test_auth_tokens.sqlite")
	path.unlink(missing_ok=True)
	try:
		import sqlite3
		raw = sqlite3.connect(str(path))
		raw.execute("CREATE TABLE web_auth_tokens (token TEXT PRIMARY KEY, email TEXT NOT NULL, created_at REAL NOT NULL, used_at REAL)")
		raw.commit()
		raw.close()
		db = Database(str(path))
		try:
			assert "anon_user_id" in _columns(db, "web_auth_tokens")
		finally:
			db.conn.close()
	finally:
		path.unlink(missing_ok=True)
