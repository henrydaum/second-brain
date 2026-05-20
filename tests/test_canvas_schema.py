"""Phase 0 schema regression for the canvas domain rework.

Asserts the new tables, columns, and indexes are created by Database._setup
so later phases can wire against a stable target. No app code reads or
writes these tables yet.
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


def test_canvas_domain_tables_exist():
	"""Phase 0: every new table is created with the expected columns."""
	db, path = _fresh_db("tables")
	try:
		assert _columns(db, "skills") == {
			"slug", "name", "kind", "code", "description",
			"embedding", "created_at", "updated_at",
		}

		assert _columns(db, "canvases") == {
			"id", "title", "artist", "size", "palette_id",
			"current_generation_id", "created_at", "updated_at",
		}

		assert _columns(db, "canvas_layers") == {
			"id", "canvas_id", "position", "skill_slug", "controls_json",
		}

		assert _columns(db, "image_generations") == {
			"id", "cache_key", "pool_key", "skill_slug",
			"input_generation_id", "controls_json", "controls_hash",
			"seed", "file_path", "bytes", "embedding",
			"created_at", "last_used", "use_count",
		}

		assert _columns(db, "user_canvas_actions") == {
			"id", "user_id", "canvas_id", "action", "ts",
		}
	finally:
		db.conn.close()
		path.unlink(missing_ok=True)


def test_skill_scores_gains_link_opens_column():
	"""Phase 0: skill_scores.link_opens is added via the lazy ALTER pattern."""
	db, path = _fresh_db("alter")
	try:
		assert "link_opens" in _columns(db, "skill_scores")
	finally:
		db.conn.close()
		path.unlink(missing_ok=True)


def test_canvas_domain_indexes_exist():
	"""Phase 0: indexes that back common queries are in place."""
	db, path = _fresh_db("idx")
	try:
		assert "idx_canvas_layers_canvas" in _indexes(db, "canvas_layers")
		gen_idx = _indexes(db, "image_generations")
		assert "idx_image_generations_pool" in gen_idx
		assert "idx_image_generations_last_used" in gen_idx
		assert "idx_image_generations_input" in gen_idx
		action_idx = _indexes(db, "user_canvas_actions")
		assert "idx_user_canvas_actions_user" in action_idx
		assert "idx_user_canvas_actions_canvas" in action_idx
	finally:
		db.conn.close()
		path.unlink(missing_ok=True)


def test_setup_is_idempotent():
	"""Re-opening the same DB doesn't crash on the new DDL or the lazy ALTER."""
	path = Path(".canvas_schema_test_reopen.sqlite")
	path.unlink(missing_ok=True)
	try:
		Database(str(path)).conn.close()
		db = Database(str(path))
		try:
			assert "link_opens" in _columns(db, "skill_scores")
			tables = {
				r["name"] for r in db.conn.execute(
					"SELECT name FROM sqlite_master WHERE type='table'"
				).fetchall()
			}
			assert "skills" in tables
			assert "canvases" in tables
		finally:
			db.conn.close()
	finally:
		path.unlink(missing_ok=True)
