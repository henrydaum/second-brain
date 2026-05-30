"""Persistence + db-backed CanvasRuntime tests.

Covers the canvas_persistence module directly (save / load / list_ids /
delete) and the CanvasRuntime integration: autosave on action, lazy load
on get, two-runtime isolation through one DB, etc.
"""

from __future__ import annotations

from pathlib import Path

from canvas import persistence as canvas_persistence
from canvas.runtime import CanvasRuntime
from canvas.state import CanvasState
from pipeline.database import Database


# =================================================================
# helpers — local file paths, matches tests/test_canvas_store.py style
# =================================================================

def _fresh_db(name: str) -> tuple[Database, Path]:
	"""Open a fresh DB at a project-local path."""
	path = Path(f".canvas_persist_test_{name}.sqlite")
	path.unlink(missing_ok=True)
	return Database(str(path)), path


def _cleanup(db: Database, path: Path) -> None:
	"""Close + remove a test DB and its WAL/shm siblings."""
	db.conn.close()
	path.unlink(missing_ok=True)
	for sibling in (
		path.with_suffix(".sqlite-wal"),
		path.with_suffix(".sqlite-shm"),
	):
		sibling.unlink(missing_ok=True)


# =================================================================
# persistence module
# =================================================================

def test_save_and_load_round_trip():
	"""save() then load() yields an equivalent CanvasState."""
	db, path = _fresh_db("roundtrip")
	try:
		cs = CanvasState(canvas_id="abc")
		cs.enact("set_palette", {"palette_id": "obsidian"})
		cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background", "controls": {"z": 2}})
		cs.render_seed = 123
		canvas_persistence.save(db, cs)

		restored = canvas_persistence.load(db, "abc")
		assert restored is not None
		assert restored.canvas_id == "abc"
		assert restored.canvas.palette_id == "obsidian"
		assert restored.canvas.layers == cs.canvas.layers
		assert restored.render_seed == 123
	finally:
		_cleanup(db, path)


def test_save_is_upsert():
	"""Saving the same id twice replaces the row in place."""
	db, path = _fresh_db("upsert")
	try:
		cs = CanvasState(canvas_id="x")
		canvas_persistence.save(db, cs)
		cs.enact("set_size", {"size": 800})
		canvas_persistence.save(db, cs)

		rows = db.conn.execute("SELECT canvas_id, state_json FROM canvas_states").fetchall()
		assert len(rows) == 1
		restored = canvas_persistence.load(db, "x")
		assert restored.canvas.size == 800
	finally:
		_cleanup(db, path)


def test_load_unknown_returns_none():
	"""Loading an id with no row returns None."""
	db, path = _fresh_db("missing")
	try:
		assert canvas_persistence.load(db, "nope") is None
	finally:
		_cleanup(db, path)


def test_list_ids_orders_by_updated_desc():
	"""list_ids returns newest-saved first."""
	db, path = _fresh_db("list")
	try:
		a = CanvasState(canvas_id="a")
		b = CanvasState(canvas_id="b")
		c = CanvasState(canvas_id="c")
		canvas_persistence.save(db, a)
		canvas_persistence.save(db, b)
		canvas_persistence.save(db, c)
		# Re-save b last; expect b first.
		canvas_persistence.save(db, b)

		ids = canvas_persistence.list_ids(db)
		assert ids[0] == "b"
		assert set(ids) == {"a", "b", "c"}
	finally:
		_cleanup(db, path)


def test_delete_removes_row():
	"""delete() drops the row; subsequent load returns None."""
	db, path = _fresh_db("delete")
	try:
		cs = CanvasState(canvas_id="d")
		canvas_persistence.save(db, cs)
		canvas_persistence.delete(db, "d")
		assert canvas_persistence.load(db, "d") is None
		# Deleting again is a no-op, not an error.
		canvas_persistence.delete(db, "d")
	finally:
		_cleanup(db, path)


# =================================================================
# CanvasRuntime with db wired in
# =================================================================

def test_runtime_autosaves_on_create():
	"""create_canvas writes a row to canvas_states immediately."""
	db, path = _fresh_db("autosave_create")
	try:
		rt = CanvasRuntime(db=db)
		cid = rt.create_canvas()
		row = db.conn.execute(
			"SELECT canvas_id FROM canvas_states WHERE canvas_id = ?", (cid,)
		).fetchone()
		assert row is not None
	finally:
		_cleanup(db, path)


def test_runtime_autosaves_on_successful_action():
	"""Every successful handle_action persists the new state."""
	db, path = _fresh_db("autosave_action")
	try:
		rt = CanvasRuntime(db=db)
		cid = rt.create_canvas()
		rt.handle_action(cid, "set_palette", {"palette_id": "obsidian"})

		# Reload directly via persistence layer to confirm the disk state.
		restored = canvas_persistence.load(db, cid)
		assert restored is not None
		assert restored.canvas.palette_id == "obsidian"
	finally:
		_cleanup(db, path)


def test_runtime_does_not_save_on_failed_action():
	"""A failed action leaves the persisted state untouched."""
	db, path = _fresh_db("no_save_on_fail")
	try:
		rt = CanvasRuntime(db=db)
		cid = rt.create_canvas()
		rt.handle_action(cid, "set_palette", {"palette_id": "obsidian"})
		# Snapshot the updated_at after the legit action.
		ts_before = db.conn.execute(
			"SELECT updated_at FROM canvas_states WHERE canvas_id = ?", (cid,)
		).fetchone()["updated_at"]

		# Now drive a guaranteed failure.
		r = rt.handle_action(cid, "remove_layer", {"chain_index": 99})
		assert not r.ok

		ts_after = db.conn.execute(
			"SELECT updated_at FROM canvas_states WHERE canvas_id = ?", (cid,)
		).fetchone()["updated_at"]
		assert ts_after == ts_before
	finally:
		_cleanup(db, path)


def test_runtime_lazy_loads_on_get():
	"""A second runtime sharing the same DB reads a canvas it never created."""
	db, path = _fresh_db("lazy_load")
	try:
		rt_a = CanvasRuntime(db=db)
		cid = rt_a.create_canvas()
		rt_a.handle_action(cid, "set_palette", {"palette_id": "obsidian"})

		# Fresh runtime, same DB — should hydrate from canvas_states.
		rt_b = CanvasRuntime(db=db)
		cs = rt_b.get(cid)
		assert cs is not None
		assert cs.canvas.palette_id == "obsidian"
		# After hydration the canvas is in rt_b's registry.
		assert cid in rt_b.canvases
	finally:
		_cleanup(db, path)


def test_runtime_lazy_loads_on_handle_action():
	"""handle_action transparently loads a persisted canvas it hasn't seen."""
	db, path = _fresh_db("lazy_action")
	try:
		rt_a = CanvasRuntime(db=db)
		cid = rt_a.create_canvas()

		rt_b = CanvasRuntime(db=db)
		r = rt_b.handle_action(cid, "set_palette", {"palette_id": "obsidian"})
		assert r.ok
		# The action mutated the lazy-loaded canvas and persisted again.
		assert rt_b.get(cid).canvas.palette_id == "obsidian"
	finally:
		_cleanup(db, path)


def test_runtime_delete_clears_persistence():
	"""rt.delete drops the row and a fresh runtime can't load it back."""
	db, path = _fresh_db("delete_runtime")
	try:
		rt = CanvasRuntime(db=db)
		cid = rt.create_canvas()
		rt.delete(cid)

		rt2 = CanvasRuntime(db=db)
		assert rt2.get(cid) is None
	finally:
		_cleanup(db, path)


def test_runtime_without_db_still_works_in_memory():
	"""Existing in-memory behavior is unchanged when db is None."""
	rt = CanvasRuntime()
	cid = rt.create_canvas()
	r = rt.handle_action(cid, "set_palette", {"palette_id": "obsidian"})
	assert r.ok
	assert rt.get(cid).canvas.palette_id == "obsidian"


def test_runtime_list_ids_unions_memory_and_db():
	"""list_ids returns both in-memory and persisted canvases."""
	db, path = _fresh_db("list_ids")
	try:
		rt = CanvasRuntime(db=db)
		a = rt.create_canvas()
		b = rt.create_canvas()
		# Forget one in-memory; should still come back via the DB.
		rt.canvases.pop(b)

		ids = rt.list_ids()
		assert set(ids) == {a, b}
	finally:
		_cleanup(db, path)
