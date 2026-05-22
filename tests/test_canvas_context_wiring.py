"""Verify CanvasRuntime is reachable from a SecondBrainContext.

Confirms the two paths a tool/task might find it: explicit ``canvas=`` arg
to build_context, and the ``services["canvas"]`` fallback used by callers
that don't know about the canvas runtime (orchestrator, tool_registry).
"""

from __future__ import annotations

from pathlib import Path

from canvas.runtime import CanvasRuntime
from pipeline.database import Database
from runtime.context import build_context


def _fresh_db(name: str) -> tuple[Database, Path]:
	"""Project-local DB, same pattern as the other canvas tests."""
	path = Path(f".canvas_wiring_test_{name}.sqlite")
	path.unlink(missing_ok=True)
	return Database(str(path)), path


def _cleanup(db: Database, path: Path) -> None:
	"""Close + remove DB and WAL siblings."""
	db.conn.close()
	for p in (path, path.with_suffix(".sqlite-wal"), path.with_suffix(".sqlite-shm")):
		p.unlink(missing_ok=True)


def test_context_picks_up_canvas_from_services():
	"""build_context resolves canvas from services if not passed explicitly."""
	db, path = _fresh_db("services")
	try:
		canvas = CanvasRuntime(db=db)
		ctx = build_context(db, {}, {"canvas": canvas})
		assert ctx.canvas is canvas
	finally:
		_cleanup(db, path)


def test_context_explicit_canvas_overrides_services():
	"""When ``canvas=`` is passed, it wins over the services entry."""
	db, path = _fresh_db("explicit")
	try:
		from_services = CanvasRuntime(db=db)
		from_arg = CanvasRuntime(db=db)
		ctx = build_context(db, {}, {"canvas": from_services}, canvas=from_arg)
		assert ctx.canvas is from_arg
	finally:
		_cleanup(db, path)


def test_context_canvas_is_none_when_not_provided():
	"""No canvas anywhere → context.canvas is None (no implicit construction)."""
	db, path = _fresh_db("none")
	try:
		ctx = build_context(db, {}, {})
		assert ctx.canvas is None
	finally:
		_cleanup(db, path)


def test_canvas_runtime_drives_through_context():
	"""A tool with the context can fully drive a canvas via context.canvas."""
	db, path = _fresh_db("drive")
	try:
		ctx = build_context(db, {}, {"canvas": CanvasRuntime(db=db)})
		cid = ctx.canvas.create_canvas()
		r = ctx.canvas.handle_action(cid, "add_layer", {"skill_slug": "fractal", "kind": "background"})
		assert r.ok
		assert ctx.canvas.get(cid).canvas.layers[0]["slug"] == "fractal"
	finally:
		_cleanup(db, path)
