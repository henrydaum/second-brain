"""Tests for the pool-hash design.

Covers:
  - canvas_pools save/load round-trip (canvas/persistence.py).
  - render_canvas writing canvas_pools on cache miss (canvas/render.py).
  - CanvasRuntime.remix(pool_hash) cloning state under a new canvas_id.
  - canvas/actions.py recording user_canvas_actions + fanning out into
    skill_scores via skill_scoring.

Skill subprocess is monkeypatched in the render tests so the suite stays fast.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from canvas import actions as canvas_actions
from canvas import persistence as canvas_persistence
from canvas import render as canvas_render
from canvas.runtime import CanvasRuntime
from canvas.state import CanvasState
from pipeline.database import Database


# =============================================================
# helpers
# =============================================================

def _fresh_db(name: str) -> tuple[Database, Path]:
	"""Project-local DB (works around Windows tmp_path permission gotcha)."""
	path = Path(f".canvas_pool_test_{name}.sqlite")
	path.unlink(missing_ok=True)
	return Database(str(path)), path


def _cleanup(db: Database, path: Path) -> None:
	"""Close + remove db + WAL siblings."""
	db.conn.close()
	for p in (path, path.with_suffix(".sqlite-wal"), path.with_suffix(".sqlite-shm")):
		p.unlink(missing_ok=True)


def _seed_state(slug: str = "fractal", **controls) -> CanvasState:
	"""One-layer creation, populated and ready to render."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": slug, "kind": "creation", "controls": controls})
	return cs


def _install_fake_run_skill(monkeypatch):
	"""Replace render.run_skill with a stub that writes a tiny PNG."""
	def fake_run_skill(skill, *, params, palette, size, seed, input_image_path, output_image_path, **kwargs):
		Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
		Image.new("RGBA", (4, 4), (32, 64, 96, 255)).save(output_image_path, format="PNG")
		return {"ok": True}
	monkeypatch.setattr(canvas_render, "run_skill", fake_run_skill)


@pytest.fixture
def renders_dir(request, monkeypatch):
	"""Project-local canvas_renders dir per test, cleaned up after."""
	target = Path(f".canvas_pool_renders_{request.node.name}")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	monkeypatch.setattr(canvas_render, "RENDERS_DIR", target)
	yield target
	shutil.rmtree(target, ignore_errors=True)


def _skill(slug: str, kind: str = "creation"):
	return SimpleNamespace(slug=slug, kind=kind, code="")


def _loader(skills: dict):
	return lambda slug: skills.get(slug)


# =============================================================
# canvas_pools persistence
# =============================================================

def test_save_pool_and_load_pool_round_trip():
	"""save_pool round-trips through load_pool by pool_hash."""
	db, path = _fresh_db("pool_rt")
	try:
		cs = _seed_state("fractal", zoom=1.0)
		ph = canvas_render.pool_hash(cs.canvas)
		canvas_persistence.save_pool(db, pool_hash=ph, state=cs.canvas.to_dict())

		loaded = canvas_persistence.load_pool(db, ph)
		assert loaded is not None
		assert loaded.get("layers") == cs.canvas.to_dict()["layers"]
		assert loaded.get("size") == cs.canvas.size
		assert loaded.get("palette_id") == cs.canvas.palette_id
	finally:
		_cleanup(db, path)


def test_save_pool_is_idempotent_on_hash():
	"""Calling save_pool twice with the same hash leaves a single row."""
	db, path = _fresh_db("pool_idempotent")
	try:
		cs = _seed_state("fractal")
		ph = canvas_render.pool_hash(cs.canvas)
		canvas_persistence.save_pool(db, pool_hash=ph, state=cs.canvas.to_dict())
		canvas_persistence.save_pool(db, pool_hash=ph, state=cs.canvas.to_dict())
		rows = db.conn.execute("SELECT COUNT(*) AS n FROM canvas_pools").fetchone()
		assert rows["n"] == 1
	finally:
		_cleanup(db, path)


def test_load_pool_unknown_returns_none():
	"""Unknown pool_hash yields None, not an exception."""
	db, path = _fresh_db("pool_unknown")
	try:
		assert canvas_persistence.load_pool(db, "deadbeefdeadbeef") is None
	finally:
		_cleanup(db, path)


# =============================================================
# render_canvas writes the pool when db is passed
# =============================================================

def test_render_canvas_writes_pool_on_cache_miss(monkeypatch, renders_dir):
	"""A fresh render with db= writes the canvas_pools row."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("render_pool")
	try:
		cs = _seed_state("fractal", z=2.0)
		loader = _loader({"fractal": _skill("fractal")})
		result = canvas_render.render_canvas(cs, skill_loader=loader, seed=11, db=db)
		assert result.cache_hit is False
		ph = result.pool_hash
		loaded = canvas_persistence.load_pool(db, ph)
		assert loaded is not None
		assert loaded["layers"] == cs.canvas.to_dict()["layers"]
	finally:
		_cleanup(db, dbpath)


def test_render_canvas_does_not_require_db(monkeypatch, renders_dir):
	"""Rendering without ``db`` still works — pool write is silently skipped."""
	_install_fake_run_skill(monkeypatch)
	cs = _seed_state("fractal")
	loader = _loader({"fractal": _skill("fractal")})
	# Just verify no exception when db is omitted.
	canvas_render.render_canvas(cs, skill_loader=loader, seed=7)


def test_render_canvas_writes_pool_on_cache_hit_too(monkeypatch, renders_dir):
	"""Pool write must run on cache hits, not just misses — otherwise canvases
	rendered before canvas_pools existed would never become shareable. INSERT OR
	IGNORE keeps repeat writes free."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("render_pool_hit")
	try:
		cs = _seed_state("fractal")
		loader = _loader({"fractal": _skill("fractal")})

		# First call: miss → renders, writes pool row.
		canvas_render.render_canvas(cs, skill_loader=loader, seed=11, db=db)
		# Clear canvas_pools to simulate "pool entry never existed yet"
		# (e.g. file on disk from before the new code).
		with db.lock:
			db.conn.execute("DELETE FROM canvas_pools")
			db.conn.commit()

		# Second call: same config → cache hit. MUST still write the pool row.
		canvas_render.render_canvas(cs, skill_loader=loader, db=db)
		ph = canvas_render.pool_hash(cs.canvas)
		assert canvas_persistence.load_pool(db, ph) is not None
	finally:
		_cleanup(db, dbpath)


# =============================================================
# CanvasRuntime.remix
# =============================================================

def test_remix_clones_pool_into_fresh_canvas_id():
	"""remix(pool_hash) materializes a new CanvasState with a fresh canvas_id."""
	db, path = _fresh_db("remix_basic")
	try:
		# Seed a pool entry directly (skip the render step for speed).
		source = _seed_state("fractal", zoom=2.0)
		ph = canvas_render.pool_hash(source.canvas)
		canvas_persistence.save_pool(db, pool_hash=ph, state=source.canvas.to_dict())

		rt = CanvasRuntime(db=db)
		fresh = rt.remix(ph)
		assert fresh is not None
		assert fresh.canvas_id != source.canvas_id  # new editing handle
		# Same content -> identical pool_hash.
		assert canvas_render.pool_hash(fresh.canvas) == ph
		# And it's registered + persisted under the new id.
		assert rt.get(fresh.canvas_id) is fresh
		assert canvas_persistence.load(db, fresh.canvas_id) is not None
	finally:
		_cleanup(db, path)


def test_remix_unknown_pool_hash_returns_none():
	"""Unknown pool_hash yields None — caller surfaces the error."""
	db, path = _fresh_db("remix_unknown")
	try:
		rt = CanvasRuntime(db=db)
		assert rt.remix("deadbeefdeadbeef") is None
	finally:
		_cleanup(db, path)


def test_remix_without_db_returns_none():
	"""Without a wired db there's no pool to read from."""
	rt = CanvasRuntime()
	assert rt.remix("anything") is None


def test_remix_does_not_disturb_original_canvas():
	"""Editing the remix doesn't touch the source canvas state."""
	db, path = _fresh_db("remix_isolation")
	try:
		source = _seed_state("fractal")
		ph = canvas_render.pool_hash(source.canvas)
		canvas_persistence.save_pool(db, pool_hash=ph, state=source.canvas.to_dict())

		rt = CanvasRuntime(db=db)
		fresh = rt.remix(ph)
		assert fresh is not None
		rt.handle_action(fresh.canvas_id, "add_layer", {"skill_slug": "swirl", "kind": "transform"})
		# Source's persisted pool still has only the one layer.
		reloaded = canvas_persistence.load_pool(db, ph)
		assert len(reloaded["layers"]) == 1
	finally:
		_cleanup(db, path)


# =============================================================
# canvas/actions.py — user_canvas_actions writes + fan-out
# =============================================================

def test_record_user_action_writes_row():
	"""record_user_action inserts into user_canvas_actions."""
	db, path = _fresh_db("ucactions_row")
	try:
		canvas_actions.record_user_action(
			db, user_id="u1", pool_hash="abc123",
			action="save", layers=[{"slug": "fractal", "kind": "creation"}],
		)
		rows = db.conn.execute(
			"SELECT user_id, pool_hash, action FROM user_canvas_actions"
		).fetchall()
		assert len(rows) == 1
		assert rows[0]["user_id"] == "u1"
		assert rows[0]["pool_hash"] == "abc123"
		assert rows[0]["action"] == "save"
	finally:
		_cleanup(db, path)


def test_record_user_action_bumps_skill_scores():
	"""'share' fans out to skill_scores via skill_scoring.record_event."""
	db, path = _fresh_db("ucactions_scores")
	try:
		canvas_actions.record_user_action(
			db, user_id="u1", pool_hash="ph1",
			action="share",
			layers=[{"slug": "fractal", "kind": "creation"},
			        {"slug": "swirl", "kind": "transform"}],
		)
		scores = {
			r["slug"]: r["shares"]
			for r in db.conn.execute("SELECT slug, shares FROM skill_scores").fetchall()
		}
		# Default weights: 0.6 to creation, 0.4 to the single transform.
		assert scores["fractal"] == 0.6
		assert scores["swirl"] == 0.4
	finally:
		_cleanup(db, path)


def test_record_user_action_stores_meta_json():
	"""meta dict is stored as JSON in user_canvas_actions.meta_json."""
	db, path = _fresh_db("ucactions_meta")
	try:
		canvas_actions.record_user_action(
			db, user_id="u1", pool_hash="ph1", action="share",
			layers=[{"slug": "fractal", "kind": "creation"}],
			meta={"title": "Sunset", "artist": "anon"},
		)
		row = db.conn.execute(
			"SELECT meta_json FROM user_canvas_actions WHERE user_id = 'u1'"
		).fetchone()
		assert row["meta_json"] is not None
		import json
		stored = json.loads(row["meta_json"])
		assert stored == {"title": "Sunset", "artist": "anon"}
	finally:
		_cleanup(db, path)


def test_list_user_canvases_returns_pool_hashes_newest_first():
	"""list_user_canvases dedupes by pool_hash and orders by most-recent ts."""
	db, path = _fresh_db("ucactions_list")
	try:
		canvas_actions.record_user_action(db, user_id="u1", pool_hash="A", action="save",
		                                    layers=[{"slug": "fractal", "kind": "creation"}])
		canvas_actions.record_user_action(db, user_id="u1", pool_hash="B", action="save",
		                                    layers=[{"slug": "fractal", "kind": "creation"}])
		# Re-save A so it becomes most recent.
		canvas_actions.record_user_action(db, user_id="u1", pool_hash="A", action="save",
		                                    layers=[{"slug": "fractal", "kind": "creation"}])

		saved = canvas_actions.list_user_canvases(db, user_id="u1", action="save")
		assert [r["pool_hash"] for r in saved] == ["A", "B"]
	finally:
		_cleanup(db, path)


def test_count_action_counts_distinct_users():
	"""Two users both saving the same pool counts as 2."""
	db, path = _fresh_db("ucactions_count")
	try:
		layers = [{"slug": "fractal", "kind": "creation"}]
		canvas_actions.record_user_action(db, user_id="u1", pool_hash="ph", action="save", layers=layers)
		canvas_actions.record_user_action(db, user_id="u2", pool_hash="ph", action="save", layers=layers)
		canvas_actions.record_user_action(db, user_id="u1", pool_hash="ph", action="save", layers=layers)
		# u1's two saves still count as one user.
		assert canvas_actions.count_action(db, pool_hash="ph", action="save") == 2
	finally:
		_cleanup(db, path)
