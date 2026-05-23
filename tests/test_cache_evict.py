"""Tests for the cache_evict task + lazy re-render in pool_share_payload."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import plugins.tasks.task_cache_evict as task_mod
from canvas import render as canvas_render
from canvas import persistence as canvas_persistence
from canvas.runtime import CanvasRuntime
from pipeline.database import Database
import plugins.frontends.frontend_web as fw


# =================================================================
# helpers
# =================================================================

def _install_fake_run_skill(monkeypatch):
	def fake_run_skill(skill, *, params, palette, size, seed, input_image_path, output_image_path, **kwargs):
		Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
		Image.new("RGBA", (4, 4), (16, 32, 64, 255)).save(output_image_path, format="PNG")
		return {"ok": True}
	monkeypatch.setattr(canvas_render, "run_skill", fake_run_skill)


@pytest.fixture
def renders_dir(request, monkeypatch):
	target = Path(f".cache_evict_test_{request.node.name}")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	monkeypatch.setattr(canvas_render, "RENDERS_DIR", target)
	monkeypatch.setattr(task_mod, "RENDERS_DIR", target)
	yield target
	shutil.rmtree(target, ignore_errors=True)


@pytest.fixture
def db(request):
	# Project-local DB path — avoids Windows pytest tmpdir permission issues.
	p = Path(f".cache_evict_test_{request.node.name}.db")
	if p.exists():
		p.unlink()
	d = Database(str(p))
	yield d
	try:
		d.conn.close()
	except Exception:
		pass
	try:
		p.unlink()
	except OSError:
		pass


def _make_pool(folder: Path, pool_hash: str, num_files: int, file_size: int, mtime: float) -> Path:
	"""Materialize a pool subfolder with given size + mtime."""
	pool = folder / pool_hash
	pool.mkdir(parents=True, exist_ok=True)
	for i in range(num_files):
		p = pool / f"{i + 1}.png"
		p.write_bytes(b"\x00" * file_size)
	# Set mtime on the most recent file (eviction reads newest within folder).
	for p in pool.iterdir():
		import os
		os.utime(p, (mtime, mtime))
	return pool


# =================================================================
# _scan_pools / _last_access_by_pool
# =================================================================

def test_scan_pools_reports_size_and_mtime(renders_dir):
	now = time.time()
	_make_pool(renders_dir, "aaaa", num_files=2, file_size=500, mtime=now - 100)
	_make_pool(renders_dir, "bbbb", num_files=1, file_size=1000, mtime=now - 200)

	pools = {p["pool_hash"]: p for p in task_mod._scan_pools(renders_dir)}
	assert pools["aaaa"]["size"] == 1000
	assert pools["bbbb"]["size"] == 1000
	# mtime within ~1s of what we set
	assert abs(pools["aaaa"]["mtime"] - (now - 100)) < 1.5
	assert abs(pools["bbbb"]["mtime"] - (now - 200)) < 1.5


def test_last_access_by_pool_aggregates_max_ts(db):
	# Two pools, several actions; should pick MAX per pool.
	now = time.time()
	with db.lock:
		db.conn.executemany(
			"INSERT INTO user_canvas_actions (user_id, pool_hash, action, ts) VALUES (?, ?, ?, ?)",
			[
				("u1", "pool_a", "share", now - 500),
				("u1", "pool_a", "download", now - 100),  # newest for A
				("u2", "pool_b", "save", now - 300),
				("u2", "pool_b", "link_open", now - 800),
			],
		)
		db.conn.commit()
	access = task_mod._last_access_by_pool(db)
	assert abs(access["pool_a"] - (now - 100)) < 1.0
	assert abs(access["pool_b"] - (now - 300)) < 1.0


# =================================================================
# end-to-end run_event
# =================================================================

def test_evicts_oldest_until_under_cap(renders_dir, db):
	"""LRU eviction by last-action-ts; folder mtime is the fallback."""
	now = time.time()
	# Three 1 MB pools. Cap will be set at 2 MB → one must go.
	_make_pool(renders_dir, "oldest", num_files=1, file_size=1_000_000, mtime=now - 9999)
	_make_pool(renders_dir, "middle", num_files=1, file_size=1_000_000, mtime=now - 5000)
	_make_pool(renders_dir, "newest", num_files=1, file_size=1_000_000, mtime=now - 100)

	# Record actions: 'oldest' has a very recent action, so it should be preserved.
	# 'middle' has an ancient action → first to go. 'newest' relies on mtime.
	with db.lock:
		db.conn.executemany(
			"INSERT INTO user_canvas_actions (user_id, pool_hash, action, ts) VALUES (?, ?, ?, ?)",
			[
				("u1", "oldest", "share", now - 60),   # protects oldest
				("u1", "middle", "share", now - 99999),  # marks middle as truly old
			],
		)
		db.conn.commit()

	task = task_mod.CacheEvict()
	ctx = SimpleNamespace(db=db, config={"canvas_cache_max_gb": 2 / 1024})  # 2 MB cap
	r = task.run_event("test-run", {}, ctx)
	assert r.success

	remaining = {p.name for p in renders_dir.iterdir()}
	# 'middle' had the oldest recorded action, evicted first.
	assert "middle" not in remaining
	# 'oldest' is preserved because its DB action timestamp is recent.
	assert "oldest" in remaining
	assert "newest" in remaining


def test_no_eviction_when_under_cap(renders_dir, db):
	_make_pool(renders_dir, "small", num_files=1, file_size=1000, mtime=time.time())
	task = task_mod.CacheEvict()
	ctx = SimpleNamespace(db=db, config={"canvas_cache_max_gb": 1.0})
	r = task.run_event("test-run", {}, ctx)
	assert r.success
	assert (renders_dir / "small").is_dir()


def test_handles_missing_renders_dir(renders_dir, db):
	shutil.rmtree(renders_dir)
	task = task_mod.CacheEvict()
	ctx = SimpleNamespace(db=db, config={"canvas_cache_max_gb": 1.0})
	r = task.run_event("test-run", {}, ctx)
	assert r.success  # graceful no-op


# =================================================================
# lazy re-render in pool_share_payload
# =================================================================

def test_pool_share_payload_re_renders_when_files_evicted(monkeypatch, renders_dir, db):
	"""After eviction, visiting a share link should re-render rather than 404."""
	_install_fake_run_skill(monkeypatch)
	cr = CanvasRuntime(db=db)

	class _SkillReg:
		def get(self, slug):
			return SimpleNamespace(slug=slug, name=slug, kind="background", controls=[], code="")
		def get_record(self, slug):
			return self.get(slug)

	runtime = SimpleNamespace(
		services={"canvas": cr, "skill_worker_pool": None},
		db=db, skill_registry=_SkillReg(), config={}, sessions={}, get_session=lambda key: None,
	)
	fe = fw.WebFrontend.__new__(fw.WebFrontend)
	fe.runtime = runtime
	fe.config = {}
	import threading
	fe._lock = threading.RLock()
	fe._outbox = {}

	# Seed a canvas and render once so canvas_pools has the row.
	key = "web:sess"
	cs = cr.for_session(key)
	cr.handle_action(cs.canvas_id, "add_layer", {
		"skill_slug": "fractal", "kind": "background", "controls": {},
	})
	rr = canvas_render.render_canvas(cs, skill_loader=runtime.skill_registry.get_record, seed=42, db=db)
	pool_hash = rr.pool_hash
	payload_before = fe.pool_share_payload(pool_hash)
	assert payload_before is not None
	original_path = Path(payload_before["image_path"])
	assert original_path.is_file()

	# Wipe the rendered files (simulating cache eviction). Keep the
	# canvas_pools row so the share link still resolves.
	shutil.rmtree(renders_dir / pool_hash)
	assert not (renders_dir / pool_hash).exists()

	# Re-request — should re-render lazily, not return None.
	payload_after = fe.pool_share_payload(pool_hash)
	assert payload_after is not None, "Expected lazy re-render, got 404"
	rerendered = Path(payload_after["image_path"])
	assert rerendered.is_file()
	assert rerendered.parent.name == pool_hash  # same pool, fresh file
	assert payload_after["pool_hash"] == pool_hash
