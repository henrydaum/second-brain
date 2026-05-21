"""End-to-end test for the share → gallery / archive listing flow.

The bug this guards against: render_canvas was upgraded to accept ``db=``
but no frontend / tool caller passed it, so ``canvas_pools`` stayed
empty and every downstream pool_hash lookup (share page, QR, gallery,
archive) silently 404'd while the chat still said "Shared ✓".

This test wires the real WebFrontend (with a stubbed runtime) through
the public share() / save_canvas() / gallery() / archive_listing() /
share_qr_png() / pool_share_payload() methods and asserts each piece
sees the canvas it should.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from canvas import render as canvas_render
from canvas.runtime import CanvasRuntime
from pipeline.database import Database
import plugins.frontends.frontend_web as fw


def _install_fake_run_skill(monkeypatch):
	"""Stub the skill subprocess: write a tiny PNG instead."""
	def fake_run_skill(skill, *, params, palette, size, seed, input_image_path, output_image_path, **kwargs):
		Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
		Image.new("RGBA", (4, 4), (16, 32, 64, 255)).save(output_image_path, format="PNG")
		return {"ok": True}
	monkeypatch.setattr(canvas_render, "run_skill", fake_run_skill)


@pytest.fixture
def renders_dir(request, monkeypatch):
	"""Project-local canvas_renders dir, cleaned up after."""
	target = Path(f".share_e2e_test_{request.node.name}")
	if target.exists():
		shutil.rmtree(target, ignore_errors=True)
	target.mkdir(parents=True, exist_ok=True)
	monkeypatch.setattr(canvas_render, "RENDERS_DIR", target)
	yield target
	shutil.rmtree(target, ignore_errors=True)


def _make_frontend(db):
	"""Build a WebFrontend instance with a stubbed runtime + skill registry."""
	cr = CanvasRuntime(db=db)

	class _SkillReg:
		def get(self, slug):
			return SimpleNamespace(slug=slug, kind="creation", code="")
		def get_record(self, slug):
			return SimpleNamespace(slug=slug, kind="creation", code="")

	runtime = SimpleNamespace(
		services={"canvas": cr},
		db=db,
		skill_registry=_SkillReg(),
		config={},
		sessions={},
		get_session=lambda key: None,
	)
	fe = fw.WebFrontend.__new__(fw.WebFrontend)
	fe.runtime = runtime
	fe.config = {}
	# WebFrontend's lock + outbox are normally set up in BaseFrontend.__init__;
	# we don't need them for these synchronous calls but stash empty defaults.
	import threading
	fe._lock = threading.RLock()
	fe._outbox = {}
	return fe


def _seed_canvas(fe, session_id="sess1"):
	"""Drop a single creation layer on the session's canvas so it has something to share."""
	key = fe.session_key(session_id) if hasattr(fe, "session_key") else f"web:{session_id}"
	cr = fe.runtime.services["canvas"]
	cs = cr.for_session(key)
	cr.handle_action(cs.canvas_id, "add_layer", {
		"skill_slug": "fractal", "kind": "creation", "controls": {},
	})
	return key, cs


def _fresh_db(name: str) -> tuple[Database, Path]:
	"""Project-local DB."""
	path = Path(f".share_e2e_db_{name}.sqlite")
	path.unlink(missing_ok=True)
	return Database(str(path)), path


def _cleanup_db(db, path):
	"""Close + remove DB."""
	db.conn.close()
	for p in (path, path.with_suffix(".sqlite-wal"), path.with_suffix(".sqlite-shm")):
		p.unlink(missing_ok=True)


# =================================================================
# Share + gallery
# =================================================================

def test_share_populates_gallery_listing(monkeypatch, renders_dir):
	"""After share(), the canvas appears in gallery() with title/artist."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("gallery")
	try:
		fe = _make_frontend(db)
		_seed_canvas(fe)
		events = fe.share("sess1", "Sunset", "Alice", ip="127.0.0.1", account_id="alice")
		# share returns a share_link event and a message
		assert any(e.get("type") == "share_link" for e in events), events
		share_link = next(e for e in events if e.get("type") == "share_link")
		assert share_link.get("url", "").endswith(f"/share/{share_link['share_id']}")
		assert share_link.get("qr_url", "").endswith("/qr.png")

		# Gallery now has one item with the right metadata.
		listing = fe.gallery("sess1", limit=10, offset=0)
		assert listing["total"] == 1
		assert len(listing["items"]) == 1
		item = listing["items"][0]
		assert item["title"] == "Sunset"
		assert item["artist"] == "Alice"
		assert item["pool_hash"] == share_link["share_id"]
		assert item["path"] == share_link["share_id"]  # for remix buttons
		assert item["url"]  # image URL present
	finally:
		_cleanup_db(db, dbpath)


def test_save_populates_archive_listing(monkeypatch, renders_dir):
	"""After save_canvas(), the canvas appears in this user's archive."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("archive")
	try:
		fe = _make_frontend(db)
		_seed_canvas(fe)
		fe.save_canvas("sess1", "127.0.0.1", "alice", title="My piece")

		listing = fe.archive_listing("sess1", "127.0.0.1", "alice", limit=10, offset=0)
		assert listing["total"] == 1
		assert listing["items"][0]["title"] == "My piece"

		# A DIFFERENT user's archive must NOT see it.
		other = fe.archive_listing("sess2", "127.0.0.2", "bob", limit=10, offset=0)
		assert other["total"] == 0
	finally:
		_cleanup_db(db, dbpath)


# =================================================================
# Share page + QR resolve
# =================================================================

def test_pool_share_payload_resolves_after_render(monkeypatch, renders_dir):
	"""Once we've rendered, the pool_hash resolves to share payload."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("payload")
	try:
		fe = _make_frontend(db)
		_seed_canvas(fe)
		fe.share("sess1", "T", "A", ip="127.0.0.1", account_id="alice")
		# Pull the pool_hash from the gallery so we don't reach inside.
		ph = fe.gallery("sess1")["items"][0]["pool_hash"]

		payload = fe.pool_share_payload(ph)
		assert payload is not None
		assert payload["pool_hash"] == ph
		assert Path(payload["image_path"]).exists()
	finally:
		_cleanup_db(db, dbpath)


def test_qr_png_is_generated_after_share(monkeypatch, renders_dir):
	"""share_qr_png returns a valid PNG for a shared pool_hash."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("qr")
	try:
		fe = _make_frontend(db)
		_seed_canvas(fe)
		# share_qr_png consults _base_url; stub to a deterministic value.
		fe._base_url = lambda: "http://localhost:8765"
		fe.share("sess1", "T", "A", ip="127.0.0.1", account_id="alice")
		ph = fe.gallery("sess1")["items"][0]["pool_hash"]

		raw = fe.share_qr_png(ph)
		assert raw is not None
		assert raw[:8] == b"\x89PNG\r\n\x1a\n"
	finally:
		_cleanup_db(db, dbpath)


def test_share_page_link_open_scores_but_qr_does_not(monkeypatch, renders_dir):
	"""HTML share-page opens count; QR generation does not."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("linkopen")
	try:
		fe = _make_frontend(db)
		fe._base_url = lambda: "http://localhost:8765"
		_seed_canvas(fe)
		fe.share("sess1", "T", "A", ip="127.0.0.1", account_id="alice")
		ph = fe.gallery("sess1")["items"][0]["pool_hash"]
		payload = fe.pool_share_payload(ph)

		fe.record_link_open(ph, "127.0.0.1", "", payload)
		fe.share_qr_png(ph)

		actions = db.conn.execute("SELECT COUNT(*) AS n FROM user_canvas_actions WHERE action = 'link_open'").fetchone()["n"]
		score = db.conn.execute("SELECT link_opens FROM skill_scores WHERE slug = 'fractal'").fetchone()["link_opens"]
		assert actions == 1
		assert score == 1.0
	finally:
		_cleanup_db(db, dbpath)


def test_share_route_redirects_to_landing_with_share_loaded(monkeypatch, renders_dir):
	"""GET /share/{pool_hash} redirects to /?share=... and records link_open."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("redirect")
	try:
		fe = _make_frontend(db)
		_seed_canvas(fe)
		fe.share("sess1", "T", "A", ip="127.0.0.1", account_id="alice")
		ph = fe.gallery("sess1")["items"][0]["pool_hash"]
		redirects = []
		handler = fw._Handler.__new__(fw._Handler)
		handler.path = f"/share/{ph}"
		handler.server = SimpleNamespace(frontend=fe)
		handler.client_address = ("127.0.0.1", 12345)
		handler.headers = {}
		handler._redirect = lambda location, extra_headers=(): redirects.append(location)
		handler.send_error = lambda code: pytest.fail(f"unexpected send_error({code})")

		handler.do_GET()

		assert redirects == [f"/?share={ph}"]
		assert db.conn.execute("SELECT COUNT(*) AS n FROM user_canvas_actions WHERE action = 'link_open'").fetchone()["n"] == 1
	finally:
		_cleanup_db(db, dbpath)


def test_share_deep_link_boot_does_not_race_initial_canvas_load():
	"""Share boot should remix instead of also firing the normal loadCanvas path."""
	app_js = Path("plugins/frontends/web/app.js").read_text(encoding="utf-8")
	assert "const bootingShare = new URLSearchParams(location.search).has(\"share\");" in app_js
	assert "if (bootingShare) handleShareDeepLink(); else loadCanvas();" in app_js
