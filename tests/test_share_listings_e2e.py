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
from billing.credits import CreditsService
from pipeline.database import Database
import plugins.frontends.frontend_web as fw
from plugins.tools.tool_manage_layers import ManageLayers


def _install_fake_run_skill(monkeypatch):
	"""Stub the skill subprocess: write a tiny PNG instead."""
	calls = []
	def fake_run_skill(skill, *, params, palette, size, seed, input_image_path, output_image_path, **kwargs):
		calls.append({"slug": getattr(skill, "slug", ""), "params": params, "seed": seed})
		Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
		Image.new("RGBA", (4, 4), (16, 32, 64, 255)).save(output_image_path, format="PNG")
		return {"ok": True}
	monkeypatch.setattr(canvas_render, "run_skill", fake_run_skill)
	return calls


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


def _make_frontend(db, skills=None):
	"""Build a WebFrontend instance with a stubbed runtime + skill registry."""
	cr = CanvasRuntime(db=db)
	skills = skills or {}

	class _SkillReg:
		def get(self, slug):
			return skills.get(slug) or SimpleNamespace(slug=slug, name=slug, kind="background", controls=[], code="")
		def get_record(self, slug):
			return self.get(slug)

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
	"""Drop a single background layer on the session's canvas so it has something to share."""
	key = fe.session_key(session_id) if hasattr(fe, "session_key") else f"web:{session_id}"
	cr = fe.runtime.services["canvas"]
	cs = cr.for_session(key)
	cr.handle_action(cs.canvas_id, "add_layer", {
		"skill_slug": "fractal", "kind": "background", "controls": {},
	})
	return key, cs


def test_canvas_payload_keeps_palette_controls_per_layer():
	palette = {"type": "palette", "label": "Palette"}
	runtime = SimpleNamespace(skill_registry=SimpleNamespace(
		get_record=lambda slug: SimpleNamespace(slug=slug, name=slug.title(), kind="background", controls=[palette]),
	))
	payload = fw._canvas_payload_full(runtime, "web:s", {"chain": [
		{"slug": "a", "kind": "background", "controls": {"palette": "frost"}},
		{"slug": "b", "kind": "filter", "controls": {"palette": "ember"}},
	]})
	assert len(payload["controls_panels"]) == 2
	assert [p["values"]["palette"] for p in payload["controls_panels"]] == ["frost", "ember"]
	assert [p["schema"][0]["type"] for p in payload["controls_panels"]] == ["palette", "palette"]


def test_canvas_payload_includes_layers_without_controls():
	runtime = SimpleNamespace(skill_registry=SimpleNamespace(
		get_record=lambda slug: SimpleNamespace(slug=slug, name=slug.title(), kind="background", controls=[]),
	))
	payload = fw._canvas_payload_full(runtime, "web:s", {"chain": [{"slug": "plain", "kind": "background", "controls": {}}]})
	assert payload["controls_panels"][0]["slug"] == "plain"
	assert payload["controls_panels"][0]["schema"] == []


def test_move_layer_reorders_and_rejects_background_displacement(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("move_layer")
	try:
		skills = {
			"base": SimpleNamespace(slug="base", name="Base", kind="background", controls=[], code=""),
			"a": SimpleNamespace(slug="a", name="A", kind="filter", controls=[], code=""),
			"b": SimpleNamespace(slug="b", name="B", kind="filter", controls=[], code=""),
		}
		fe = _make_frontend(db, skills=skills)
		key, cs = _seed_canvas(fe)
		cr = fe.runtime.services["canvas"]
		cs.canvas.layers[0]["slug"] = "base"
		cs.canvas.layers[0]["kind"] = "background"
		cr.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "a", "kind": "filter"})
		cr.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "b", "kind": "filter"})

		events = fe.move_layer("sess1", 2, 1)
		assert events[0]["type"] == "hero_image"
		assert [l["slug"] for l in cr.for_session(key).canvas.layers] == ["base", "b", "a"]
		bad = fe.move_layer("sess1", 1, 0)
		assert bad[0]["type"] == "error"
	finally:
		_cleanup_db(db, dbpath)


def test_regenerate_applies_staged_controls_without_new_seed(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	seeds = iter([111, 222])
	monkeypatch.setattr(canvas_render, "_mint_seed", lambda: next(seeds))
	db, dbpath = _fresh_db("regen_staged")
	try:
		fe = _make_frontend(db)
		key, cs = _seed_canvas(fe)
		assert fe.regenerate("sess1", force_new_seed=True)[0]["type"] == "hero_image"
		old_seed = cs.render_seed

		events = fe.regenerate("sess1", controls=[
			{"chain_index": 0, "name": "zoom", "value": 2.5},
			{"chain_index": 0, "name": "label", "value": "kept"},
		])

		assert events[0]["type"] == "hero_image"
		assert cs.canvas.layers[0]["controls"] == {"zoom": 2.5, "label": "kept"}
		assert cs.render_seed == old_seed
		assert fw._canvas_payload_full(fe.runtime, key, {"chain": cs.canvas.layers})["controls_panels"][0]["values"]["zoom"] == 2.5
	finally:
		_cleanup_db(db, dbpath)


def test_randomize_forces_new_seed(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	seeds = iter([111, 222])
	monkeypatch.setattr(canvas_render, "_mint_seed", lambda: next(seeds))
	db, dbpath = _fresh_db("regen_randomize")
	try:
		fe = _make_frontend(db)
		_, cs = _seed_canvas(fe)
		fe.regenerate("sess1", force_new_seed=True)
		old_seed = cs.render_seed
		events = fe.regenerate("sess1", force_new_seed=True)
		assert events[0]["type"] == "hero_image"
		assert cs.render_seed == 222
		assert cs.render_seed != old_seed
	finally:
		_cleanup_db(db, dbpath)


def test_regenerate_invalid_staged_control_skips_render(monkeypatch, renders_dir):
	calls = _install_fake_run_skill(monkeypatch)
	seeds = iter([111, 222])
	monkeypatch.setattr(canvas_render, "_mint_seed", lambda: next(seeds))
	db, dbpath = _fresh_db("regen_invalid_control")
	try:
		fe = _make_frontend(db)
		_, cs = _seed_canvas(fe)
		fe.regenerate("sess1", force_new_seed=True)
		calls.clear()
		old_seed = cs.render_seed

		events = fe.regenerate("sess1", controls=[{"chain_index": 99, "name": "zoom", "value": 2}])

		assert events[0]["type"] == "error"
		assert calls == []
		assert cs.render_seed == old_seed
		assert "zoom" not in cs.canvas.layers[0].get("controls", {})
	finally:
		_cleanup_db(db, dbpath)


def test_regenerate_render_failure_rolls_back_staged_controls(monkeypatch, renders_dir):
	def fake_run_skill(skill, *, params, output_image_path, **_kwargs):
		if params.get("bad"):
			raise TypeError("bad control")
		Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
		Image.new("RGBA", (4, 4), (16, 32, 64, 255)).save(output_image_path, format="PNG")
		return {"ok": True}
	monkeypatch.setattr(canvas_render, "run_skill", fake_run_skill)
	db, dbpath = _fresh_db("regen_render_rollback")
	try:
		fe = _make_frontend(db)
		_, cs = _seed_canvas(fe)
		assert fe.regenerate("sess1", force_new_seed=True)[0]["type"] == "hero_image"
		before = cs.to_dict()

		events = fe.regenerate("sess1", controls=[{"chain_index": 0, "name": "bad", "value": True}])

		assert events[0]["type"] == "error"
		assert "bad control" in events[0]["content"]
		assert cs.to_dict() == before
	finally:
		_cleanup_db(db, dbpath)


def test_delete_render_failure_rolls_back_layer_change(monkeypatch, renders_dir):
	def fake_run_skill(*_args, **_kwargs):
		raise RuntimeError("render exploded")
	monkeypatch.setattr(canvas_render, "run_skill", fake_run_skill)
	db, dbpath = _fresh_db("delete_render_rollback")
	try:
		fe = _make_frontend(db)
		_, cs = _seed_canvas(fe)
		cr = fe.runtime.services["canvas"]
		cr.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "swirl", "kind": "filter"})
		before = cs.to_dict()

		events = fe.delete_layer("sess1", 1)

		assert events[0]["type"] == "error"
		assert "render exploded" in events[0]["content"]
		assert cs.to_dict() == before
	finally:
		_cleanup_db(db, dbpath)


def test_delete_background_clears_canvas_in_web_frontend(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("delete_background_clears")
	try:
		fe = _make_frontend(db)
		_, cs = _seed_canvas(fe)
		cr = fe.runtime.services["canvas"]
		cr.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "swirl", "kind": "filter"})

		events = fe.delete_layer("sess1", 0)

		assert events == [{"type": "canvas_reset"}]
		assert cs.canvas.layers == []
	finally:
		_cleanup_db(db, dbpath)


def test_manual_render_denial_surfaces_shared_credit_error_and_restores_state(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("manual_credit_denial")
	try:
		fe = _make_frontend(db)
		key, cs = _seed_canvas(fe)
		credits = CreditsService({"web_credits": {"free": {"five_hours": 0, "week": 0}}})
		credits.bind_web_session(db, key, "sess1")
		fe.runtime.services["credits"] = credits
		before = cs.canvas.palette_id

		events = fe.set_palette("sess1", "obsidian")

		assert events[0]["error"]["code"] == "out_of_credits"
		assert cs.canvas.palette_id == before
	finally:
		_cleanup_db(db, dbpath)


def test_download_render_denial_surfaces_shared_credit_error_without_mutation(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("download_credit_denial")
	try:
		fe = _make_frontend(db)
		key, cs = _seed_canvas(fe)
		credits = CreditsService({"web_credits": {"free": {"five_hours": 0, "week": 0}}})
		credits.bind_web_session(db, key, "sess1")
		fe.runtime.services["credits"] = credits
		before = cs.to_dict()

		events = fe.render_for_download("sess1", 2)

		assert events[0]["error"]["code"] == "out_of_credits"
		assert cs.to_dict() == before
	finally:
		_cleanup_db(db, dbpath)


def test_agent_manage_layers_clear_pushes_canvas_reset(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("agent_clear_reset")
	fe = None
	try:
		fe = fw.WebFrontend()
		cr = CanvasRuntime(db=db)
		runtime = SimpleNamespace(
			services={"canvas": cr},
			db=db,
			skill_registry=SimpleNamespace(get_record=lambda slug: SimpleNamespace(slug=slug, name=slug, kind="background", controls=[], code="")),
			config={},
			sessions={},
			get_session=lambda key: None,
		)
		fe.bind(runtime, commands=None, config={})
		key, cs = _seed_canvas(fe)
		runtime.sessions[key] = SimpleNamespace()
		assert cs.canvas.layers

		result = ManageLayers().run(SimpleNamespace(
			session_key=key,
			canvas=cr,
			skill_registry=runtime.skill_registry,
			db=db,
			config={},
			services={},
		), action="clear")

		assert result.success
		assert cs.canvas.layers == []
		assert fe._drain(key)[0]["type"] == "canvas_reset"
	finally:
		if fe is not None:
			try:
				fe.unbind()
			except Exception:
				pass
		_cleanup_db(db, dbpath)


# =================================================================
# Skill search picker — backend endpoints
# =================================================================

def _make_frontend_with_skills(db, skill_specs):
	"""Frontend with a registry whose list_records() returns the given specs."""
	cr = CanvasRuntime(db=db)

	class _SkillReg:
		def __init__(self, specs):
			self._by_slug = {s["slug"]: SimpleNamespace(**s) for s in specs}
		def get(self, slug):
			return self._by_slug.get(slug)
		def get_record(self, slug):
			return self._by_slug.get(slug)
		def list_records(self, include_hidden=False):
			return list(self._by_slug.values())

	specs = [{**s, "controls": s.get("controls", []), "code": ""} for s in skill_specs]
	runtime = SimpleNamespace(
		services={"canvas": cr},
		db=db,
		skill_registry=_SkillReg(specs),
		config={},
		sessions={},
		get_session=lambda key: None,
	)
	fe = fw.WebFrontend.__new__(fw.WebFrontend)
	fe.runtime = runtime
	fe.config = {}
	import threading
	fe._lock = threading.RLock()
	fe._outbox = {}
	return fe


def test_skills_payload_returns_registered_skills():
	"""skills_payload exposes the registry as a lightweight list."""
	db, dbpath = _fresh_db("skills_payload")
	try:
		fe = _make_frontend_with_skills(db, [
			{"slug": "mandelbrot_explorer", "name": "Mandelbrot Explorer", "description": "Zoomable fractal.", "kind": "background"},
			{"slug": "generate_mandelbrot", "name": "Generate Mandelbrot", "description": "Static fractal.", "kind": "background"},
			{"slug": "swirl", "name": "Swirl", "description": "Twists pixels.", "kind": "filter"},
		])
		out = fe.skills_payload()
		assert {r["slug"] for r in out} == {"mandelbrot_explorer", "generate_mandelbrot", "swirl"}
		row = next(r for r in out if r["slug"] == "swirl")
		assert row == {"slug": "swirl", "name": "Swirl", "description": "Twists pixels.", "kind": "filter"}
	finally:
		_cleanup_db(db, dbpath)


def test_add_layer_via_search_endpoint(monkeypatch, renders_dir):
	"""POST /api/add_layer adds the chosen skill and re-renders."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("add_layer_endpoint")
	try:
		fe = _make_frontend_with_skills(db, [
			{"slug": "fractal", "name": "Fractal", "description": "", "kind": "background"},
			{"slug": "grain", "name": "Grain", "description": "", "kind": "filter"},
		])
		# Establish a canvas via the runtime (matches _seed_canvas).
		key = fe.session_key("sess1")
		cr = fe.runtime.services["canvas"]
		cs = cr.for_session(key)
		cr.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "fractal", "kind": "background", "controls": {}})

		events = fe.add_layer("sess1", "grain")

		assert events and events[0]["type"] == "hero_image"
		assert [layer["slug"] for layer in cs.canvas.layers] == ["fractal", "grain"]
	finally:
		_cleanup_db(db, dbpath)


def test_add_layer_unknown_slug_returns_error():
	"""POST /api/add_layer with an unknown slug doesn't mutate canvas."""
	db, dbpath = _fresh_db("add_layer_unknown")
	try:
		fe = _make_frontend_with_skills(db, [
			{"slug": "fractal", "name": "Fractal", "description": "", "kind": "background"},
		])
		key = fe.session_key("sess1")
		cr = fe.runtime.services["canvas"]
		cs = cr.for_session(key)
		before = list(cs.canvas.layers)

		events = fe.add_layer("sess1", "nonexistent_skill")

		assert events[0]["type"] == "error"
		assert "nonexistent_skill" in events[0]["content"]
		assert cs.canvas.layers == before
	finally:
		_cleanup_db(db, dbpath)


def test_add_layer_background_replaces_existing(monkeypatch, renders_dir):
	"""Adding a background skill replaces the existing background (Canvas.push_chain_entry semantics)."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("add_layer_bg_replace")
	try:
		fe = _make_frontend_with_skills(db, [
			{"slug": "fractal", "name": "Fractal", "description": "", "kind": "background"},
			{"slug": "voronoi", "name": "Voronoi", "description": "", "kind": "background"},
			{"slug": "grain", "name": "Grain", "description": "", "kind": "filter"},
		])
		key = fe.session_key("sess1")
		cr = fe.runtime.services["canvas"]
		cs = cr.for_session(key)
		cr.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "fractal", "kind": "background", "controls": {}})
		cr.handle_action(cs.canvas_id, "add_layer", {"skill_slug": "grain", "kind": "filter", "controls": {}})

		# Swapping background should drop the grain layer too (background reset clears chain).
		events = fe.add_layer("sess1", "voronoi")

		assert events and events[0]["type"] == "hero_image"
		assert [layer["slug"] for layer in cs.canvas.layers] == ["voronoi"]
	finally:
		_cleanup_db(db, dbpath)


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


def test_magic_link_claims_guest_saved_and_shared_canvases(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("guest_claim")
	try:
		fe = _make_frontend(db)
		sid, ip = "guest", "127.0.0.1"
		_seed_canvas(fe, sid)
		fe.save_canvas(sid, ip, "", title="Kept")
		fe.share(sid, "Posted", "Guest", ip=ip)
		guest_id = fe._anon_user_id(sid, ip)

		assert fe.request_magic_link("artist@example.com", sid, ip)["ok"]
		token = db.conn.execute("SELECT token FROM web_auth_tokens WHERE email = 'artist@example.com'").fetchone()["token"]
		account_id = fe.verify_magic_link(token)

		assert fe.archive_listing(sid, ip, account_id)["items"][0]["title"] == "Kept"
		rows = db.conn.execute(
			"SELECT action, user_id FROM user_canvas_actions WHERE action IN ('save', 'share')"
		).fetchall()
		assert {(row["action"], row["user_id"]) for row in rows} == {("save", account_id), ("share", account_id)}
	finally:
		_cleanup_db(db, dbpath)


def test_checkout_claims_guest_saved_and_shared_canvases(monkeypatch, renders_dir):
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("guest_checkout_claim")
	try:
		fe = _make_frontend(db)
		sid, ip = "buyer", "127.0.0.1"
		_seed_canvas(fe, sid)
		fe.save_canvas(sid, ip, "")
		fe.share(sid, "Posted", "Guest", ip=ip)
		guest_id = fe._anon_user_id(sid, ip)
		monkeypatch.setattr("billing.storefront.stripe.verify_webhook", lambda *_args: {
			"id": "checkout-1",
			"type": "checkout.session.completed",
			"data": {"object": {
				"customer_email": "buyer@example.com", "amount_total": 299,
				"metadata": {"session_id": sid, "anon_user_id": guest_id, "ip_hash": fe._ip_hash(ip)},
			}},
		})

		assert fe.handle_stripe_webhook(b"{}", "sig")["ok"]
		account_id = db.conn.execute("SELECT account_id FROM web_users WHERE email = 'buyer@example.com'").fetchone()["account_id"]
		rows = db.conn.execute(
			"SELECT action, user_id FROM user_canvas_actions WHERE action IN ('save', 'share')"
		).fetchall()
		assert {(row["action"], row["user_id"]) for row in rows} == {("save", account_id), ("share", account_id)}
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


def test_share_route_serves_og_html_with_redirect(monkeypatch, renders_dir):
	"""GET /share/{pool_hash} serves an HTML page containing OG/Twitter Card
	tags + a meta-refresh + JS redirect to /?share=... and records
	link_open. Crawlers see OG tags; browsers see the redirect."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("og_html")
	try:
		fe = _make_frontend(db)
		fe._base_url = lambda: "http://localhost:8765"
		_seed_canvas(fe)
		fe.share("sess1", "Mandelbrot Bay", "Henry", ip="127.0.0.1", account_id="alice")
		ph = fe.gallery("sess1")["items"][0]["pool_hash"]

		body_chunks = []
		handler = fw._Handler.__new__(fw._Handler)
		handler.path = f"/share/{ph}"
		handler.server = SimpleNamespace(frontend=fe)
		handler.client_address = ("127.0.0.1", 12345)
		handler.headers = {}
		handler._redirect = lambda location, extra_headers=(): pytest.fail(f"unexpected 303 redirect to {location}")
		handler.send_error = lambda code: pytest.fail(f"unexpected send_error({code})")
		# Capture _html() output via a stub.
		handler._html = lambda body, status=200: body_chunks.append(body)

		handler.do_GET()

		assert len(body_chunks) == 1
		html = body_chunks[0]
		# OG / Twitter Card tags present and well-formed.
		assert 'property="og:title"' in html
		assert 'property="og:image"' in html
		assert 'name="twitter:card" content="summary_large_image"' in html
		assert f"http://localhost:8765/share/{ph}/image.png" in html
		assert "Mandelbrot Bay by Henry" in html
		# Redirect mechanics for human browsers.
		assert f'url=/?share={ph}' in html             # meta-refresh
		assert f'"/?share={ph}"' in html               # inline JS replace target
		# Link-open is still recorded for the share visit.
		assert db.conn.execute("SELECT COUNT(*) AS n FROM user_canvas_actions WHERE action = 'link_open'").fetchone()["n"] == 1
	finally:
		_cleanup_db(db, dbpath)


def test_share_og_html_escapes_user_supplied_strings(monkeypatch, renders_dir):
	"""Title/artist must be HTML-escaped — they come from user input."""
	_install_fake_run_skill(monkeypatch)
	db, dbpath = _fresh_db("og_escape")
	try:
		fe = _make_frontend(db)
		fe._base_url = lambda: "http://localhost:8765"
		_seed_canvas(fe)
		fe.share("sess1", '<script>alert(1)</script>', 'O"Brien & Co', ip="127.0.0.1", account_id="alice")
		ph = fe.gallery("sess1")["items"][0]["pool_hash"]

		html = fe.share_og_html(ph)
		assert html is not None
		# Raw script tag must NOT appear; the escaped form must.
		assert "<script>alert(1)</script>" not in html
		assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
		# Quote in artist name escaped inside attribute context.
		assert 'O"Brien' not in html  # raw quote would break attribute
		assert "O&quot;Brien" in html
		assert "&amp; Co" in html
	finally:
		_cleanup_db(db, dbpath)


def test_share_og_html_returns_none_for_unknown_pool():
	"""Unknown pool_hash → None → caller falls back to a plain redirect."""
	fe = SimpleNamespace(pool_share_payload=lambda _ph: None)
	# Bind the real method, then invoke it with our stub `self`.
	assert fw.WebFrontend.share_og_html(fe, "deadbeef") is None


def test_broken_share_route_redirects_home():
	"""Invalid /share/{pool_hash} page links fall back to the main app."""
	fe = SimpleNamespace(pool_share_payload=lambda _share_id: None)
	redirects = []
	handler = fw._Handler.__new__(fw._Handler)
	handler.path = "/share/not-real"
	handler.server = SimpleNamespace(frontend=fe)
	handler.client_address = ("127.0.0.1", 12345)
	handler.headers = {}
	handler._redirect = lambda location, extra_headers=(): redirects.append(location)
	handler.send_error = lambda code: pytest.fail(f"unexpected send_error({code})")

	handler.do_GET()

	assert redirects == ["/"]


def test_share_deep_link_boot_does_not_race_initial_canvas_load():
	"""Share boot should remix instead of also firing the normal loadCanvas path."""
	app_js = Path("plugins/frontends/web/app.js").read_text(encoding="utf-8")
	assert "const bootingShare = new URLSearchParams(location.search).has(\"share\");" in app_js
	assert "if (bootingShare) handleShareDeepLink(); else loadCanvas();" in app_js
