"""Phase 1 unit tests for pipeline.canvas_store.

Exercises CRUD, the layer-level cache, the sample-vs-mint seed rule, and
the user-action fan-out into skill_events / skill_scores. The renderer is
a fake that just returns a deterministic byte string so we can assert
cache hits/misses without the skill plugin system.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pipeline.canvas_store import CanvasStore, LayerSpec
from pipeline.database import Database


class _CountingRenderer:
	"""Fake renderer that counts invocations and returns deterministic bytes."""

	def __init__(self):
		"""Initialize the counting renderer."""
		self.calls: list[dict] = []

	def __call__(self, skill_slug, skill_kind, input_path, controls, seed, size, palette_id):
		"""Render call entry — record args, return distinguishable bytes."""
		self.calls.append({
			"skill_slug": skill_slug,
			"skill_kind": skill_kind,
			"input_path": str(input_path) if input_path else None,
			"controls": dict(controls),
			"seed": seed,
			"size": size,
			"palette_id": palette_id,
		})
		# Bytes differ by skill+seed so cache files are distinguishable on disk.
		return f"{skill_slug}:{seed}:{len(self.calls)}".encode("utf-8")


def _fresh(name: str) -> tuple[Database, Path, _CountingRenderer, CanvasStore]:
	"""Build a fresh DB + store rooted in a project-local test dir."""
	db_path = Path(f".canvas_store_test_{name}.sqlite")
	files_dir = Path(f".canvas_store_test_{name}_files")
	db_path.unlink(missing_ok=True)
	if files_dir.exists():
		shutil.rmtree(files_dir)
	db = Database(str(db_path))
	renderer = _CountingRenderer()
	store = CanvasStore(db, file_dir=files_dir, renderer=renderer)
	return db, db_path, renderer, store


def _cleanup(db: Database, db_path: Path) -> None:
	"""Internal helper: close + remove test fixtures."""
	db.conn.close()
	db_path.unlink(missing_ok=True)
	for sibling in (db_path.with_suffix(".sqlite-wal"), db_path.with_suffix(".sqlite-shm")):
		sibling.unlink(missing_ok=True)
	files_dir = Path(str(db_path).replace(".sqlite", "_files").replace(".canvas_store_test_", ".canvas_store_test_"))
	if files_dir.exists():
		shutil.rmtree(files_dir, ignore_errors=True)


def _seed_skills(store: CanvasStore) -> None:
	"""Internal helper: insert the two skills used across tests."""
	store.upsert_skill("fractal", kind="creation", name="Fractal")
	store.upsert_skill("swirl", kind="transform", name="Swirl")


# =============================================================
# SKILLS + CANVAS CRUD
# =============================================================

def test_upsert_skill_and_get():
	"""upsert_skill writes a row and get_skill round-trips."""
	db, db_path, _, store = _fresh("crud_skill")
	try:
		store.upsert_skill("fractal", kind="creation", name="Fractal", description="d")
		row = store.get_skill("fractal")
		assert row is not None
		assert row["slug"] == "fractal"
		assert row["kind"] == "creation"
		assert row["name"] == "Fractal"
		assert row["description"] == "d"
		# Upsert updates instead of duplicating.
		store.upsert_skill("fractal", kind="creation", name="Fractal v2")
		assert store.get_skill("fractal")["name"] == "Fractal v2"
	finally:
		_cleanup(db, db_path)


def test_create_canvas_and_get():
	"""create_canvas stores layers in order and get_canvas returns them."""
	db, db_path, _, store = _fresh("crud_canvas")
	try:
		_seed_skills(store)
		cid = store.create_canvas(
			[LayerSpec("fractal", {"zoom": 1.0}), LayerSpec("swirl", {"angle": 30})],
			title="t", size=512,
		)
		canvas = store.get_canvas(cid)
		assert canvas["id"] == cid
		assert canvas["title"] == "t"
		assert canvas["size"] == 512
		assert canvas["current_generation_id"] is None
		assert [layer["skill_slug"] for layer in canvas["layers"]] == ["fractal", "swirl"]
		assert canvas["layers"][0]["controls"] == {"zoom": 1.0}
		assert canvas["layers"][1]["position"] == 1
	finally:
		_cleanup(db, db_path)


def test_create_canvas_requires_layers():
	"""Empty layer list is rejected — a canvas always has at least one layer."""
	db, db_path, _, store = _fresh("crud_empty")
	try:
		try:
			store.create_canvas([])
		except ValueError:
			return
		raise AssertionError("expected ValueError for empty layers")
	finally:
		_cleanup(db, db_path)


def test_update_layer_controls():
	"""update_layer_controls swaps the editable controls and bumps updated_at."""
	db, db_path, _, store = _fresh("update_layer")
	try:
		_seed_skills(store)
		cid = store.create_canvas([LayerSpec("fractal", {"zoom": 1.0})])
		store.update_layer_controls(cid, 0, {"zoom": 2.5})
		canvas = store.get_canvas(cid)
		assert canvas["layers"][0]["controls"] == {"zoom": 2.5}
	finally:
		_cleanup(db, db_path)


# =============================================================
# RENDER PATH — CACHE
# =============================================================

def test_render_canvas_first_run_renders_each_layer():
	"""First render of a fresh canvas invokes the renderer once per layer."""
	db, db_path, renderer, store = _fresh("render_first")
	try:
		_seed_skills(store)
		cid = store.create_canvas(
			[LayerSpec("fractal", {"zoom": 1.0}), LayerSpec("swirl", {"angle": 30})],
			size=512,
		)
		path, gen_id = store.render_canvas(cid, conversation_id="c1")
		assert path.exists()
		assert gen_id > 0
		assert len(renderer.calls) == 2
		# Transform got the creation's output as its input.
		assert renderer.calls[0]["input_path"] is None
		assert renderer.calls[1]["input_path"] is not None
		# canvases.current_generation_id is updated.
		assert store.get_canvas(cid)["current_generation_id"] == gen_id
	finally:
		_cleanup(db, db_path)


def test_render_canvas_second_run_same_session_reuses_seeds_and_hits_cache():
	"""Rendering the same canvas a second time in the same conversation is fully cached."""
	db, db_path, renderer, store = _fresh("render_cache")
	try:
		_seed_skills(store)
		cid = store.create_canvas(
			[LayerSpec("fractal", {}), LayerSpec("swirl", {})], size=256,
		)
		store.render_canvas(cid, conversation_id="c1")
		calls_after_first = len(renderer.calls)
		assert calls_after_first == 2
		store.render_canvas(cid, conversation_id="c1")
		# Second call should be entirely cache hits — no new renders.
		assert len(renderer.calls) == calls_after_first
	finally:
		_cleanup(db, db_path)


def test_render_canvas_new_conversation_samples_existing_seed():
	"""A fresh conversation samples an already-cached seed instead of minting."""
	db, db_path, renderer, store = _fresh("render_sample")
	try:
		_seed_skills(store)
		cid = store.create_canvas([LayerSpec("fractal", {})], size=64)
		store.render_canvas(cid, conversation_id="c1")
		seed_first = renderer.calls[0]["seed"]
		# New conversation: should sample the existing seed -> cache hit -> no new render.
		store.render_canvas(cid, conversation_id="c2")
		assert len(renderer.calls) == 1  # still one — sampled the cached seed
		# Hit count on the generation row should have bumped.
		row = db.conn.execute(
			"SELECT seed, use_count FROM image_generations"
		).fetchone()
		assert row["seed"] == seed_first
		assert row["use_count"] >= 1
	finally:
		_cleanup(db, db_path)


def test_render_canvas_force_new_seed_mints_fresh():
	"""force_new_seed bypasses the sample step and renders with a new seed."""
	db, db_path, renderer, store = _fresh("render_force")
	try:
		_seed_skills(store)
		cid = store.create_canvas([LayerSpec("fractal", {})], size=64)
		store.render_canvas(cid, conversation_id="c1")
		store.render_canvas(cid, conversation_id="c2", force_new_seed=True)
		assert len(renderer.calls) == 2
		assert renderer.calls[0]["seed"] != renderer.calls[1]["seed"]
		# Pool now has two distinct seeds for the same pool_key.
		rows = db.conn.execute(
			"SELECT DISTINCT seed FROM image_generations"
		).fetchall()
		assert len({r["seed"] for r in rows}) == 2
	finally:
		_cleanup(db, db_path)


def test_changing_layer_controls_invalidates_downstream_cache_only():
	"""Editing a transform's controls re-renders that layer but reuses the creation's output."""
	db, db_path, renderer, store = _fresh("render_edit")
	try:
		_seed_skills(store)
		cid = store.create_canvas(
			[LayerSpec("fractal", {}), LayerSpec("swirl", {"angle": 30})],
			size=256,
		)
		store.render_canvas(cid, conversation_id="c1")
		assert len(renderer.calls) == 2
		# Tweak the transform's controls; creation is unchanged.
		store.update_layer_controls(cid, 1, {"angle": 90})
		store.render_canvas(cid, conversation_id="c1")
		# Exactly one additional render — the new swirl layer.
		assert len(renderer.calls) == 3
		assert renderer.calls[2]["skill_slug"] == "swirl"
		assert renderer.calls[2]["controls"] == {"angle": 90}
	finally:
		_cleanup(db, db_path)


def test_changing_creation_controls_invalidates_full_chain():
	"""Editing the creation layer also forces the transform to re-render (new input)."""
	db, db_path, renderer, store = _fresh("render_edit_base")
	try:
		_seed_skills(store)
		cid = store.create_canvas(
			[LayerSpec("fractal", {"zoom": 1.0}), LayerSpec("swirl", {"angle": 30})],
			size=256,
		)
		store.render_canvas(cid, conversation_id="c1")
		assert len(renderer.calls) == 2
		store.update_layer_controls(cid, 0, {"zoom": 9.0})
		store.render_canvas(cid, conversation_id="c1")
		# Both layers re-rendered since the creation's output (transform's
		# input) changed identity.
		assert len(renderer.calls) == 4
	finally:
		_cleanup(db, db_path)


# =============================================================
# USER ACTIONS
# =============================================================

def test_record_user_action_inserts_and_updates_skill_scores():
	"""A share action writes user_canvas_actions and bumps skill_scores."""
	db, db_path, _, store = _fresh("action_share")
	try:
		_seed_skills(store)
		cid = store.create_canvas(
			[LayerSpec("fractal", {}), LayerSpec("swirl", {})],
		)
		store.record_user_action(user_id="u1", canvas_id=cid, action="share")
		actions = db.conn.execute(
			"SELECT user_id, canvas_id, action FROM user_canvas_actions"
		).fetchall()
		assert len(actions) == 1
		assert actions[0]["action"] == "share"
		# Skill scores split 0.6 to creation, 0.4 to single transform.
		scores = {
			r["slug"]: r["shares"] for r in db.conn.execute(
				"SELECT slug, shares FROM skill_scores"
			).fetchall()
		}
		assert scores["fractal"] == 0.6
		assert scores["swirl"] == 0.4
	finally:
		_cleanup(db, db_path)


def test_link_open_action_updates_link_opens_column():
	"""link_open is wired through _KIND_FIELDS into the new link_opens column."""
	db, db_path, _, store = _fresh("action_link")
	try:
		_seed_skills(store)
		cid = store.create_canvas([LayerSpec("fractal", {})])
		store.record_user_action(user_id="u_anon", canvas_id=cid, action="link_open")
		row = db.conn.execute(
			"SELECT link_opens FROM skill_scores WHERE slug = 'fractal'"
		).fetchone()
		assert row is not None
		# Creation-only chain → full 1.0 weight on the single skill.
		assert row["link_opens"] == 1.0
	finally:
		_cleanup(db, db_path)


def test_list_user_canvases_returns_action_history():
	"""list_user_canvases returns canvases the user has acted on, newest first."""
	db, db_path, _, store = _fresh("action_list")
	try:
		_seed_skills(store)
		c1 = store.create_canvas([LayerSpec("fractal", {})])
		c2 = store.create_canvas([LayerSpec("fractal", {"zoom": 2.0})])
		store.record_user_action(user_id="u1", canvas_id=c1, action="save")
		store.record_user_action(user_id="u1", canvas_id=c2, action="save")
		store.record_user_action(user_id="u1", canvas_id=c1, action="share")
		saved = store.list_user_canvases("u1", "save")
		assert {c["id"] for c in saved} == {c1, c2}
		shared = store.list_user_canvases("u1", "share")
		assert [c["id"] for c in shared] == [c1]
	finally:
		_cleanup(db, db_path)
