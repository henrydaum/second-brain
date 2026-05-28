"""Tests for skill metadata embeddings and semantic search."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.database import Database
from plugins.skills.helpers import skill_meta
from plugins.tasks.task_embed_skills import EmbedSkills
from plugins.tasks.task_index_skills import IndexSkills
from plugins.tools.tool_search_skills import SearchSkills


class FakeEmbedder:
    model_name = "fake"

    def encode(self, text):
        text = str(text).lower()
        return np.array([1.0, 0.0], dtype=np.float32) if any(w in text for w in ("water", "ocean", "wave")) else np.array([0.0, 1.0], dtype=np.float32)


def _fresh_db(name: str):
    path = Path(f".skill_embed_test_{name}.sqlite")
    path.unlink(missing_ok=True)
    db = Database(str(path))
    db.ensure_output_table("embed_skills", EmbedSkills.output_schema)
    return db, path


def _cleanup(db, path):
    db.conn.close()
    for p in (path, path.with_suffix(".sqlite-wal"), path.with_suffix(".sqlite-shm")):
        p.unlink(missing_ok=True)


def _skill_file(root: Path, name="skill_wave.py", hidden=False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(f"""
from plugins.BaseSkill import BaseSkill

class WaveSkill(BaseSkill):
    name = "Ocean Wave"
    description = "Water interference and coastal waves."
    kind = "background"
    hidden = {hidden}

    def run(self, canvas):
        pass
""".strip(), encoding="utf-8")
    return path


def test_embed_skills_writes_skill_embedding(monkeypatch):
    root = Path(".skill_embed_dir_task")
    db, dbpath = _fresh_db("task")
    try:
        monkeypatch.setattr(skill_meta, "SKILL_DIRS", (root.resolve(),))
        path = _skill_file(root)
        result = EmbedSkills().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and len(result.data) == 1
        db.write_outputs("skill_embeddings", result.data)
        row = db.conn.execute("SELECT slug, name, dim, model FROM skill_embeddings WHERE path = ?", (str(path),)).fetchone()
        assert dict(row) == {"slug": "ocean_wave", "name": "Ocean Wave", "dim": 2, "model": "fake"}
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_embed_skills_skips_non_skill_python(monkeypatch):
    root = Path(".skill_embed_dir_skip")
    root.mkdir(exist_ok=True)
    db, dbpath = _fresh_db("skip")
    try:
        monkeypatch.setattr(skill_meta, "SKILL_DIRS", (root.resolve(),))
        path = root / "helper.py"
        path.write_text("print('x')", encoding="utf-8")
        result = EmbedSkills().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and result.data == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_embed_skills_skips_prefixed_helper_file(monkeypatch):
    root = Path(".skill_embed_dir_helper")
    helper = root / "helpers"
    helper.mkdir(parents=True, exist_ok=True)
    db, dbpath = _fresh_db("helper")
    try:
        monkeypatch.setattr(skill_meta, "SKILL_DIRS", (root.resolve(),))
        path = helper / "skill_store.py"
        path.write_text("def slugify(name):\n    return name\n", encoding="utf-8")
        result = EmbedSkills().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and result.data == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_embed_skills_skips_prefixed_non_skill_in_skill_dir(monkeypatch):
    root = Path(".skill_embed_dir_bad")
    root.mkdir(exist_ok=True)
    db, dbpath = _fresh_db("bad")
    try:
        monkeypatch.setattr(skill_meta, "SKILL_DIRS", (root.resolve(),))
        path = root / "skill_bad.py"
        path.write_text("class Bad: pass", encoding="utf-8")
        result = EmbedSkills().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and result.data == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_search_skills_ranks_vectors_and_filters_hidden():
    db, path = _fresh_db("search")
    try:
        db.write_outputs("skill_embeddings", [
            _row("a.py", "ocean_wave", "Ocean Wave", "Water waves", [1, 0]),
            _row("b.py", "flame", "Flame", "Fire sparks", [0, 1]),
            _row("c.py", "hidden_water", "Hidden Water", "Ocean", [1, 0], hidden=1),
        ])
        result = SearchSkills().run(SimpleNamespace(db=db, services={"text_embedder": FakeEmbedder()}), query="ocean", limit=2)
        assert result.success
        assert [s["slug"] for s in result.data["skills"]] == ["ocean_wave", "flame"]
    finally:
        _cleanup(db, path)


def test_search_skills_dedupes_same_slug_across_paths():
	"""skill_embeddings is keyed on path — two paths sharing one slug must collapse to one result."""
	db, path = _fresh_db("dedup_slug")
	try:
		db.write_outputs("skill_embeddings", [
			# Same slug, different paths — exactly what triggers the duplication bug.
			_row("plugins/skills/skill_voronoi.py", "voronoi_visage", "Voronoi Visage", "An abstract portrait.", [1, 0]),
			_row("art_kit/skill_voronoi.py", "voronoi_visage", "Voronoi Visage", "An abstract portrait.", [1, 0]),
			_row("c.py", "flame", "Flame", "Fire sparks", [0, 1]),
		])
		result = SearchSkills().run(SimpleNamespace(db=db, services={"text_embedder": FakeEmbedder()}), query="ocean", limit=10)
		assert result.success
		slugs = [s["slug"] for s in result.data["skills"]]
		assert slugs.count("voronoi_visage") == 1, f"expected 1 voronoi_visage, got {slugs}"
	finally:
		_cleanup(db, path)


def test_search_skills_accepts_slug_alias():
    db, path = _fresh_db("alias")
    try:
        db.write_outputs("skill_embeddings", [_row("a.py", "ocean_wave", "Ocean Wave", "Water waves", [1, 0])])
        result = SearchSkills().run(SimpleNamespace(db=db, services={"text_embedder": FakeEmbedder()}), slug="wave")
        assert result.success and result.data["skills"][0]["slug"] == "ocean_wave"
    finally:
        _cleanup(db, path)


def test_search_skills_can_filter_to_built_ins():
    db, path = _fresh_db("builtins")
    try:
        root = Path(__file__).resolve().parents[1]
        db.write_outputs("skill_embeddings", [
            _row(str(root / "plugins" / "skills" / "skill_wave.py"), "built_wave", "Built Wave", "Water waves", [1, 0]),
            _row(str(root / "data" / "sandbox_skills" / "skill_wave.py"), "sandbox_wave", "Sandbox Wave", "Water waves", [1, 0]),
        ])
        result = SearchSkills().run(_ctx(db, {"weigh_popularity": False}), query="water", limit=10, built_in_only=True)
        assert result.success
        assert [s["slug"] for s in result.data["skills"]] == ["built_wave"]
    finally:
        _cleanup(db, path)


def test_search_skills_exposes_popularity_config():
    keys = {s[1]: s for s in SearchSkills.config_settings}
    assert keys["weigh_popularity"][3] is True
    assert keys["popularity_alpha"][3] == 0.1


def test_search_skills_popularity_breaks_equal_cosine():
    db, path = _fresh_db("pop_equal")
    try:
        db.write_outputs("skill_embeddings", [_row("a.py", "plain", "Plain", "Water", [1, 0]), _row("b.py", "popular", "Popular", "Water", [1, 0])])
        _score(db, "popular", shares=5)
        result = SearchSkills().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 0.5}), query="water", limit=2)
        assert [s["slug"] for s in result.data["skills"]] == ["popular", "plain"]
    finally:
        _cleanup(db, path)


def test_search_skills_alpha_zero_is_pure_cosine():
    db, path = _fresh_db("alpha0")
    try:
        db.write_outputs("skill_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0]), _row("b.py", "fire", "Fire", "Flame", [0, 1])])
        _score(db, "fire", shares=99)
        result = SearchSkills().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 0.0}), query="water", limit=2)
        assert result.data["skills"][0]["slug"] == "water"
    finally:
        _cleanup(db, path)


def test_search_skills_alpha_one_is_popularity_order():
    db, path = _fresh_db("alpha1")
    try:
        db.write_outputs("skill_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0]), _row("b.py", "fire", "Fire", "Flame", [0, 1])])
        _score(db, "fire", saves=9)
        result = SearchSkills().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 1.0}), query="water", limit=2)
        assert result.data["skills"][0]["slug"] == "fire"
    finally:
        _cleanup(db, path)


def test_search_skills_can_disable_popularity():
    db, path = _fresh_db("pop_off")
    try:
        db.write_outputs("skill_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0]), _row("b.py", "fire", "Fire", "Flame", [0, 1])])
        _score(db, "fire", remixes=99)
        result = SearchSkills().run(_ctx(db, {"weigh_popularity": False, "popularity_alpha": 1.0}), query="water", limit=2)
        assert result.data["skills"][0]["slug"] == "water"
    finally:
        _cleanup(db, path)


def test_search_skills_returns_all_popularity_fields():
    db, path = _fresh_db("pop_fields")
    try:
        db.write_outputs("skill_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0])])
        _score(db, "water", shares=1, downloads=2, remixes=3, saves=4, link_opens=5)
        skill = SearchSkills().run(_ctx(db, {}), query="water").data["skills"][0]
        assert {k: skill[k] for k in ("shares", "downloads", "remixes", "saves", "link_opens")} == {
            "shares": 1.0, "downloads": 2.0, "remixes": 3.0, "saves": 4.0, "link_opens": 5.0,
        }
        assert set(("score", "embedding_rank", "bm25_rank")) <= set(skill)
    finally:
        _cleanup(db, path)


def test_skill_embedding_cleanup_uses_path_key():
    db, path = _fresh_db("cleanup")
    try:
        db.write_outputs("skill_embeddings", [_row("gone.py", "gone", "Gone", "Gone desc", [1, 0])])
        db.clean_output_tables("gone.py", ["skill_embeddings"])
        assert db.conn.execute("SELECT COUNT(*) AS n FROM skill_embeddings").fetchone()["n"] == 0
    finally:
        _cleanup(db, path)


def test_index_skills_writes_fts_row(monkeypatch):
    root = Path(".skill_index_dir")
    db, dbpath = _fresh_db("index")
    try:
        monkeypatch.setattr(skill_meta, "SKILL_DIRS", (root.resolve(),))
        db.ensure_output_table("index_skills", IndexSkills.output_schema)
        path = _skill_file(root)
        result = IndexSkills().run([str(path)], SimpleNamespace(db=db))[0]
        assert result.success
        rows = db.conn.execute("SELECT slug, name, kind FROM skill_fts WHERE path = ?", (str(path),)).fetchall()
        assert len(rows) == 1 and dict(rows[0]) == {"slug": "ocean_wave", "name": "Ocean Wave", "kind": "background"}
        # Re-indexing the same path must not duplicate the row.
        IndexSkills().run([str(path)], SimpleNamespace(db=db))
        n = db.conn.execute("SELECT COUNT(*) AS n FROM skill_fts WHERE path = ?", (str(path),)).fetchone()["n"]
        assert n == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_search_bm25_lifts_exact_name_match_over_popular_neighbor():
    """The original bug: a low-engagement skill with an exact-name query is
    pushed out of the top results by popular neighbors. With FTS5/BM25 fused
    in via RRF, the named skill resurfaces."""
    db, path = _fresh_db("hybrid")
    try:
        db.ensure_output_table("index_skills", IndexSkills.output_schema)
        # All three skills get the same query-side cosine (the fake embedder
        # is binary), so without BM25 the popularity boost on `popular_a`
        # would win even for a query that exactly names `target`.
        db.write_outputs("skill_embeddings", [
            _row("a.py", "popular_a", "Popular A", "Water and waves.", [1, 0]),
            _row("b.py", "popular_b", "Popular B", "Water and waves.", [1, 0]),
            _row("c.py", "target", "Chromatic Aberration", "Splits color channels.", [0, 1]),
        ])
        _score(db, "popular_a", shares=50)
        _score(db, "popular_b", saves=40)
        db.conn.execute(
            "INSERT INTO skill_fts (slug, name, description, kind, path, hidden) VALUES (?, ?, ?, ?, ?, 0)",
            ("popular_a", "Popular A", "Water and waves.", "background", "a.py"),
        )
        db.conn.execute(
            "INSERT INTO skill_fts (slug, name, description, kind, path, hidden) VALUES (?, ?, ?, ?, ?, 0)",
            ("popular_b", "Popular B", "Water and waves.", "background", "b.py"),
        )
        db.conn.execute(
            "INSERT INTO skill_fts (slug, name, description, kind, path, hidden) VALUES (?, ?, ?, ?, ?, 0)",
            ("target", "Chromatic Aberration", "Splits color channels.", "filter", "c.py"),
        )
        db.conn.commit()
        result = SearchSkills().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 0.25}),
                                     query="chromatic aberration", limit=3)
        assert result.success
        assert result.data["skills"][0]["slug"] == "target"
    finally:
        _cleanup(db, path)


def _row(path, slug, name, desc, vec, hidden=0):
    arr = np.asarray(vec, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr) or 1.0)
    return {
        "path": path, "slug": slug, "name": name, "description": desc,
        "kind": "background", "hidden": hidden, "embedding": arr.astype("<f4").tobytes(),
        "dim": int(arr.size), "model": "fake", "updated_at": 1.0,
    }


def _ctx(db, config):
    return SimpleNamespace(db=db, config=config, services={"text_embedder": FakeEmbedder()})


def _score(db, slug, **vals):
    cols = {"shares": 0, "downloads": 0, "remixes": 0, "saves": 0, "link_opens": 0, **vals}
    db.conn.execute(
        "INSERT OR REPLACE INTO skill_scores (slug, shares, downloads, remixes, saves, link_opens, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1)",
        (slug, cols["shares"], cols["downloads"], cols["remixes"], cols["saves"], cols["link_opens"]),
    )
    db.conn.commit()
