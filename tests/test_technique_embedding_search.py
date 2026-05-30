"""Tests for technique metadata embeddings and semantic search."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.database import Database
from plugins.techniques.helpers import technique_meta
from plugins.tasks.task_embed_techniques import EmbedTechniques
from plugins.tasks.task_index_techniques import IndexTechniques
from plugins.tools.tool_search_techniques import SearchTechniques


class FakeEmbedder:
    model_name = "fake"

    def encode(self, text):
        text = str(text).lower()
        return np.array([1.0, 0.0], dtype=np.float32) if any(w in text for w in ("water", "ocean", "wave")) else np.array([0.0, 1.0], dtype=np.float32)


def _fresh_db(name: str):
    path = Path(f".technique_embed_test_{name}.sqlite")
    path.unlink(missing_ok=True)
    db = Database(str(path))
    db.ensure_output_table("embed_techniques", EmbedTechniques.output_schema)
    return db, path


def _cleanup(db, path):
    db.conn.close()
    for p in (path, path.with_suffix(".sqlite-wal"), path.with_suffix(".sqlite-shm")):
        p.unlink(missing_ok=True)


def _technique_file(root: Path, name="technique_wave.py", hidden=False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(f"""
from plugins.BaseTechnique import BaseTechnique

class WaveTechnique(BaseTechnique):
    name = "Ocean Wave"
    description = "Water interference and coastal waves."
    kind = "background"
    hidden = {hidden}

    def run(self, canvas):
        pass
""".strip(), encoding="utf-8")
    return path


def test_embed_techniques_writes_technique_embedding(monkeypatch):
    root = Path(".technique_embed_dir_task")
    db, dbpath = _fresh_db("task")
    try:
        monkeypatch.setattr(technique_meta, "TECHNIQUE_DIRS", (root.resolve(),))
        path = _technique_file(root)
        result = EmbedTechniques().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and len(result.data) == 1
        db.write_outputs("technique_embeddings", result.data)
        row = db.conn.execute("SELECT slug, name, dim, model FROM technique_embeddings WHERE path = ?", (str(path),)).fetchone()
        assert dict(row) == {"slug": "ocean_wave", "name": "Ocean Wave", "dim": 2, "model": "fake"}
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_embed_techniques_skips_non_technique_python(monkeypatch):
    root = Path(".technique_embed_dir_skip")
    root.mkdir(exist_ok=True)
    db, dbpath = _fresh_db("skip")
    try:
        monkeypatch.setattr(technique_meta, "TECHNIQUE_DIRS", (root.resolve(),))
        path = root / "helper.py"
        path.write_text("print('x')", encoding="utf-8")
        result = EmbedTechniques().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and result.data == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_embed_techniques_skips_prefixed_helper_file(monkeypatch):
    root = Path(".technique_embed_dir_helper")
    helper = root / "helpers"
    helper.mkdir(parents=True, exist_ok=True)
    db, dbpath = _fresh_db("helper")
    try:
        monkeypatch.setattr(technique_meta, "TECHNIQUE_DIRS", (root.resolve(),))
        path = helper / "technique_store.py"
        path.write_text("def slugify(name):\n    return name\n", encoding="utf-8")
        result = EmbedTechniques().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and result.data == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_embed_techniques_skips_prefixed_non_technique_in_technique_dir(monkeypatch):
    root = Path(".technique_embed_dir_bad")
    root.mkdir(exist_ok=True)
    db, dbpath = _fresh_db("bad")
    try:
        monkeypatch.setattr(technique_meta, "TECHNIQUE_DIRS", (root.resolve(),))
        path = root / "technique_bad.py"
        path.write_text("class Bad: pass", encoding="utf-8")
        result = EmbedTechniques().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and result.data == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_search_techniques_ranks_vectors_and_filters_hidden():
    db, path = _fresh_db("search")
    try:
        db.write_outputs("technique_embeddings", [
            _row("a.py", "ocean_wave", "Ocean Wave", "Water waves", [1, 0]),
            _row("b.py", "flame", "Flame", "Fire sparks", [0, 1]),
            _row("c.py", "hidden_water", "Hidden Water", "Ocean", [1, 0], hidden=1),
        ])
        result = SearchTechniques().run(SimpleNamespace(db=db, services={"text_embedder": FakeEmbedder()}), query="ocean", limit=2)
        assert result.success
        assert [s["slug"] for s in result.data["techniques"]] == ["ocean_wave", "flame"]
    finally:
        _cleanup(db, path)


def test_search_techniques_dedupes_same_slug_across_paths():
	"""technique_embeddings is keyed on path — two paths sharing one slug must collapse to one result."""
	db, path = _fresh_db("dedup_slug")
	try:
		db.write_outputs("technique_embeddings", [
			# Same slug, different paths — exactly what triggers the duplication bug.
			_row("plugins/techniques/technique_voronoi.py", "voronoi_visage", "Voronoi Visage", "An abstract portrait.", [1, 0]),
			_row("art_kit/technique_voronoi.py", "voronoi_visage", "Voronoi Visage", "An abstract portrait.", [1, 0]),
			_row("c.py", "flame", "Flame", "Fire sparks", [0, 1]),
		])
		result = SearchTechniques().run(SimpleNamespace(db=db, services={"text_embedder": FakeEmbedder()}), query="ocean", limit=10)
		assert result.success
		slugs = [s["slug"] for s in result.data["techniques"]]
		assert slugs.count("voronoi_visage") == 1, f"expected 1 voronoi_visage, got {slugs}"
	finally:
		_cleanup(db, path)


def test_search_techniques_accepts_slug_alias():
    db, path = _fresh_db("alias")
    try:
        db.write_outputs("technique_embeddings", [_row("a.py", "ocean_wave", "Ocean Wave", "Water waves", [1, 0])])
        result = SearchTechniques().run(SimpleNamespace(db=db, services={"text_embedder": FakeEmbedder()}), slug="wave")
        assert result.success and result.data["techniques"][0]["slug"] == "ocean_wave"
    finally:
        _cleanup(db, path)


def test_search_techniques_can_filter_to_built_ins():
    db, path = _fresh_db("builtins")
    try:
        root = Path(__file__).resolve().parents[1]
        db.write_outputs("technique_embeddings", [
            _row(str(root / "plugins" / "techniques" / "technique_wave.py"), "built_wave", "Built Wave", "Water waves", [1, 0]),
            _row(str(root / "data" / "sandbox_techniques" / "technique_wave.py"), "sandbox_wave", "Sandbox Wave", "Water waves", [1, 0]),
        ])
        result = SearchTechniques().run(_ctx(db, {"weigh_popularity": False}), query="water", limit=10, built_in_only=True)
        assert result.success
        assert [s["slug"] for s in result.data["techniques"]] == ["built_wave"]
    finally:
        _cleanup(db, path)


def test_search_techniques_exposes_popularity_config():
    keys = {s[1]: s for s in SearchTechniques.config_settings}
    assert keys["weigh_popularity"][3] is True
    assert keys["popularity_alpha"][3] == 0.1


def test_search_techniques_popularity_breaks_equal_cosine():
    db, path = _fresh_db("pop_equal")
    try:
        db.write_outputs("technique_embeddings", [_row("a.py", "plain", "Plain", "Water", [1, 0]), _row("b.py", "popular", "Popular", "Water", [1, 0])])
        _score(db, "popular", shares=5)
        result = SearchTechniques().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 0.5}), query="water", limit=2)
        assert [s["slug"] for s in result.data["techniques"]] == ["popular", "plain"]
    finally:
        _cleanup(db, path)


def test_search_techniques_alpha_zero_is_pure_cosine():
    db, path = _fresh_db("alpha0")
    try:
        db.write_outputs("technique_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0]), _row("b.py", "fire", "Fire", "Flame", [0, 1])])
        _score(db, "fire", shares=99)
        result = SearchTechniques().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 0.0}), query="water", limit=2)
        assert result.data["techniques"][0]["slug"] == "water"
    finally:
        _cleanup(db, path)


def test_search_techniques_alpha_one_is_popularity_order():
    db, path = _fresh_db("alpha1")
    try:
        db.write_outputs("technique_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0]), _row("b.py", "fire", "Fire", "Flame", [0, 1])])
        _score(db, "fire", saves=9)
        result = SearchTechniques().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 1.0}), query="water", limit=2)
        assert result.data["techniques"][0]["slug"] == "fire"
    finally:
        _cleanup(db, path)


def test_search_techniques_can_disable_popularity():
    db, path = _fresh_db("pop_off")
    try:
        db.write_outputs("technique_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0]), _row("b.py", "fire", "Fire", "Flame", [0, 1])])
        _score(db, "fire", remixes=99)
        result = SearchTechniques().run(_ctx(db, {"weigh_popularity": False, "popularity_alpha": 1.0}), query="water", limit=2)
        assert result.data["techniques"][0]["slug"] == "water"
    finally:
        _cleanup(db, path)


def test_search_techniques_returns_all_popularity_fields():
    db, path = _fresh_db("pop_fields")
    try:
        db.write_outputs("technique_embeddings", [_row("a.py", "water", "Water", "Ocean", [1, 0])])
        _score(db, "water", shares=1, downloads=2, remixes=3, saves=4, link_opens=5)
        technique = SearchTechniques().run(_ctx(db, {}), query="water").data["techniques"][0]
        assert {k: technique[k] for k in ("shares", "downloads", "remixes", "saves", "link_opens")} == {
            "shares": 1.0, "downloads": 2.0, "remixes": 3.0, "saves": 4.0, "link_opens": 5.0,
        }
        assert set(("score", "embedding_rank", "bm25_rank")) <= set(technique)
    finally:
        _cleanup(db, path)


def test_technique_embedding_cleanup_uses_path_key():
    db, path = _fresh_db("cleanup")
    try:
        db.write_outputs("technique_embeddings", [_row("gone.py", "gone", "Gone", "Gone desc", [1, 0])])
        db.clean_output_tables("gone.py", ["technique_embeddings"])
        assert db.conn.execute("SELECT COUNT(*) AS n FROM technique_embeddings").fetchone()["n"] == 0
    finally:
        _cleanup(db, path)


def test_index_techniques_writes_fts_row(monkeypatch):
    root = Path(".technique_index_dir")
    db, dbpath = _fresh_db("index")
    try:
        monkeypatch.setattr(technique_meta, "TECHNIQUE_DIRS", (root.resolve(),))
        db.ensure_output_table("index_techniques", IndexTechniques.output_schema)
        path = _technique_file(root)
        result = IndexTechniques().run([str(path)], SimpleNamespace(db=db))[0]
        assert result.success
        rows = db.conn.execute("SELECT slug, name, kind FROM technique_fts WHERE path = ?", (str(path),)).fetchall()
        assert len(rows) == 1 and dict(rows[0]) == {"slug": "ocean_wave", "name": "Ocean Wave", "kind": "background"}
        # Re-indexing the same path must not duplicate the row.
        IndexTechniques().run([str(path)], SimpleNamespace(db=db))
        n = db.conn.execute("SELECT COUNT(*) AS n FROM technique_fts WHERE path = ?", (str(path),)).fetchone()["n"]
        assert n == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_search_bm25_lifts_exact_name_match_over_popular_neighbor():
    """The original bug: a low-engagement technique with an exact-name query is
    pushed out of the top results by popular neighbors. With FTS5/BM25 fused
    in via RRF, the named technique resurfaces."""
    db, path = _fresh_db("hybrid")
    try:
        db.ensure_output_table("index_techniques", IndexTechniques.output_schema)
        # All three techniques get the same query-side cosine (the fake embedder
        # is binary), so without BM25 the popularity boost on `popular_a`
        # would win even for a query that exactly names `target`.
        db.write_outputs("technique_embeddings", [
            _row("a.py", "popular_a", "Popular A", "Water and waves.", [1, 0]),
            _row("b.py", "popular_b", "Popular B", "Water and waves.", [1, 0]),
            _row("c.py", "target", "Chromatic Aberration", "Splits color channels.", [0, 1]),
        ])
        _score(db, "popular_a", shares=50)
        _score(db, "popular_b", saves=40)
        db.conn.execute(
            "INSERT INTO technique_fts (slug, name, description, kind, path, hidden) VALUES (?, ?, ?, ?, ?, 0)",
            ("popular_a", "Popular A", "Water and waves.", "background", "a.py"),
        )
        db.conn.execute(
            "INSERT INTO technique_fts (slug, name, description, kind, path, hidden) VALUES (?, ?, ?, ?, ?, 0)",
            ("popular_b", "Popular B", "Water and waves.", "background", "b.py"),
        )
        db.conn.execute(
            "INSERT INTO technique_fts (slug, name, description, kind, path, hidden) VALUES (?, ?, ?, ?, ?, 0)",
            ("target", "Chromatic Aberration", "Splits color channels.", "filter", "c.py"),
        )
        db.conn.commit()
        result = SearchTechniques().run(_ctx(db, {"weigh_popularity": True, "popularity_alpha": 0.25}),
                                     query="chromatic aberration", limit=3)
        assert result.success
        assert result.data["techniques"][0]["slug"] == "target"
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
        "INSERT OR REPLACE INTO technique_scores (slug, shares, downloads, remixes, saves, link_opens, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1)",
        (slug, cols["shares"], cols["downloads"], cols["remixes"], cols["saves"], cols["link_opens"]),
    )
    db.conn.commit()
