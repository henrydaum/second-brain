"""Tests for skill metadata embeddings and semantic search."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.database import Database
from plugins.tasks import task_embed_skills
from plugins.tasks.task_embed_skills import EmbedSkills
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
    kind = "creation"
    hidden = {hidden}

    def run(self, canvas):
        pass
""".strip(), encoding="utf-8")
    return path


def test_embed_skills_writes_skill_embedding(monkeypatch):
    root = Path(".skill_embed_dir_task")
    db, dbpath = _fresh_db("task")
    try:
        monkeypatch.setattr(task_embed_skills, "SKILL_DIRS", (root.resolve(),))
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
        monkeypatch.setattr(task_embed_skills, "SKILL_DIRS", (root.resolve(),))
        path = root / "helper.py"
        path.write_text("print('x')", encoding="utf-8")
        result = EmbedSkills().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert result.success and result.data == []
    finally:
        shutil.rmtree(root, ignore_errors=True)
        _cleanup(db, dbpath)


def test_embed_skills_fails_invalid_skill_file(monkeypatch):
    root = Path(".skill_embed_dir_bad")
    root.mkdir(exist_ok=True)
    db, dbpath = _fresh_db("bad")
    try:
        monkeypatch.setattr(task_embed_skills, "SKILL_DIRS", (root.resolve(),))
        path = root / "skill_bad.py"
        path.write_text("class Bad: pass", encoding="utf-8")
        result = EmbedSkills().run([str(path)], SimpleNamespace(services={"text_embedder": FakeEmbedder()}))[0]
        assert not result.success and "BaseSkill" in result.error
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


def test_search_skills_accepts_slug_alias():
    db, path = _fresh_db("alias")
    try:
        db.write_outputs("skill_embeddings", [_row("a.py", "ocean_wave", "Ocean Wave", "Water waves", [1, 0])])
        result = SearchSkills().run(SimpleNamespace(db=db, services={"text_embedder": FakeEmbedder()}), slug="wave")
        assert result.success and result.data["skills"][0]["slug"] == "ocean_wave"
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


def _row(path, slug, name, desc, vec, hidden=0):
    arr = np.asarray(vec, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr) or 1.0)
    return {
        "path": path, "slug": slug, "name": name, "description": desc,
        "kind": "creation", "hidden": hidden, "embedding": arr.astype("<f4").tobytes(),
        "dim": int(arr.size), "model": "fake", "updated_at": 1.0,
    }
