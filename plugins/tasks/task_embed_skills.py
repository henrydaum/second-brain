"""Path task that embeds canvas skill metadata for semantic search."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from plugins.BaseTask import BaseTask, TaskResult
from plugins.skills.helpers.skill_meta import is_skill_module, read_skill_meta


class EmbedSkills(BaseTask):
    name = "embed_skills"
    modalities = ["text"]
    writes = ["skill_embeddings"]
    requires_services = ["text_embedder"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS skill_embeddings (
            path TEXT PRIMARY KEY,
            slug TEXT,
            name TEXT,
            description TEXT,
            kind TEXT,
            hidden INTEGER DEFAULT 0,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            model TEXT,
            updated_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_skill_embeddings_hidden ON skill_embeddings(hidden);
        CREATE INDEX IF NOT EXISTS idx_skill_embeddings_slug ON skill_embeddings(slug);
    """
    batch_size = 16

    def run(self, paths: list[str], context) -> list[TaskResult]:
        embedder = (getattr(context, "services", {}) or {}).get("text_embedder")
        if embedder is None:
            return [TaskResult.failed("text_embedder service unavailable") for _ in paths]
        return [_embed_path(path, embedder) for path in paths]


def _embed_path(path: str, embedder) -> TaskResult:
    p = Path(path)
    if not is_skill_module(p):
        return TaskResult()
    try:
        meta = read_skill_meta(p)
        if meta is None:
            return TaskResult()
        vec = _norm(embedder.encode(meta["name"] + "\n\n" + meta["description"]))
        if vec is None:
            return TaskResult.failed("text_embedder returned no embedding")
        return TaskResult(data=[{
            **meta, "path": str(p), "embedding": vec.astype("<f4").tobytes(),
            "dim": int(vec.size), "model": str(getattr(embedder, "model_name", "") or ""),
            "updated_at": time.time(),
        }])
    except Exception as e:
        return TaskResult.failed(str(e))


def _norm(raw):
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if n else arr
