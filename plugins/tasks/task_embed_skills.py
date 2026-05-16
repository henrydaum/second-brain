"""Embed agent-authored canvas skills."""

from __future__ import annotations

import time
from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult
from plugins.helpers.skill_store import metadata_from_source
from paths import SKILLS_DIR


class EmbedSkills(BaseTask):
    name = "embed_skills"
    modalities = ["text"]
    reads = []
    writes = ["skill_embeddings"]
    requires_services = ["text_embedder"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS skill_embeddings (
            path TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            kind TEXT,
            owner TEXT,
            embedding BLOB,
            model_name TEXT,
            embedded_at REAL
        );
    """
    batch_size = 12
    timeout = 120

    def run(self, paths, context):
        embedder = context.services.get("text_embedder")
        if not embedder or not embedder.loaded:
            return [TaskResult.failed("text_embedder service not loaded") for _ in paths]
        skills = []
        for raw in paths:
            p = Path(raw)
            if SKILLS_DIR.resolve() not in p.resolve().parents or not p.name.endswith(".skill.py"):
                skills.append(None); continue
            try:
                src = p.read_text(encoding="utf-8")
                meta = metadata_from_source(src)
                skills.append((p, meta, f"{meta.get('SKILL_NAME') or p.stem}\n{meta.get('SKILL_DESCRIPTION') or ''}"))
            except Exception as e:
                skills.append(e)
        texts = [s[2] for s in skills if isinstance(s, tuple)]
        vectors = embedder.encode(texts) if texts else []
        if texts and vectors is None:
            return [TaskResult.failed("skill embedding failed") if isinstance(s, tuple) else TaskResult() for s in skills]
        now = time.time(); vi = 0; out = []
        for item in skills:
            if item is None:
                out.append(TaskResult())
            elif isinstance(item, Exception):
                out.append(TaskResult.failed(str(item)))
            else:
                p, meta, _ = item; v = vectors[vi]; vi += 1
                out.append(TaskResult(data=[{
                    "path": str(p.resolve()),
                    "name": str(meta.get("SKILL_NAME") or p.stem),
                    "description": str(meta.get("SKILL_DESCRIPTION") or ""),
                    "kind": str(meta.get("SKILL_KIND") or "creation"),
                    "owner": str(meta.get("SKILL_OWNER") or ""),
                    "embedding": v.astype("float32").tobytes(),
                    "model_name": embedder.model_name,
                    "embedded_at": now,
                }]))
        return out
