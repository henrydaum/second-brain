"""Embed agent-authored canvas skills (new BaseSkill class form).

Reads class-attribute metadata from ``skill_*.py`` files in the baked-in
``plugins/skills/`` directory and ``DATA_DIR/sandbox_skills/`` and writes a
row per skill into the ``skill_embeddings`` table for cross-process search
(e.g. the web frontend's gallery).
"""

from __future__ import annotations

import time
from pathlib import Path

from paths import SANDBOX_SKILLS, ROOT_DIR
from plugins.BaseTask import BaseTask, TaskResult
from plugins.skills.helpers.skill_store import _read_class_metadata

_BUILT_IN_SKILLS_PKG = ROOT_DIR / "plugins" / "skills"


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
        roots = (SANDBOX_SKILLS.resolve(), _BUILT_IN_SKILLS_PKG.resolve())
        skills = []
        for raw in paths:
            p = Path(raw)
            try:
                resolved = p.resolve()
            except Exception:
                skills.append(None); continue
            in_dir = any(root in resolved.parents for root in roots)
            if not in_dir or not p.name.startswith("skill_") or p.suffix != ".py":
                skills.append(None); continue
            try:
                src = p.read_text(encoding="utf-8")
                meta = _read_class_metadata(src)
                if not meta.get("name"):
                    skills.append(None); continue
                skills.append((p, meta, f"{meta.get('name')}\n{meta.get('description') or ''}"))
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
                    "name": str(meta.get("name") or p.stem),
                    "description": str(meta.get("description") or ""),
                    "kind": str(meta.get("kind") or "creation"),
                    "owner": str(meta.get("owner") or ""),
                    "embedding": v.astype("float32").tobytes(),
                    "model_name": embedder.model_name,
                    "embedded_at": now,
                }]))
        return out
