"""Path task that embeds canvas skill metadata for semantic search."""

from __future__ import annotations

import ast
import time
from pathlib import Path

import numpy as np

from paths import ROOT_DIR, SANDBOX_SKILLS
from plugins.BaseTask import BaseTask, TaskResult
from plugins.skills.helpers.skill_store import slugify

SKILL_DIRS = ((ROOT_DIR / "plugins" / "skills").resolve(), SANDBOX_SKILLS.resolve())


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
    if not _is_skill_module(p):
        return TaskResult()
    try:
        meta = _skill_meta(p)
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


def _in_skill_dir(path: Path) -> bool:
    try:
        r = path.resolve()
        return any(r.parent == d for d in SKILL_DIRS)
    except Exception:
        return False


def _is_skill_module(path: Path) -> bool:
    return path.suffix.lower() == ".py" and path.name.startswith("skill_") and _in_skill_dir(path)


def _skill_meta(path: Path) -> dict:
    code = path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    if not any(isinstance(n, ast.ImportFrom) and n.module == "plugins.BaseSkill" and any(a.name == "BaseSkill" for a in n.names) for n in tree.body):
        return None
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and any(_base_name(b) == "BaseSkill" for b in n.bases)), None)
    if cls is None:
        return None
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"name", "description", "kind", "hidden"}}
    name = str(vals.get("name") or "").strip()
    desc = str(vals.get("description") or "").strip()
    if not name or not desc:
        raise ValueError("skill file must declare non-empty name and description")
    return {
        "slug": slugify(name) or path.stem.removeprefix("skill_"),
        "name": name,
        "description": desc,
        "kind": str(vals.get("kind") or "background"),
        "hidden": int(bool(vals.get("hidden", False))),
    }


def _base_name(node) -> str:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


def _norm(raw):
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.size == 0:
        return None
    n = float(np.linalg.norm(arr))
    return arr / n if n else arr
