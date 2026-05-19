"""Runtime registry of loaded skills, plus embedding cache and search.

Populated by plugin_discovery at startup and kept in sync by the plugin
watcher. Tools read through this registry instead of touching the skill
files directly; persistence ops in ``skill_store`` write/edit files, then
ask the registry to reload the affected slug.

Search is cosine-similarity over text-embedder vectors of ``f"{name}\\n
{description}"``, multiplied by an implicit-signal boost from
``skill_scoring``. Falls back to a lexical substring match when the
embedder is unavailable so the tool still works during local dev.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable

from plugins.BaseSkill import BaseSkill
from plugins.skills.helpers.skill_store import Skill, slugify, to_skill_record

logger = logging.getLogger("SkillRegistry")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    # Text embedder outputs are L2-normalized.
    return float(sum(x * y for x, y in zip(a, b)))


def _embed_text(text_embedder, text: str) -> list[float] | None:
    if not text_embedder:
        return None
    if not getattr(text_embedder, "loaded", False):
        try:
            text_embedder.load()
        except Exception:
            return None
    if not getattr(text_embedder, "loaded", False):
        return None
    try:
        vec = text_embedder.encode([text])[0]
        return [float(x) for x in vec]
    except Exception:
        return None


class SkillRegistry:
    """Slug-keyed registry of live BaseSkill instances."""

    def __init__(self):
        self._lock = threading.RLock()
        self._skills: dict[str, BaseSkill] = {}
        # slug -> {"mtime": float, "vector": list[float]}
        self._emb_cache: dict[str, dict] = {}

    # -- registration ------------------------------------------------------

    def register(self, instance: BaseSkill) -> None:
        slug = slugify(getattr(instance, "name", "") or "")
        if not slug:
            logger.warning("Skipping skill with no name: %r", instance)
            return
        with self._lock:
            existing = self._skills.get(slug)
            if existing is not None and existing is not instance:
                logger.info("Skill '%s' replaced by reload", slug)
            self._skills[slug] = instance
            self._emb_cache.pop(slug, None)

    def unregister(self, slug: str) -> BaseSkill | None:
        with self._lock:
            inst = self._skills.pop(slug, None)
            self._emb_cache.pop(slug, None)
        return inst

    def unregister_by_source(self, source_path: str | Path) -> list[str]:
        """Drop every skill loaded from a given file."""
        target = str(Path(source_path).resolve()) if source_path else ""
        if not target:
            return []
        removed: list[str] = []
        with self._lock:
            for slug, inst in list(self._skills.items()):
                if str(Path(getattr(inst, "_source_path", "") or "").resolve()) == target:
                    self._skills.pop(slug, None)
                    self._emb_cache.pop(slug, None)
                    removed.append(slug)
        return removed

    # -- lookup ------------------------------------------------------------

    def get(self, slug: str) -> BaseSkill | None:
        with self._lock:
            return self._skills.get(slug)

    def get_record(self, slug: str) -> Skill | None:
        """Return a runner-facing Skill DTO for ``slug`` (or None)."""
        inst = self.get(slug)
        return to_skill_record(inst) if inst is not None else None

    def list(self, *, include_hidden: bool = False) -> list[BaseSkill]:
        with self._lock:
            items = list(self._skills.values())
        if not include_hidden:
            items = [s for s in items if not getattr(s, "hidden", False)]
        items.sort(key=lambda s: float(getattr(s, "created_at", 0.0) or 0.0), reverse=True)
        return items

    def list_records(self, *, include_hidden: bool = False) -> list[Skill]:
        return [to_skill_record(s) for s in self.list(include_hidden=include_hidden)]

    # -- search ------------------------------------------------------------

    def warm_embeddings(self, text_embedder) -> None:
        """Reconcile the in-memory embedding cache against current registrations."""
        with self._lock:
            existing = list(self._skills.items())
        for slug, inst in existing:
            src_path = Path(getattr(inst, "_source_path", "") or "")
            mtime = src_path.stat().st_mtime if src_path.is_file() else 0.0
            cached = self._emb_cache.get(slug)
            if cached and cached.get("mtime") == mtime:
                continue
            text = f"{getattr(inst, 'name', '')}\n{getattr(inst, 'description', '')}"
            vec = _embed_text(text_embedder, text)
            if vec is None:
                continue
            with self._lock:
                self._emb_cache[slug] = {"mtime": mtime, "vector": vec}
        # Prune stale cache entries.
        with self._lock:
            stale = [s for s in self._emb_cache if s not in self._skills]
            for s in stale:
                self._emb_cache.pop(s, None)

    def search(
        self, query: str, *, top_k: int = 5, kind: str | None = None,
        text_embedder=None, include_hidden: bool = False,
        score_lookup=None,
    ) -> list[dict]:
        """Return up to ``top_k`` ranked matches as plain dicts.

        ``score_lookup``: optional callable ``(iterable[slug]) -> dict[slug, stats]``
        for blending implicit signals. Pass
        ``functools.partial(skill_scoring.get_scores, db)`` from the caller
        to enable the boost; pass None to skip it.
        """
        self.warm_embeddings(text_embedder)
        qvec = _embed_text(text_embedder, query) if text_embedder else None

        with self._lock:
            entries = []
            for slug, inst in self._skills.items():
                if kind and getattr(inst, "kind", "") != kind:
                    continue
                if getattr(inst, "hidden", False) and not include_hidden:
                    continue
                entries.append((slug, inst))

        if qvec is None:
            q = (query or "").lower()
            out: list[dict] = []
            for slug, inst in entries:
                haystack = f"{getattr(inst, 'name', '')}\n{getattr(inst, 'description', '')}".lower()
                score = 1.0 if (q and q in haystack) else 0.0
                out.append(_row_for(slug, inst, score))
            out.sort(key=lambda r: r["score"], reverse=True)
            return out[:top_k]

        with self._lock:
            cache = dict(self._emb_cache)

        stats_map: dict = {}
        if score_lookup is not None:
            try:
                stats_map = score_lookup([s for s, _ in entries]) or {}
            except Exception:
                stats_map = {}

        rows: list[dict] = []
        for slug, inst in entries:
            vec = (cache.get(slug) or {}).get("vector")
            if vec is None:
                continue
            cos = _cosine(qvec, vec)
            multiplier = 1.0
            if stats_map:
                try:
                    from plugins.skills.helpers.skill_scoring import search_multiplier
                    multiplier = search_multiplier(stats_map.get(slug) or {})
                except Exception:
                    multiplier = 1.0
            rows.append(_row_for(slug, inst, cos * multiplier))
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:top_k]


def _row_for(slug: str, inst: BaseSkill, score: float) -> dict:
    return {
        "slug": slug,
        "name": getattr(inst, "name", "") or slug,
        "description": getattr(inst, "description", "") or "",
        "kind": getattr(inst, "kind", "") or "creation",
        "owner": getattr(inst, "owner", "") or "",
        "score": float(score),
    }
