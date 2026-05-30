"""Runtime registry of loaded techniques.

Populated by plugin_discovery at startup and kept in sync by the plugin
watcher. Tools read through this registry instead of touching the technique
files directly; persistence ops in ``technique_store`` write/edit files, then
ask the registry to reload the affected slug.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from plugins.BaseTechnique import BaseTechnique
from plugins.techniques.helpers.technique_store import Technique, slugify, to_technique_record

logger = logging.getLogger("TechniqueRegistry")


class TechniqueRegistry:
    """Slug-keyed registry of live BaseTechnique instances."""

    def __init__(self):
        self._lock = threading.RLock()
        self._techniques: dict[str, BaseTechnique] = {}

    # -- registration ------------------------------------------------------

    def register(self, instance: BaseTechnique) -> None:
        slug = slugify(getattr(instance, "name", "") or "")
        if not slug:
            logger.warning("Skipping technique with no name: %r", instance)
            return
        with self._lock:
            existing = self._techniques.get(slug)
            if existing is not None and existing is not instance:
                logger.info("Technique '%s' replaced by reload", slug)
            self._techniques[slug] = instance

    def unregister(self, slug: str) -> BaseTechnique | None:
        with self._lock:
            return self._techniques.pop(slug, None)

    def unregister_by_source(self, source_path: str | Path) -> list[str]:
        """Drop every technique loaded from a given file."""
        target = str(Path(source_path).resolve()) if source_path else ""
        if not target:
            return []
        removed: list[str] = []
        with self._lock:
            for slug, inst in list(self._techniques.items()):
                if str(Path(getattr(inst, "_source_path", "") or "").resolve()) == target:
                    self._techniques.pop(slug, None)
                    removed.append(slug)
        return removed

    # -- lookup ------------------------------------------------------------

    def get(self, slug: str) -> BaseTechnique | None:
        with self._lock:
            return self._techniques.get(slug)

    def get_record(self, slug: str) -> Technique | None:
        """Return a runner-facing Technique DTO for ``slug`` (or None)."""
        inst = self.get(slug)
        return to_technique_record(inst) if inst is not None else None

    def list(self, *, include_hidden: bool = False) -> list[BaseTechnique]:
        with self._lock:
            items = list(self._techniques.values())
        if not include_hidden:
            items = [s for s in items if not getattr(s, "hidden", False)]
        items.sort(key=lambda s: float(getattr(s, "created_at", 0.0) or 0.0), reverse=True)
        return items

    def list_records(self, *, include_hidden: bool = False) -> list[Technique]:
        return [to_technique_record(s) for s in self.list(include_hidden=include_hidden)]
