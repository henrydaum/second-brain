"""Runtime registry of loaded skills.

Populated by plugin_discovery at startup and kept in sync by the plugin
watcher. Tools read through this registry instead of touching the skill
files directly; persistence ops in ``skill_store`` write/edit files, then
ask the registry to reload the affected slug.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from plugins.BaseSkill import BaseSkill
from plugins.skills.helpers.skill_store import Skill, slugify, to_skill_record

logger = logging.getLogger("SkillRegistry")


class SkillRegistry:
    """Slug-keyed registry of live BaseSkill instances."""

    def __init__(self):
        self._lock = threading.RLock()
        self._skills: dict[str, BaseSkill] = {}

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

    def unregister(self, slug: str) -> BaseSkill | None:
        with self._lock:
            return self._skills.pop(slug, None)

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
