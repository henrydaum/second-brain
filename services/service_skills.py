"""Skills registry service.

A *skill* is a folder containing a ``SKILL.md`` — markdown instructions for a
specific kind of task, with a tiny frontmatter header::

    ---
    name: trip-planning
    description: How to research and plan a multi-day trip.
    ---
    <instructions...>

Skills live under the tree-level ``helpers/skills/`` folder of each plugin
root (kernel ``plugins/``, sandbox, installed), mirroring the kernel tree's
cross-family ``helpers/`` layout. Discovery scans all three roots the same
way the parser service scans ``services/helpers/parse_*.py``; later roots
win on name collision, matching plugin discovery precedence.

Progressive disclosure: only each skill's name + description goes into the
system prompt (via ``agent_prompt_for``); the agent pulls the full text with
the ``use_skill`` tool when a task matches.
"""

from __future__ import annotations

dependencies_files = []
dependencies_pip = []

import logging
from dataclasses import dataclass
from pathlib import Path

from plugins.BaseService import BaseService, EXTENSION
from plugins.helpers.plugin_paths import PLUGIN_ROOTS

logger = logging.getLogger("SkillsService")

SKILL_FILENAME = "SKILL.md"
MAX_DESCRIPTION_CHARS = 200


@dataclass(frozen=True)
class Skill:
    """One discovered skill."""
    name: str
    description: str
    path: Path        # the SKILL.md file
    root_name: str    # which plugin root it came from


def skills_dir(root_path: Path) -> Path:
    """The skills folder inside one plugin root."""
    return root_path / "helpers" / "skills"


class SkillsService(BaseService):
    """Scan plugin roots for skills and serve their content."""
    model_name = "skills"
    shared = True
    lifecycle = EXTENSION

    def __init__(self, config: dict | None = None):
        super().__init__()
        self._config = config or {}
        self._cache: dict[Path, tuple[float, Skill | None]] = {}

    def _load(self) -> bool:
        self.scan()
        self.loaded = True
        return True

    def unload(self) -> None:
        self._cache.clear()
        self.loaded = False

    # ──────────────────────────────────────────────────────────────────
    # Registry
    # ──────────────────────────────────────────────────────────────────

    def scan(self) -> dict[str, Skill]:
        """Discover skills across all plugin roots (later roots win)."""
        found: dict[str, Skill] = {}
        for root in PLUGIN_ROOTS:
            base = skills_dir(root.path)
            if not base.is_dir():
                continue
            for skill_file in sorted(base.glob(f"*/{SKILL_FILENAME}")):
                skill = self._parse_cached(skill_file, root.name)
                if skill is not None:
                    found[skill.name] = skill
        return found

    def list(self) -> list[Skill]:
        """All skills, sorted by name."""
        return sorted(self.scan().values(), key=lambda s: s.name)

    def get(self, name: str) -> Skill | None:
        """One skill by name (case-insensitive)."""
        skills = self.scan()
        return skills.get(name) or skills.get((name or "").strip().lower())

    def read(self, name: str) -> str | None:
        """Full SKILL.md text for one skill, plus a support-file listing."""
        skill = self.get(name)
        if skill is None:
            return None
        text = skill.path.read_text(encoding="utf-8")
        support = sorted(
            p.relative_to(skill.path.parent).as_posix()
            for p in skill.path.parent.rglob("*")
            if p.is_file() and p.name != SKILL_FILENAME
        )[:50]
        if support:
            text += ("\n\n---\nSupport files in this skill's folder "
                     f"({skill.path.parent}): {', '.join(support)}. "
                     "Read them with a file tool if the instructions reference them.")
        return text

    # ──────────────────────────────────────────────────────────────────
    # Prompt contribution — the index, not the content
    # ──────────────────────────────────────────────────────────────────

    def agent_prompt_for(self, ctx) -> str:
        skills = self.list()
        if not skills:
            return ""
        lines = [
            "## Skills",
            "Installed skills — markdown playbooks for specific tasks. When a request "
            "matches one, call the `use_skill` tool with its name BEFORE doing the task; "
            "this index holds only summaries, not the instructions.",
            "",
        ]
        lines += [f"- {s.name}: {s.description}" for s in skills]
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # Frontmatter parsing (stdlib, mtime-cached)
    # ──────────────────────────────────────────────────────────────────

    def _parse_cached(self, path: Path, root_name: str) -> Skill | None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        cached = self._cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        skill = _parse_skill(path, root_name)
        self._cache[path] = (mtime, skill)
        return skill


def build_services(config) -> dict:
    """Build the skills service."""
    return {"skills": SkillsService(config)}


def _parse_skill(path: Path, root_name: str) -> Skill | None:
    """Parse one SKILL.md's frontmatter; folder name is the fallback name."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta = _frontmatter(text)
    name = (meta.get("name") or path.parent.name).strip().lower()
    description = " ".join((meta.get("description") or "").split())[:MAX_DESCRIPTION_CHARS]
    if not name:
        return None
    if not description:
        logger.warning("Skill %s has no description; skipping (the index would be useless).", path)
        return None
    return Skill(name=name, description=description, path=path, root_name=root_name)


def _frontmatter(text: str) -> dict[str, str]:
    """Minimal ``--- key: value ---`` header parser (no YAML dependency)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith(" "):
            out[key.strip().lower()] = value.strip()
    return {}
