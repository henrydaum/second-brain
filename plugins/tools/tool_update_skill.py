"""update_skill: edit an owned canvas skill."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers import skill_store

logger = logging.getLogger("SkillTools")


class UpdateSkill(BaseTool):
    name = "update_skill"
    description = "Update a canvas skill you own. Everyone can execute skills, but only the owner can edit them."
    max_calls = 3
    parameters = {"type": "object", "properties": {"slug": {"type": "string"}, "name": {"type": "string"}, "description": {"type": "string"}, "code": {"type": "string"}}, "required": ["slug"]}

    def run(self, context, **kwargs) -> ToolResult:
        try:
            skill = skill_store.update_skill(str(kwargs.get("slug") or ""), owner=_owner(context), name=kwargs.get("name"), description=kwargs.get("description"), code=kwargs.get("code"), text_embedder=context.services.get("text_embedder"))
            _notify(context, skill.path)
            return ToolResult(data=skill.to_dict(), llm_summary=f"Updated skill '{skill.slug}'.")
        except Exception as e:
            logger.exception("update_skill failed: slug=%r owner=%r", kwargs.get("slug"), _owner(context))
            return ToolResult.failed(str(e))


def _owner(context) -> str:
    return str(getattr(context, "session_key", "") or "local").split(":", 1)[0]


def _notify(context, path: str) -> None:
    try:
        p = Path(path); context.db.upsert_file(str(p), p.name, p.suffix.lower(), "text", p.stat().st_mtime); context.orchestrator.on_file_discovered(str(p), p.suffix.lower(), "text")
    except Exception:
        pass
