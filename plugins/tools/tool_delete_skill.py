"""delete_skill: hide an owned canvas skill from discovery (soft-delete)."""

from __future__ import annotations

import logging

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers import skill_store

logger = logging.getLogger("SkillTools")


class DeleteSkill(BaseTool):
    name = "delete_skill"
    description = "Hide a canvas skill from search results. The skill file stays on disk so that any shared canvas that uses it can still be replayed — links never break. Skills owned by another frontend cannot be hidden."
    max_calls = 2
    parameters = {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}

    def run(self, context, **kwargs) -> ToolResult:
        slug = str(kwargs.get("slug") or "")
        try:
            skill = skill_store.read_skill(slug)
            hidden = skill_store.delete_skill(slug, owner=str(getattr(context, "session_key", "") or "local"))
            return ToolResult(data={"hidden": hidden}, llm_summary=f"Hid skill '{slug}' (file kept for link replay)." if hidden else f"No skill named '{slug}' exists.")
        except Exception as e:
            logger.exception("delete_skill failed: slug=%r", slug)
            return ToolResult.failed(str(e))
