"""delete_skill: remove an owned canvas skill."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers import skill_store


class DeleteSkill(BaseTool):
    name = "delete_skill"
    description = "Delete a canvas skill you own. Skills owned by another frontend cannot be deleted."
    max_calls = 2
    parameters = {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}

    def run(self, context, **kwargs) -> ToolResult:
        slug = str(kwargs.get("slug") or "")
        try:
            skill = skill_store.read_skill(slug)
            deleted = skill_store.delete_skill(slug, owner=str(getattr(context, "session_key", "") or "local").split(":", 1)[0])
            if deleted and skill and context.orchestrator:
                context.orchestrator.on_file_deleted(skill.path)
            if deleted and skill and context.db:
                context.db.remove_file(skill.path)
            return ToolResult(data={"deleted": deleted}, llm_summary=f"Deleted skill '{slug}'." if deleted else f"No skill named '{slug}' exists.")
        except Exception as e:
            return ToolResult.failed(str(e))
