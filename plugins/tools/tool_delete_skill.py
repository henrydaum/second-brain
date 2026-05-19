"""delete_skill: hide an owned canvas skill from discovery (soft-delete)."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.skills.helpers import skill_store

logger = logging.getLogger("SkillTools")


class DeleteSkill(BaseTool):
    name = "delete_skill"
    description = "Hide one of your skills from search and the UI. This is a soft-delete: the file stays on disk so previously shared canvases still replay correctly — links never break. You can only hide skills you own."
    max_calls = 2
    parameters = {"type": "object", "properties": {"slug": {"type": "string"}}, "required": ["slug"]}

    def run(self, context, **kwargs) -> ToolResult:
        slug = str(kwargs.get("slug") or "")
        registry = getattr(context, "skill_registry", None)
        if registry is None:
            return ToolResult.failed("skill registry not available")
        inst = registry.get(slug)
        if inst is None:
            return ToolResult(data={"hidden": False}, llm_summary=f"No skill named '{slug}' exists.")
        path = Path(getattr(inst, "_source_path", "") or "")
        if not path.is_file():
            return ToolResult.failed(f"skill '{slug}' has no on-disk file")
        try:
            hidden = skill_store.soft_delete_skill(path, owner_session_key=_owner(context))
            if hidden:
                _reload(context, path)
            summary = f"Hid skill '{slug}' (file kept for link replay)." if hidden else f"No skill named '{slug}' exists."
            return ToolResult(data={"hidden": hidden}, llm_summary=summary)
        except Exception as e:
            logger.exception("delete_skill failed: slug=%r", slug)
            return ToolResult.failed(str(e))


def _owner(context) -> str:
    return str(getattr(context, "session_key", "") or "local")


def _reload(context, path: Path) -> None:
    registry = getattr(context, "skill_registry", None)
    if registry is None:
        return
    try:
        from plugins.plugin_discovery import load_single_plugin
        load_single_plugin("skill", path, skill_registry=registry)
    except Exception:
        logger.exception("delete_skill: failed to reload %s", path)
