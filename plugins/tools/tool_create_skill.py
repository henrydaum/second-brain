"""create_skill: persist a new canvas skill."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers import skill_store

logger = logging.getLogger("SkillTools")


class CreateSkill(BaseTool):
    name = "create_skill"
    description = "Create a Python canvas skill. Code must define run(canvas, **params), use canvas.palette slots, and call canvas.commit(image)."
    max_calls = 4
    parameters = {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"}, "kind": {"type": "string", "enum": ["creation", "transform"]}, "code": {"type": "string"}}, "required": ["name", "description", "kind", "code"]}

    def run(self, context, **kwargs) -> ToolResult:
        try:
            skill = skill_store.write_skill(
                name=str(kwargs.get("name") or ""),
                description=str(kwargs.get("description") or ""),
                kind=str(kwargs.get("kind") or "creation"),
                owner=_owner(context),
                code=str(kwargs.get("code") or ""),
                text_embedder=context.services.get("text_embedder"),
            )
            _notify(context, skill.path)
            return ToolResult(data=skill.to_dict(), llm_summary=f"Created {skill.kind} skill '{skill.slug}'. Now call execute_skill with this slug.")
        except Exception as e:
            logger.exception("create_skill failed: name=%r kind=%r owner=%r", kwargs.get("name"), kwargs.get("kind"), _owner(context))
            return ToolResult.failed(str(e))


def _owner(context) -> str:
    return str(getattr(context, "session_key", "") or "local").split(":", 1)[0]


def _notify(context, path: str) -> None:
    try:
        p = Path(path)
        context.db.upsert_file(str(p), p.name, p.suffix.lower(), "text", p.stat().st_mtime)
        context.orchestrator.on_file_discovered(str(p), p.suffix.lower(), "text")
    except Exception:
        pass
