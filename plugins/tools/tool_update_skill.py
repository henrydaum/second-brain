"""update_skill: edit an owned canvas skill."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.skills.helpers import skill_store

logger = logging.getLogger("SkillTools")


class UpdateSkill(BaseTool):
    name = "update_skill"
    description = "Edit a canvas skill you own (matches your session_key). Use to fix a bug or iterate on an existing variant. Anyone can execute any skill, but you cannot edit one owned by another session — clone-and-adjust via read_skill + create_skill if you need to fork."
    max_calls = 3
    parameters = {
        "type": "object",
        "properties": {
            "slug": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "code": {"type": "string", "description": "Complete replacement Python source for one BaseSkill subclass with `def run(self, canvas)`. Declare controls with descriptors. Omit to keep current code."},
        },
        "required": ["slug"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        slug = str(kwargs.get("slug") or "")
        registry = getattr(context, "skill_registry", None)
        if registry is None:
            return ToolResult.failed("skill registry not available")
        inst = registry.get(slug)
        if inst is None:
            return ToolResult.failed(f"no skill named '{slug}'")
        path = Path(getattr(inst, "_source_path", "") or "")
        if not path.is_file():
            return ToolResult.failed(f"skill '{slug}' has no on-disk file")
        try:
            skill = skill_store.rewrite_skill(
                path, owner_session_key=_owner(context),
                name=kwargs.get("name"), description=kwargs.get("description"),
                code=kwargs.get("code"),
            )
            err = _reload(context, path)
            if err:
                return ToolResult.failed(err)
            live = _live_record(context, skill.slug)
            if live is not None:
                skill = live
            _notify(context, skill.path)
            return ToolResult(data=skill.to_dict(), llm_summary=f"Updated skill '{skill.slug}'.")
        except Exception as e:
            logger.exception("update_skill failed: slug=%r owner=%r", slug, _owner(context))
            return ToolResult.failed(str(e))


def _owner(context) -> str:
    return str(getattr(context, "session_key", "") or "local")


def _reload(context, path: Path) -> str | None:
    registry = getattr(context, "skill_registry", None)
    if registry is None:
        return None
    try:
        from plugins.plugin_discovery import load_single_plugin
        _, err = load_single_plugin("skill", path, skill_registry=registry)
        return err
    except Exception:
        logger.exception("update_skill: failed to reload %s", path)
        return f"failed to reload {path.name}"


def _live_record(context, slug: str):
    registry = getattr(context, "skill_registry", None)
    return registry.get_record(slug) if registry is not None else None


def _notify(context, path: str) -> None:
    try:
        p = Path(path)
        context.db.upsert_file(str(p), p.name, p.suffix.lower(), "text", p.stat().st_mtime)
        context.orchestrator.on_file_discovered(str(p), p.suffix.lower(), "text")
    except Exception:
        pass
