"""create_skill: persist a new canvas skill."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.skills.helpers import skill_store

logger = logging.getLogger("SkillTools")


class CreateSkill(BaseTool):
    name = "create_skill"
    description = "Author a new canvas skill by supplying a complete BaseSkill class file. Declare controls directly as class attributes with Slider/Enum/Bool/Pan/Text/Palette descriptors, e.g. `slot = Enum(['background','primary'], default='primary', label='Palette Slot')`, and read them as self.<name> inside run(self, canvas). Do not define get_controls(), plain control defaults, or controls=[...]. Palette() is only a whole-layer palette override; use Enum for choosing a palette slot. Prefer colors from canvas.palette slots or art_kit.palette_color(t), seed RNGs from canvas.seed, and call canvas.commit(image) on every path."
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "kind": {"type": "string", "enum": ["background", "filter", "object"], "description": "background = produces a fresh image from scratch (layer 0); filter = reads the current canvas and replaces it with a same-shape opaque image; object = reads the current canvas, returns RGBA, framework alpha-composites onto the prior canvas (overlays like typography). filters/objects require a background already in the chain."},
            "code": {"type": "string", "description": "Complete Python source for one BaseSkill subclass with `def run(self, canvas)`. Use descriptor controls as class attributes."},
        },
        "required": ["name", "description", "kind", "code"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        try:
            skill, path = skill_store.write_skill(
                name=str(kwargs.get("name") or ""),
                description=str(kwargs.get("description") or ""),
                kind=str(kwargs.get("kind") or "background"),
                owner=_owner(context),
                code=str(kwargs.get("code") or ""),
            )
            err = _register(context, path)
            if err:
                try:
                    path.unlink()
                except OSError:
                    pass
                return ToolResult.failed(err)
            live = _live_record(context, skill.slug)
            if live is not None:
                skill = live
            _notify(context, str(path))
            return ToolResult(
                data=skill.to_dict(),
                llm_summary=f"Created {skill.kind} skill '{skill.slug}'. Now call execute_skill with this slug.",
            )
        except Exception as e:
            logger.exception("create_skill failed: name=%r kind=%r owner=%r", kwargs.get("name"), kwargs.get("kind"), _owner(context))
            return ToolResult.failed(str(e))


def _owner(context) -> str:
    return str(getattr(context, "session_key", "") or "local")


def _register(context, path: Path) -> str | None:
    """Load the freshly-written skill into the SkillRegistry so it's
    discoverable immediately without waiting for the watcher to fire."""
    registry = getattr(context, "skill_registry", None)
    if registry is None:
        return None
    try:
        from plugins.plugin_discovery import load_single_plugin
        _, err = load_single_plugin("skill", path, skill_registry=registry)
        return err
    except Exception:
        logger.exception("create_skill: failed to register %s with SkillRegistry", path)
        return f"failed to register {path.name}"


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
