"""create_skill: persist a new canvas skill."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.skills.helpers import skill_store

logger = logging.getLogger("SkillTools")


class CreateSkill(BaseTool):
    name = "create_skill"
    description = "Author a new canvas skill — supply the module-level body (imports + `def run(canvas, **params)`) and Second Brain wraps it in a BaseSkill subclass and writes it to the sandbox skills folder. Only use when search_skills + read_skill cannot supply a close enough starting point. Prefer colors from canvas.palette slots (or art_kit.palette_color(t)), seed RNGs from canvas.seed, and call canvas.commit(image) on every path. Allowed imports only: math, random, colorsys, numpy, PIL.*, and `from plugins.BaseSkill import BaseSkill`. Optionally declare up to 3 user-facing controls (slider/enum/bool/pan) plus a palette control when the skill actually uses palette and should show a layer-specific override. Built-in skills may use BaseSkill descriptors directly; this tool still uses the dict control schema."
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "kind": {"type": "string", "enum": ["background", "filter", "object"], "description": "background = produces a fresh image from scratch (layer 0); filter = reads the current canvas and replaces it with a same-shape opaque image; object = reads the current canvas, returns RGBA, framework alpha-composites onto the prior canvas (overlays like typography). filters/objects require a background already in the chain."},
            "code": {"type": "string", "description": "Module-level body: any needed imports plus `def run(canvas, **params):`. Will be wrapped in a BaseSkill class automatically."},
            "controls": {
                "type": "array",
                "default": [],
                "description": "Optional user-facing controls. Each item is {type, name, label, ...type-specific fields}. Types: slider {min,max,step,default}; enum {options:[{value,label}],default}; bool {default}; pan {x_param,y_param,step,x_default,y_default}; palette (no extras, shown only for skills that use canvas.palette/art_kit.palette_color). Max 3 non-palette + 1 palette. Names must match run() params except palette.",
                "items": {"type": "object"},
            },
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
                controls=list(kwargs.get("controls") or []),
            )
            _register(context, path)
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


def _register(context, path: Path) -> None:
    """Load the freshly-written skill into the SkillRegistry so it's
    discoverable immediately without waiting for the watcher to fire."""
    registry = getattr(context, "skill_registry", None)
    if registry is None:
        return
    try:
        from plugins.plugin_discovery import load_single_plugin
        load_single_plugin("skill", path, skill_registry=registry)
    except Exception:
        logger.exception("create_skill: failed to register %s with SkillRegistry", path)


def _notify(context, path: str) -> None:
    try:
        p = Path(path)
        context.db.upsert_file(str(p), p.name, p.suffix.lower(), "text", p.stat().st_mtime)
        context.orchestrator.on_file_discovered(str(p), p.suffix.lower(), "text")
    except Exception:
        pass
