"""create_technique: persist a new canvas technique."""

from __future__ import annotations

import logging
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.techniques.helpers import technique_store

logger = logging.getLogger("TechniqueTools")


class CreateTechnique(BaseTool):
    name = "create_technique"
    description = "Author a new canvas technique by supplying a complete BaseTechnique class file. Import every descriptor you use from plugins.BaseTechnique, e.g. `from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette`. Declare controls directly as class attributes with Slider/Enum/Bool/Pan/Text/Palette descriptors, e.g. `slot = Enum(['background','primary'], default='primary', label='Palette Slot')`, and read them as self.<name> inside run(self, canvas). Do not define get_controls(), plain control defaults, or controls=[...]. Palette() is only a whole-layer palette override; use Enum for choosing a palette slot. Prefer colors from canvas.palette slots or art_kit.palette_color(t), seed RNGs from canvas.seed, and call canvas.commit(image) on every path."
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "kind": {"type": "string", "enum": ["background", "filter", "object"], "description": "background = produces a fresh image from scratch (layer 0); filter = reads the current canvas and replaces it with a same-shape opaque image; object = reads the current canvas, returns RGBA, framework alpha-composites onto the prior canvas (overlays like typography). filters/objects require a background already in the chain."},
            "code": {"type": "string", "description": "Complete Python source for one BaseTechnique subclass with `def run(self, canvas)`. Use descriptor controls as class attributes."},
        },
        "required": ["name", "description", "kind", "code"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        try:
            technique, path = technique_store.write_technique(
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
            live = _live_record(context, technique.slug)
            if live is not None:
                technique = live
            _notify(context, str(path))
            return ToolResult(
                data=technique.to_dict(),
                llm_summary=f"Created {technique.kind} technique '{technique.slug}'. Now call execute_technique with this slug.",
            )
        except Exception as e:
            logger.exception("create_technique failed: name=%r kind=%r owner=%r", kwargs.get("name"), kwargs.get("kind"), _owner(context))
            return ToolResult.failed(str(e))


def _owner(context) -> str:
    return str(getattr(context, "session_key", "") or "local")


def _register(context, path: Path) -> str | None:
    """Load the freshly-written technique into the TechniqueRegistry so it's
    discoverable immediately without waiting for the watcher to fire."""
    registry = getattr(context, "technique_registry", None)
    if registry is None:
        return None
    try:
        from plugins.plugin_discovery import load_single_plugin
        _, err = load_single_plugin("technique", path, technique_registry=registry)
        return err
    except Exception:
        logger.exception("create_technique: failed to register %s with TechniqueRegistry", path)
        return f"failed to register {path.name}"


def _live_record(context, slug: str):
    registry = getattr(context, "technique_registry", None)
    return registry.get_record(slug) if registry is not None else None


def _notify(context, path: str) -> None:
    try:
        p = Path(path)
        context.db.upsert_file(str(p), p.name, p.suffix.lower(), "text", p.stat().st_mtime)
        context.orchestrator.on_file_discovered(str(p), p.suffix.lower(), "text")
    except Exception:
        pass
