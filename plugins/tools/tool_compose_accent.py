"""compose_accent: scatter small focal marks across the canvas."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compose_common import build_result, coerce_seed
from plugins.tools.helpers.layer_generators import accents


class ComposeAccent(BaseTool):
    name = "compose_accent"
    description = (
        "Set the ACCENT layer — sparse focal marks that catch the eye without "
        "dominating: 'sparks' and 'stars' for highlights, 'drip_marks' for "
        "paint drips, 'tally_marks' for human-mark warmth, 'arrows' for "
        "direction, 'dots' for rhythm. Use sparingly — too many accents and "
        "the piece feels busy. Re-calling REPLACES the accent (never stacks)."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "style": {
                "type": "string",
                "enum": list(accents.ACCENT_STYLES),
                "description": "Which accent mark style.",
            },
            "count": {
                "type": "string",
                "enum": ["few", "several", "many"],
                "description": "How many accent marks to scatter.",
            },
            "size": {
                "type": "string",
                "enum": ["small", "medium", "large"],
                "description": "Size of each accent mark.",
            },
            "seed": {"type": "integer"},
        },
        "required": ["style"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        style = str(kwargs.get("style") or "sparks")
        if style not in accents.ACCENT_STYLES:
            return ToolResult.failed(f"Unknown accent style: {style}")
        count = str(kwargs.get("count") or "several")
        size_hint = str(kwargs.get("size") or "medium")
        seed = coerce_seed(kwargs.get("seed"))
        state = lc.get_state(session_key)
        size = tuple(state.get("size") or lc.DEFAULT_SIZE)
        img = accents.render(style, size, seed, count=count, size_hint=size_hint)
        path = lc.layer_path(session_key, "accent")
        img.save(path, "PNG", optimize=True)
        lc.set_image_layer(session_key, "accent", self.name, {"style": style, "count": count, "size": size_hint, "seed": seed}, path)
        return build_result(context, session_key, f"Accent set to {style} (count={count}, size={size_hint}, seed={seed}).")
