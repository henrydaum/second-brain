"""compose_background: set the bottom layer (always opaque)."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compose_common import build_result, coerce_seed
from plugins.tools.helpers.layer_generators import backgrounds


class ComposeBackground(BaseTool):
    name = "compose_background"
    description = (
        "Set the background layer of the canvas — the base wash, gradient, or sky "
        "the rest of the piece sits on. Re-calling this tool REPLACES the existing "
        "background (it never stacks). Pick `style` based on the mood: 'flat_wash' "
        "and 'paper_grain' for quiet meditations, 'cloud_noise' and 'radial_gradient' "
        "for atmospheric pieces, 'sky_band' for horizons."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "style": {
                "type": "string",
                "enum": list(backgrounds.BACKGROUND_STYLES),
                "description": "Which background style to render.",
            },
            "density": {
                "type": "string",
                "enum": ["minimal", "moderate", "rich", "dense"],
                "description": "How much visual activity the background carries (mostly affects noise/texture).",
            },
            "seed": {
                "type": "integer",
                "description": "Optional seed for reproducible variation. Omit for a fresh roll.",
            },
        },
        "required": ["style"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        style = str(kwargs.get("style") or "flat_wash")
        if style not in backgrounds.BACKGROUND_STYLES:
            return ToolResult.failed(f"Unknown background style: {style}")
        density = str(kwargs.get("density") or "moderate")
        seed = coerce_seed(kwargs.get("seed"))
        state = lc.get_state(session_key)
        size = tuple(state.get("size") or lc.DEFAULT_SIZE)
        img = backgrounds.render(style, size, seed, density=density)
        path = lc.layer_path(session_key, "background")
        img.save(path, "PNG", optimize=True)
        lc.set_image_layer(session_key, "background", self.name, {"style": style, "density": density, "seed": seed}, path)
        return build_result(context, session_key, f"Background set to {style} (density={density}, seed={seed}).")
