"""compose_atmosphere: apply a final lighting/atmosphere pass."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compose_common import build_result, coerce_seed


ATMOSPHERE_STYLES = [
    "noir_vignette",
    "golden_hour",
    "bloom",
    "film_grain",
    "soft_haze",
    "chromatic_drift",
    "noir",
    "none",
]


class ComposeAtmosphere(BaseTool):
    name = "compose_atmosphere"
    description = (
        "Apply a final ATMOSPHERE pass — the lighting of the piece. "
        "'noir_vignette' pulls the eye to center, 'golden_hour' warms everything, "
        "'bloom' adds luminous glow to bright areas, 'film_grain' adds vintage "
        "noise, 'soft_haze' softens edges dreamily, 'chromatic_drift' separates "
        "RGB channels for a glitchy unsettled feel, 'noir' goes desaturated and "
        "cinematic, 'none' clears any existing atmosphere. Re-calling REPLACES."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "style": {
                "type": "string",
                "enum": ATMOSPHERE_STYLES,
                "description": "Atmosphere/lighting style.",
            },
            "strength": {
                "type": "string",
                "enum": ["subtle", "moderate", "strong"],
                "description": "How strong the atmosphere effect is.",
            },
            "seed": {"type": "integer"},
        },
        "required": ["style"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        style = str(kwargs.get("style") or "none")
        if style not in ATMOSPHERE_STYLES:
            return ToolResult.failed(f"Unknown atmosphere style: {style}")
        strength = str(kwargs.get("strength") or "moderate")
        seed = coerce_seed(kwargs.get("seed"))
        lc.set_pass(session_key, "atmosphere", self.name, {"style": style, "strength": strength, "seed": seed})
        return build_result(context, session_key, f"Atmosphere set to {style} (strength={strength}, seed={seed}).")
