"""compose_palette: apply a color-theory pass over the canvas."""

from __future__ import annotations

import json
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compose_common import build_result, coerce_seed


PRESETS_PATH = Path(__file__).parent / "helpers" / "data" / "presets.json"


def _palette_schemes() -> dict:
    try:
        data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        return data.get("palette_schemes", {})
    except Exception:
        return {}


_SCHEMES = _palette_schemes()
_SCHEME_NAMES = list(_SCHEMES.keys()) or [
    "rothko_warm", "rothko_cool", "mondrian_primary", "ukiyo_indigo",
    "sumi_mono", "pollock_earth", "nocturne", "pastel_sorbet",
    "noir", "klimt_gold", "matisse_cutouts",
]


class ComposePalette(BaseTool):
    name = "compose_palette"
    description = (
        "Apply a COLOR PALETTE pass over the canvas — recolors the existing layers "
        "into a harmonious scheme. This is the color-theory dial: 'rothko_warm' for "
        "glowing meditation, 'mondrian_primary' for stark primaries, 'ukiyo_indigo' "
        "for restrained Japanese-print feel, 'sumi_mono' for ink monochrome, "
        "'nocturne' for midnight blues, 'pastel_sorbet' for soft and airy, 'noir' "
        "for desaturated cinematic, 'klimt_gold' for warm opulent, "
        "'matisse_cutouts' for flat saturated joy. `intensity` controls how strongly "
        "the palette overrides the source colors. Re-calling REPLACES the palette."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "scheme": {
                "type": "string",
                "enum": _SCHEME_NAMES,
                "description": "Color scheme to harmonize toward.",
            },
            "intensity": {
                "type": "string",
                "enum": ["subtle", "moderate", "strong"],
                "description": "How strongly the palette dominates the source colors.",
            },
            "seed": {"type": "integer"},
        },
        "required": ["scheme"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        scheme = str(kwargs.get("scheme") or "rothko_warm")
        if scheme not in _SCHEME_NAMES:
            return ToolResult.failed(f"Unknown palette scheme: {scheme}")
        intensity = str(kwargs.get("intensity") or "moderate")
        seed = coerce_seed(kwargs.get("seed"))
        lc.set_pass(session_key, "palette", self.name, {"scheme": scheme, "intensity": intensity, "seed": seed})
        return build_result(context, session_key, f"Palette set to {scheme} (intensity={intensity}, seed={seed}).")
