"""compose_texture: add surface detail over the form."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compose_common import build_result, coerce_seed
from plugins.tools.helpers.layer_generators import textures


class ComposeTexture(BaseTool):
    name = "compose_texture"
    description = (
        "Set the TEXTURE layer — surface detail laid over the form. Use to give a "
        "piece tactile quality: 'cross_hatch' and 'scribble' for drawn feel, "
        "'brushwork' for painterly, 'dot_grain' and 'static_noise' for grain, "
        "'fiber_weave' for fabric. Keep strength subtle unless you want texture to "
        "dominate. Re-calling REPLACES the texture (never stacks)."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "style": {
                "type": "string",
                "enum": list(textures.TEXTURE_STYLES),
                "description": "Which texture pattern to apply.",
            },
            "scale": {
                "type": "string",
                "enum": ["fine", "medium", "coarse"],
                "description": "Size of the texture units.",
            },
            "strength": {
                "type": "string",
                "enum": ["subtle", "moderate", "strong"],
                "description": "How visible the texture is over the form.",
            },
            "seed": {"type": "integer"},
        },
        "required": ["style"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        style = str(kwargs.get("style") or "static_noise")
        if style not in textures.TEXTURE_STYLES:
            return ToolResult.failed(f"Unknown texture style: {style}")
        scale = str(kwargs.get("scale") or "medium")
        strength = str(kwargs.get("strength") or "moderate")
        seed = coerce_seed(kwargs.get("seed"))
        state = lc.get_state(session_key)
        size = tuple(state.get("size") or lc.DEFAULT_SIZE)
        img = textures.render(style, size, seed, scale=scale, strength=strength)
        path = lc.layer_path(session_key, "texture")
        img.save(path, "PNG", optimize=True)
        lc.set_image_layer(session_key, "texture", self.name, {"style": style, "scale": scale, "strength": strength, "seed": seed}, path)
        return build_result(context, session_key, f"Texture set to {style} (scale={scale}, strength={strength}, seed={seed}).")
