"""compose_form: set the main subject/structure layer."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compose_common import build_result, coerce_seed
from plugins.tools.helpers.layer_generators import forms


class ComposeForm(BaseTool):
    name = "compose_form"
    description = (
        "Set the FORM layer — the dominant shapes and structure of the piece, the "
        "'subject' the eye lands on. Choose `type` to set the visual personality:\n"
        "  'primary_grid' — bold geometric energy (Mondrian feel)\n"
        "  'soft_horizontal_bands' — quiet meditative bands (Rothko feel)\n"
        "  'organic_blobs' — biomorphic flowing shapes\n"
        "  'architectural_arches' — sturdy arcades\n"
        "  'branching_tree' — dendritic growth\n"
        "  'voronoi_shards' — stained-glass fragmentation\n"
        "  'radial_burst' — explosive focus from a center\n"
        "  'silhouette_horizon' — landscape silhouette\n"
        "  'mandelbrot_zoom' — Mandelbrot fractal zoomed into a random famous region\n"
        "  'julia_set' — Julia set with a random c parameter, wildly varied per seed\n"
        "  'newton_rings' — Newton fractal stained-glass cells\n"
        "  'burning_ship' — burning-ship fractal: gothic spires and ship-like silhouettes\n"
        "  'multibrot' — z^n + c with n ∈ 3..7 chosen by seed; petal/star structures\n"
        "  'tricorn' — z̄^2 + c: mirrored mandelbrot with three lobes\n"
        "  'phoenix' — z^2 + c + p·z_prev: filamentous bird-like structures\n"
        "  'sierpinski_carpet' — 2D Menger sponge, recursive self-similar grid\n"
        "  'menger_3d' — 3D Menger sponge raymarched from a random viewpoint\n"
        "  'fractalize_canvas' — takes the CURRENT canvas and re-renders it through a fractal orbit-trap lens. The image's pixels get scattered along Mandelbrot/Julia filaments. Use this AFTER building a piece to fractal-ify it.\n"
        "All fractal forms have transparent alpha around the set so the background bleeds through — "
        "they layer beautifully over gradients and cloud noise. Re-calling REPLACES the form."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": list(forms.FORM_TYPES),
                "description": "Which form archetype to render.",
            },
            "scale": {
                "type": "string",
                "enum": ["small", "medium", "large", "filling"],
                "description": "How large the form sits within the canvas.",
            },
            "density": {
                "type": "string",
                "enum": ["sparse", "moderate", "dense"],
                "description": "How many elements / how busy the form is.",
            },
            "seed": {
                "type": "integer",
                "description": "Optional seed for reproducible variation.",
            },
        },
        "required": ["type"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        form_type = str(kwargs.get("type") or "organic_blobs")
        if form_type not in forms.FORM_TYPES:
            return ToolResult.failed(f"Unknown form type: {form_type}")
        scale = str(kwargs.get("scale") or "medium")
        density = str(kwargs.get("density") or "moderate")
        seed = coerce_seed(kwargs.get("seed"))
        state = lc.get_state(session_key)
        size = tuple(state.get("size") or lc.DEFAULT_SIZE)
        img = forms.render(form_type, size, seed, scale=scale, density=density, session_key=session_key)
        path = lc.layer_path(session_key, "form")
        img.save(path, "PNG", optimize=True)
        lc.set_image_layer(session_key, "form", self.name, {"type": form_type, "scale": scale, "density": density, "seed": seed}, path)
        return build_result(context, session_key, f"Form set to {form_type} (scale={scale}, density={density}, seed={seed}).")
