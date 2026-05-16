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
        "'subject' the eye lands on. Choose `type` to set the visual personality: "
        "'primary_grid' for bold geometric energy, 'soft_horizontal_bands' for "
        "quiet meditative pieces, 'organic_blobs' for biomorphic flow, "
        "'architectural_arches' for sturdy permanence, 'branching_tree' for "
        "dendritic growth, 'voronoi_shards' for stained-glass fragmentation, "
        "'radial_burst' for explosive focus, 'silhouette_horizon' for landscape. "
        "Re-calling this tool REPLACES the form (it never stacks)."
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
        img = forms.render(form_type, size, seed, scale=scale, density=density)
        path = lc.layer_path(session_key, "form")
        img.save(path, "PNG", optimize=True)
        lc.set_image_layer(session_key, "form", self.name, {"type": form_type, "scale": scale, "density": density, "seed": seed}, path)
        return build_result(context, session_key, f"Form set to {form_type} (scale={scale}, density={density}, seed={seed}).")
