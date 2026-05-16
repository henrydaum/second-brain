"""Recipe-aware refinement for generated dreams."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.fractal_gallery import current
from plugins.tools.helpers.fractal_gallery import read_json
from plugins.tools.tool_generate_dream import refine_recipe, render_dream


class RefineDream(BaseTool):
    name = "refine_dream"
    description = "Refine the current dream by changing its saved recipe and re-rendering from source. Use for follow-ups like colder, less busy, more organic, more alien, or more symmetrical."
    parameters = {"type": "object", "properties": {"change_request": {"type": "string", "description": "Short natural-language refinement request."}}, "required": ["change_request"]}
    requires_services = []; max_calls = 1

    def run(self, context, **kw):
        item = current(getattr(context, "session_key", None))
        if not item: return ToolResult.failed("No current dream canvas to refine.")
        meta = read_json(str(item["path"]).rsplit(".", 1)[0] + ".json")
        recipe = meta.get("recipe")
        if not recipe: return ToolResult.failed("The current canvas has no dream recipe. Generate a dream first.")
        return render_dream(context, refine_recipe(recipe, str(kw.get("change_request") or "")))
