"""Discover the discovered classic image techniques and their editable controls."""
from guest.bases import BaseTool


class SearchTechniques(BaseTool):
    name = "search_techniques"
    dependencies_files = ["scripts/canvas_catalog.py", "scripts/canvas_technique_template.py"]
    description = (
        "Find classic image editing techniques by ordinary words (crop, upload photo, "
        "sharpness, drawing, colour). Omit query to list the suite. Pass script for the "
        "exact control types, defaults, ranges, suggested increments, units and an add_layer "
        "example. recipe='photo' or 'composition' returns an end-to-end worked workflow. "
        "guide=true returns a template and instructions for authoring a new technique_ script."
    )
    parameters = {
        "type": "object",
        "properties": {
            "guide": {"type": "boolean", "description": "Return the technique authoring template and workflow."},
            "query": {"type": "string", "description": "Words describing the desired edit."},
            "script": {"type": "string", "description": "Exact script name for full control details and example."},
            "recipe": {"type": "string", "enum": ["photo", "composition"], "description": "Worked workflow with tool calls, without executing them."},
        },
    }

    def run(self, sdk, query="", script=None, recipe=None, guide=False):
        return sdk.scripts.run("canvas_catalog.py", action="guide" if guide else "search", query=query, script=script, recipe=recipe)
