"""Discover the shipped classic image techniques and their editable controls."""
from guest.bases import BaseTool


class SearchTechniques(BaseTool):
    name = "search_techniques"
    dependencies_files = ["scripts/canvas_catalog.py"]
    description = (
        "Find classic image editing techniques by ordinary words (crop, upload photo, "
        "sharpness, drawing, colour). Omit query to list the suite. Pass script for the "
        "exact control types, defaults, ranges, suggested increments, units and an add_layer "
        "example. recipe='photo' or 'composition' returns an end-to-end worked workflow. "
        "Read the controls before using a technique; no script authoring is needed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words describing the desired edit."},
            "script": {"type": "string", "description": "Exact script name for full control details and example."},
            "recipe": {"type": "string", "enum": ["photo", "composition"], "description": "Worked workflow with tool calls, without executing them."},
        },
    }

    def run(self, sdk, query="", script=None, recipe=None):
        return sdk.scripts.run("canvas_catalog.py", query=query, script=script, recipe=recipe)
