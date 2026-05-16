"""clear_layer: remove a single layer slot from the canvas."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compose_common import build_result


class ClearLayer(BaseTool):
    name = "clear_layer"
    description = (
        "Remove a single layer from the canvas. Use this when the user wants to "
        "back out a specific contribution without resetting everything. Valid "
        "names: 'background', 'form', 'texture', 'accent', 'palette', 'atmosphere'."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": list(lc.ALL_SLOTS),
                "description": "Which layer slot to clear.",
            }
        },
        "required": ["name"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        name = str(kwargs.get("name") or "")
        if name not in lc.ALL_SLOTS:
            return ToolResult.failed(f"Unknown layer: {name}")
        try:
            lc.clear_slot(session_key, name)
        except ValueError as e:
            return ToolResult.failed(str(e))
        return build_result(context, session_key, f"Cleared {name} layer.")
