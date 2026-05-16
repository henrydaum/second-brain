"""reset_canvas: wipe all layers and start fresh."""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compositor import recompose


class ResetCanvas(BaseTool):
    name = "reset_canvas"
    description = (
        "Wipe ALL layers and start with a blank canvas. Use when the user wants "
        "a fresh start, not when they just want to back out one element (use "
        "clear_layer for that)."
    )
    max_calls = 2
    parameters = {"type": "object", "properties": {}}

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None)
        lc.reset(session_key)
        # Recompose now produces a blank grey composite — push it as an attachment
        # so the frontend updates instead of showing the stale prior canvas.
        composite_path, _ = recompose(session_key)
        return ToolResult(
            data={"reset": True, "composite_path": str(composite_path)},
            llm_summary="Canvas wiped. All layers cleared. Compose a new background to begin.",
            attachment_paths=[str(composite_path)],
        )
