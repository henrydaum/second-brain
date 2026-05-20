"""manage_layers: agent-side control over the canvas chain (delete/move/clear).

Thin adapter onto the DeleteLayer / MoveLayer / ResetCanvas state-machine
actions; lets the agent reuse the same CANVAS_ACTION_* event lifecycle
that user clicks already emit.
"""

from __future__ import annotations

import logging

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers import layered_canvas as lc

logger = logging.getLogger("ManageLayers")


def _enact(context, session_key: str, action_type: str, payload: dict) -> ToolResult:
    runtime = getattr(context, "runtime", None)
    if runtime is None:
        return ToolResult.failed("runtime not bound; cannot enact canvas action")
    session = runtime.get_session(session_key) if hasattr(runtime, "get_session") else None
    cs = getattr(session, "cs", None) if session else None
    if cs is None:
        return ToolResult.failed("no live conversation state for this session")
    try:
        result = cs.enact(action_type, payload, actor_id="agent")
    except Exception as e:
        logger.exception("manage_layers enact crashed: action=%s", action_type)
        return ToolResult.failed(str(e))
    if not result.ok:
        return ToolResult.failed(result.error.message if result.error else (result.message or "action failed"))
    snap = (result.data or {}).get("canvas") or lc.canvas(session_key) or {}
    attach = [snap["path"]] if snap and snap.get("path") else []
    return ToolResult(data={"canvas": snap, "chain": snap.get("chain") or []},
                      llm_summary="", attachment_paths=attach)


class ManageLayers(BaseTool):
    name = "manage_layers"
    description = (
        "Edit the canvas layer chain (max 4 layers: 1 creation + up to 3 "
        "transforms). action=delete removes layer at chain_index (0 is the "
        "creation — deleting it clears the canvas). action=move reorders from "
        "from_index to to_index; layer 0 must stay a creation. action=clear "
        "wipes the canvas entirely. Surviving layers are replayed end-to-end "
        "to rebuild the image."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["delete", "move", "clear"]},
            "chain_index": {"type": "integer", "description": "Target layer index for delete."},
            "from_index": {"type": "integer", "description": "Source layer index for move."},
            "to_index": {"type": "integer", "description": "Destination layer index for move."},
        },
        "required": ["action"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None) or "local"
        action = str(kwargs.get("action") or "").lower()
        if action == "clear":
            result = _enact(context, session_key, "reset_canvas", {})
            if result.data:
                result.llm_summary = "Cleared the canvas."
            return result
        if action == "delete":
            idx = int(kwargs.get("chain_index", -1))
            result = _enact(context, session_key, "delete_canvas_layer", {"chain_index": idx})
            if result.data:
                result.llm_summary = f"Deleted layer {idx}."
            return result
        if action == "move":
            fi = int(kwargs.get("from_index", -1))
            ti = int(kwargs.get("to_index", -1))
            result = _enact(context, session_key, "move_canvas_layer", {"from_index": fi, "to_index": ti})
            if result.data:
                result.llm_summary = f"Moved layer {fi} to position {ti}."
            return result
        return ToolResult.failed(f"Unknown action '{action}'. Use delete, move, or clear.")
