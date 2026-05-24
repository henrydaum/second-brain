"""manage_layers: agent-side control over the canvas chain (delete/move/clear).

Routes through the new CanvasRuntime: ``context.canvas.handle_action(...)``
for the mutation, then ``render_canvas(...)`` to refresh the image (unless
the action cleared the chain, in which case there's nothing to render).
"""

from __future__ import annotations

import logging

from events.event_bus import bus
from events.event_channels import CANVAS_CHANGED
from canvas.render import bus_progress, render_canvas
from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("ManageLayers")


def _snap_after(cs, render_result) -> dict:
	"""Build the canvas-dict shape the frontend (and old callers) expect."""
	if render_result is None:
		return {
			"path": None,
			"chain": list(cs.canvas.layers),
			"size": cs.canvas.size,
			"palette_id": cs.canvas.palette_id,
			"canvas_id": cs.canvas_id,
		}
	return {
		"path": str(render_result.image_path),
		"chain": list(cs.canvas.layers),
		"size": cs.canvas.size,
		"palette_id": cs.canvas.palette_id,
		"canvas_id": cs.canvas_id,
		"pool_hash": render_result.pool_hash,
		"seed": render_result.seed,
		"cache_hit": render_result.cache_hit,
	}


def _enact_and_render(context, action_type: str, payload: dict) -> ToolResult:
	"""Mutate via context.canvas, then render if the chain still has layers."""
	session_key = getattr(context, "session_key", None) or "local"
	canvas_rt = getattr(context, "canvas", None)
	skill_registry = getattr(context, "skill_registry", None)
	if canvas_rt is None:
		return ToolResult.failed("canvas runtime not available on context")
	cs = canvas_rt.for_session(session_key)
	result = canvas_rt.handle_action(cs.canvas_id, action_type, payload)
	if not result.ok:
		msg = result.error.message if result.error else (result.message or f"{action_type} failed")
		return ToolResult.failed(msg)

	if not cs.canvas.layers:
		snap = _snap_after(cs, None)
		if session_key:
			bus.emit(CANVAS_CHANGED, {"session_key": session_key, "action": action_type})
		return ToolResult(data={"canvas": snap, "chain": []}, llm_summary="")

	if skill_registry is None:
		return ToolResult.failed("skill registry not available; cannot re-render")

	try:
		render_result = render_canvas(
			cs,
			skill_loader=skill_registry.get_record,
			db=getattr(context, "db", None),
			on_event=bus_progress(getattr(context, "session_key", None), float((getattr(context, "config", {}) or {}).get("skill_timeout_s") or 30)),
			worker_pool=(getattr(context, "services", None) or {}).get("skill_worker_pool"),
		)
	except Exception as e:
		logger.exception("manage_layers render crashed: action=%s", action_type)
		return ToolResult.failed(str(e))

	snap = _snap_after(cs, render_result)
	if session_key:
		bus.emit(CANVAS_CHANGED, {"session_key": session_key, "action": action_type, "canvas": snap})
	return ToolResult(
		data={"canvas": snap, "chain": snap["chain"]},
		llm_summary="",
		attachment_paths=[snap["path"]],
	)


class ManageLayers(BaseTool):
	name = "manage_layers"
	description = (
		"Edit the canvas layer chain (max 4 layers: 1 background + up to 5 "
		"filters/objects). action=delete removes layer at chain_index (0 is the "
		"background — deleting it clears the canvas). action=move reorders from "
		"from_index to to_index; layer 0 must stay a background. action=clear "
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
		"""Dispatch on the ``action`` argument to the matching canvas action."""
		action = str(kwargs.get("action") or "").lower()
		if action == "clear":
			result = _enact_and_render(context, "clear", {})
			if result.data:
				result.llm_summary = "Cleared the canvas."
			return result
		if action == "delete":
			idx = int(kwargs.get("chain_index", -1))
			result = _enact_and_render(context, "remove_layer", {"chain_index": idx})
			if result.data:
				result.llm_summary = f"Deleted layer {idx}."
			return result
		if action == "move":
			fi = int(kwargs.get("from_index", -1))
			ti = int(kwargs.get("to_index", -1))
			result = _enact_and_render(context, "move_layer", {"from_index": fi, "to_index": ti})
			if result.data:
				result.llm_summary = f"Moved layer {fi} to position {ti}."
			return result
		return ToolResult.failed(f"Unknown action '{action}'. Use delete, move, or clear.")
