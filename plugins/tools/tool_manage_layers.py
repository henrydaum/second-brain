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
	technique_registry = getattr(context, "technique_registry", None)
	if canvas_rt is None:
		return ToolResult.failed("canvas runtime not available on context")
	cs = canvas_rt.for_session(session_key)
	credits = (getattr(context, "services", None) or {}).get("credits")
	db = getattr(context, "db", None)
	def rerender(state):
		if not state.canvas.layers:
			return None
		if technique_registry is None:
			raise RuntimeError("technique registry not available; cannot re-render")
		return render_canvas(
			state,
			technique_loader=technique_registry.get_record,
			db=db,
			on_event=bus_progress(getattr(context, "session_key", None), float((getattr(context, "config", {}) or {}).get("technique_timeout_s") or 30)),
			worker_pool=(getattr(context, "services", None) or {}).get("technique_worker_pool"),
			authorize_uncached=credits.render_authorizer(db, session_key, allow_prompt_overrun=True) if credits and db and session_key.startswith("web:") else None,
		)
	try:
		result, render_result = canvas_rt.render_actions(
			cs.canvas_id, [(action_type, payload)], rerender,
		)
	except Exception as e:
		logger.exception("manage_layers render crashed: action=%s", action_type)
		return ToolResult.failed(str(e))
	if not result.ok:
		msg = result.error.message if result.error else (result.message or f"{action_type} failed")
		return ToolResult.failed(msg)
	if not cs.canvas.layers:
		snap = _snap_after(cs, None)
		if session_key:
			bus.emit(CANVAS_CHANGED, {"session_key": session_key, "action": action_type})
		return ToolResult(data={"canvas": snap, "chain": []}, llm_summary="")

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
		"Edit the canvas layer chain (max 6 layers: 1 background + up to 5 "
		"filters/objects). action=delete removes layer at chain_index (0 is the "
		"background — deleting it clears the canvas). action=move reorders from "
		"from_index to to_index; layer 0 must stay a background. action=clear "
		"wipes the canvas entirely. action=set_control updates one control on "
		"one layer (chain_index, name, value) — use read_technique on the layer's "
		"slug first to see the control declarations (Slider min/max/step, Enum "
		"options, Bool, Pan, Text) which are the source of truth for valid "
		"names and values. Surviving layers are replayed end-to-end to rebuild "
		"the image."
	)
	max_calls = 4
	parameters = {
		"type": "object",
		"properties": {
			"action": {"type": "string", "enum": ["delete", "move", "clear", "set_control"]},
			"chain_index": {"type": "integer", "description": "Target layer index for delete or set_control."},
			"from_index": {"type": "integer", "description": "Source layer index for move."},
			"to_index": {"type": "integer", "description": "Destination layer index for move."},
			"name": {"type": "string", "description": "Control name for set_control — must match a control declared on the layer's technique (see read_technique)."},
			"value": {"description": "New control value for set_control. Type depends on the control (number for Slider, string for Enum/Text/Palette, bool for Bool)."},
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
		if action == "set_control":
			idx = int(kwargs.get("chain_index", -1))
			name = str(kwargs.get("name") or "")
			value = kwargs.get("value")
			result = _enact_and_render(context, "set_control", {"chain_index": idx, "name": name, "value": value})
			if result.data:
				result.llm_summary = f"Set {name}={value!r} on layer {idx}."
			return result
		return ToolResult.failed(f"Unknown action '{action}'. Use delete, move, clear, or set_control.")
