"""Phase -> action_type -> class dispatch table for canvas actions.

Mirrors state_machine/action_map.py's structure. Bare bones uses a single
phase, ``CANVAS_IDLE`` — every action is immediate, no forms or approvals.
"""

from __future__ import annotations

from typing import Any

from canvas.action import (
	CanvasAction,
	CanvasAddLayer,
	CanvasClear,
	CanvasInvalidAction,
	CanvasMoveLayer,
	CanvasRedo,
	CanvasRegenerate,
	CanvasRemoveLayer,
	CanvasSetControl,
	CanvasSetPalette,
	CanvasSetSize,
	CanvasUndo,
)
from canvas.state import CANVAS_IDLE


CANVAS_ACTION_MAP: dict[str, dict[str, type[CanvasAction]]] = {
	CANVAS_IDLE: {
		CanvasClear.action_type: CanvasClear,
		CanvasRegenerate.action_type: CanvasRegenerate,
		CanvasAddLayer.action_type: CanvasAddLayer,
		CanvasRemoveLayer.action_type: CanvasRemoveLayer,
		CanvasMoveLayer.action_type: CanvasMoveLayer,
		CanvasSetControl.action_type: CanvasSetControl,
		CanvasSetPalette.action_type: CanvasSetPalette,
		CanvasSetSize.action_type: CanvasSetSize,
		CanvasUndo.action_type: CanvasUndo,
		CanvasRedo.action_type: CanvasRedo,
	},
}


def create_canvas_action(cs, action_type: str, content: Any = None) -> CanvasAction:
	"""Resolve ``action_type`` against the current phase, return an action.

	Unknown phase or unknown action_type returns a CanvasInvalidAction
	whose ``enact`` produces a failed ActionResult — matching the
	InvalidAction pattern from state_machine/action.py.
	"""
	phase_map = CANVAS_ACTION_MAP.get(cs.phase, {})
	cls = phase_map.get(action_type, CanvasInvalidAction)
	return cls(cs, content)


def legal_canvas_actions(phase: str = CANVAS_IDLE) -> list[str]:
	"""Return the action_types legal in ``phase``."""
	return sorted((CANVAS_ACTION_MAP.get(phase) or {}).keys())
