"""CanvasState — the state machine for one canvas.

Composes the existing ``Canvas`` dataclass (state_machine/canvas.py) so we
reuse its already-tested mutators (apply_palette, apply_control, etc.) and
add the state-machine layer around it: phase, event history, and a single
``enact(action_type, content)`` dispatch entry point.

No participants, no turn_priority, no phase stack. Bare bones.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from canvas.canvas import Canvas
from state_machine.errors import ActionError, ActionResult


CANVAS_IDLE = "canvas_idle"

# Mirrors ConversationState's convention — bound on the event log so a
# long-lived canvas doesn't grow its in-memory state unboundedly.
HISTORY_LIMIT = 100

# Cap on each of undo_stack / redo_stack. Each entry is a {canvas, render_seed}
# snapshot — small JSON, but unbounded growth would bloat state_json on the DB.
UNDO_LIMIT = 50

# Action types that push a snapshot onto undo_stack on successful execution.
# Undo/redo themselves are excluded so they only shuffle between the stacks.
UNDOABLE_ACTIONS = frozenset({
	"add_layer", "remove_layer", "move_layer",
	"set_control", "set_palette", "set_size",
	"clear", "regenerate",
})


def _new_canvas_id() -> str:
	"""Url-safe short id. Matches pipeline.canvas_store's format."""
	return secrets.token_urlsafe(8).rstrip("=")


class CanvasState:
	"""One canvas, addressable by ``canvas_id``."""

	def __init__(
		self,
		canvas: Canvas | None = None,
		canvas_id: str | None = None,
		phase: str = CANVAS_IDLE,
		history: list[dict] | None = None,
		render_seed: int | None = None,
		undo_stack: list[dict] | None = None,
		redo_stack: list[dict] | None = None,
	):
		"""Initialize the canvas state."""
		self.canvas_id: str = canvas_id or _new_canvas_id()
		self.canvas: Canvas = canvas or Canvas()
		self.phase: str = phase
		self.history: list[dict] = list(history or [])
		self.render_seed: int | None = int(render_seed) if render_seed is not None else None
		self.last_error: ActionError | None = None
		# Snapshots of {canvas: canvas.to_dict(), render_seed} from before each
		# undoable mutation. Most recent on the end of undo_stack.
		self.undo_stack: list[dict] = list(undo_stack or [])
		self.redo_stack: list[dict] = list(redo_stack or [])

	def _snapshot(self) -> dict:
		"""Capture the render-determining state for undo/redo."""
		return {"canvas": self.canvas.to_dict(), "render_seed": self.render_seed}

	def event(self, type_: str, **data: Any) -> dict:
		"""Append an event to the history, trimming to ``HISTORY_LIMIT``."""
		ev: dict[str, Any] = {"type": type_, "at": time.time(), **data}
		self.history.append(ev)
		if len(self.history) > HISTORY_LIMIT:
			# Drop the oldest entries to stay bounded.
			del self.history[: len(self.history) - HISTORY_LIMIT]
		return ev

	def enact(self, action_type: str, content: Any = None) -> ActionResult:
		"""Dispatch ``action_type`` to its action class and run it.

		For undoable actions, captures a pre-state snapshot and — if the
		action succeeded and actually changed the render-determining state —
		pushes it onto undo_stack and clears redo_stack. No-ops (e.g. setting
		a control to its current value) don't add a stack entry.
		"""
		# Imported here to avoid an import cycle (action -> state -> action).
		from canvas.action_map import create_canvas_action

		undoable = action_type in UNDOABLE_ACTIONS
		pre = self._snapshot() if undoable else None
		action = create_canvas_action(self, action_type, content)
		result = action.enact()
		if undoable and result.success and pre != self._snapshot():
			self.undo_stack.append(pre)
			if len(self.undo_stack) > UNDO_LIMIT:
				del self.undo_stack[: len(self.undo_stack) - UNDO_LIMIT]
			self.redo_stack.clear()
		return result

	# ── serialization ────────────────────────────────────────────────────

	def to_dict(self) -> dict[str, Any]:
		"""Serialize to a plain dict (round-trips via from_dict)."""
		return {
			"canvas_id": self.canvas_id,
			"phase": self.phase,
			"canvas": self.canvas.to_dict(),
			"history": list(self.history),
			"render_seed": self.render_seed,
			"undo_stack": list(self.undo_stack),
			"redo_stack": list(self.redo_stack),
		}

	@classmethod
	def from_dict(cls, data: dict[str, Any] | None) -> "CanvasState":
		"""Restore from a dict produced by ``to_dict``."""
		if not data:
			return cls()
		return cls(
			canvas=Canvas.from_dict(data.get("canvas")),
			canvas_id=data.get("canvas_id"),
			phase=str(data.get("phase") or CANVAS_IDLE),
			history=list(data.get("history") or []),
			render_seed=data.get("render_seed"),
			undo_stack=list(data.get("undo_stack") or []),
			redo_stack=list(data.get("redo_stack") or []),
		)
