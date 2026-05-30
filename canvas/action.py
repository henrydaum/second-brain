"""Canvas actions — parallel Action base + concrete subclasses.

Mirrors state_machine/action.py's contract (is_legal / execute / enact /
ActionResult) but drops the conversation-specific bits (participants,
turn_priority). Each action mutates ``cs.canvas`` (a Canvas dataclass)
through the dataclass's existing pure mutators.

Bare-bones: no rendering, no seed work, no persistence. ``CanvasRegenerate``
just records intent for a future renderer to consume.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from state_machine.errors import (
	ActionError,
	ActionResult,
	ERROR_EXECUTION_FAILED,
	ERROR_INVALID_ACTION,
	ERROR_INVALID_INPUT,
)

logger = logging.getLogger("CanvasAction")


class CanvasAction:
	"""Base class for canvas-side actions."""

	action_type = "canvas_action"

	def __init__(self, cs, content: Any = None):
		"""Initialize the action."""
		self.cs = cs
		self.content: dict = dict(content or {})
		self.illegal_code: str = ERROR_INVALID_ACTION

	def is_legal(self) -> Tuple[bool, Optional[str]]:
		"""Phase check happens at dispatch; subclass overrides for action-specific gates."""
		return True, None

	def execute(self) -> ActionResult:
		"""Apply the action. Subclasses must implement."""
		raise NotImplementedError

	def error(self, code: str, message: str, **details: Any) -> ActionError:
		"""Build an ActionError anchored to the current phase."""
		return ActionError(code, message, details, self.cs.phase)

	def enact(self) -> ActionResult:
		"""Run legality, execute, normalize failures. Mirrors Action.enact."""
		legal, reason = self.is_legal()
		if not legal:
			err = self.error(self.illegal_code, reason or self.illegal_code)
			self.cs.last_error = err
			result = ActionResult.fail(self.action_type, err)
			result.events.append(self.cs.event("error", error=err.to_dict()))
			return result
		try:
			result = self.execute()
		except ActionError as err:
			self.cs.last_error = err
			result = ActionResult.fail(self.action_type, err)
			result.events.append(self.cs.event("error", error=err.to_dict()))
		except Exception as exc:
			logger.debug(
				"Error executing %s: %r", type(self).__name__, self.content, exc_info=True,
			)
			err = self.error(ERROR_EXECUTION_FAILED, str(exc) or type(exc).__name__)
			self.cs.last_error = err
			result = ActionResult.fail(self.action_type, err)
			result.events.append(self.cs.event("error", error=err.to_dict()))
		return result


class CanvasInvalidAction(CanvasAction):
	"""Returned when an unknown action_type is dispatched."""

	action_type = "canvas_invalid"

	def is_legal(self) -> Tuple[bool, Optional[str]]:
		"""Always reject."""
		return False, "That action is not legal for a canvas in this phase."

	def execute(self) -> ActionResult:
		"""Unreachable — is_legal already failed."""
		raise self.error(ERROR_INVALID_ACTION, "Invalid canvas action.", phase=self.cs.phase)


# =================================================================
# Mutators
# =================================================================

class CanvasClear(CanvasAction):
	"""Reset the chain and current image; preserve palette + size."""

	action_type = "clear"

	def execute(self) -> ActionResult:
		"""Apply the clear."""
		self.cs.canvas.reset()
		ev = self.cs.event("clear")
		return ActionResult(True, self.action_type, events=[ev])


class CanvasRegenerate(CanvasAction):
	"""Record a render request, optionally asking for a fresh seed."""

	action_type = "regenerate"

	def execute(self) -> ActionResult:
		"""Record the regenerate intent."""
		ev = self.cs.event("regenerate", needs_new_seed=bool(self.content.get("force_new_seed")))
		return ActionResult(True, self.action_type, events=[ev])


class CanvasAddLayer(CanvasAction):
	"""Append a background (clears the chain) or transform layer."""

	action_type = "add_layer"

	def execute(self) -> ActionResult:
		"""Push the entry; let Canvas validate the kind."""
		slug = self.content.get("technique_slug")
		kind = self.content.get("kind")
		if not slug:
			raise self.error(ERROR_INVALID_INPUT, "add_layer requires 'technique_slug'.")
		if kind not in ("background", "filter", "object"):
			raise self.error(ERROR_INVALID_INPUT, f"add_layer 'kind' must be 'background', 'filter', or 'object' (got {kind!r}).")
		entry = {
			"slug": slug,
			"kind": kind,
			"controls": dict(self.content.get("controls") or {}),
		}
		self.cs.canvas.push_chain_entry(entry)
		ev = self.cs.event("add_layer", technique_slug=slug, kind=kind)
		return ActionResult(True, self.action_type, events=[ev])


class CanvasRemoveLayer(CanvasAction):
	"""Delete a layer by chain index."""

	action_type = "remove_layer"

	def execute(self) -> ActionResult:
		"""Delete the entry at chain_index."""
		idx = self.content.get("chain_index")
		if not isinstance(idx, int):
			raise self.error(ERROR_INVALID_INPUT, "remove_layer requires integer 'chain_index'.")
		if idx == 0:
			self.cs.canvas.reset()
		else:
			self.cs.canvas.delete_entry(idx)
		ev = self.cs.event("remove_layer", chain_index=idx)
		return ActionResult(True, self.action_type, events=[ev])


class CanvasMoveLayer(CanvasAction):
	"""Reorder layers. Canvas enforces 'background must lead'."""

	action_type = "move_layer"

	def execute(self) -> ActionResult:
		"""Move from_index -> to_index."""
		src = self.content.get("from_index")
		dst = self.content.get("to_index")
		if not isinstance(src, int) or not isinstance(dst, int):
			raise self.error(ERROR_INVALID_INPUT, "move_layer requires integer 'from_index' and 'to_index'.")
		self.cs.canvas.move_entry(src, dst)
		ev = self.cs.event("move_layer", from_index=src, to_index=dst)
		return ActionResult(True, self.action_type, events=[ev])


class CanvasSetControl(CanvasAction):
	"""Update one control on one chain entry."""

	action_type = "set_control"

	def execute(self) -> ActionResult:
		"""Apply the control update."""
		idx = self.content.get("chain_index")
		name = self.content.get("name")
		if not isinstance(idx, int):
			raise self.error(ERROR_INVALID_INPUT, "set_control requires integer 'chain_index'.")
		if not name:
			raise self.error(ERROR_INVALID_INPUT, "set_control requires 'name'.")
		self.cs.canvas.apply_control(idx, name, self.content.get("value"))
		ev = self.cs.event("set_control", chain_index=idx, name=name, value=self.content.get("value"))
		return ActionResult(True, self.action_type, events=[ev])


class CanvasSetPalette(CanvasAction):
	"""Change the canvas-wide palette and propagate to chain entries."""

	action_type = "set_palette"

	def execute(self) -> ActionResult:
		"""Apply the palette change."""
		palette_id = self.content.get("palette_id")
		if not palette_id:
			raise self.error(ERROR_INVALID_INPUT, "set_palette requires 'palette_id'.")
		self.cs.canvas.apply_palette(str(palette_id))
		ev = self.cs.event("set_palette", palette_id=palette_id)
		return ActionResult(True, self.action_type, events=[ev])


class CanvasSetSize(CanvasAction):
	"""Update the canvas size (clamped to MIN/MAX by Canvas.set_size)."""

	action_type = "set_size"

	def execute(self) -> ActionResult:
		"""Apply the size change."""
		size = self.content.get("size")
		if not isinstance(size, int):
			raise self.error(ERROR_INVALID_INPUT, "set_size requires integer 'size'.")
		self.cs.canvas.set_size(size)
		ev = self.cs.event("set_size", size=self.cs.canvas.size)
		return ActionResult(True, self.action_type, events=[ev])


# =================================================================
# Undo / Redo
# =================================================================
# Snapshot-based: pop from one stack, push current state onto the other,
# restore. CanvasState.enact populates undo_stack for the other action
# classes; these two are excluded from UNDOABLE_ACTIONS so they never
# snapshot themselves.

class CanvasUndo(CanvasAction):
	"""Restore the most recent prior canvas state."""

	action_type = "undo"

	def execute(self) -> ActionResult:
		"""Pop undo_stack onto canvas, push current state to redo_stack."""
		if not self.cs.undo_stack:
			raise self.error(ERROR_INVALID_ACTION, "Nothing to undo.")
		from canvas.canvas import Canvas
		snapshot = self.cs.undo_stack.pop()
		self.cs.redo_stack.append(self.cs._snapshot())
		self.cs.canvas = Canvas.from_dict(snapshot.get("canvas"))
		self.cs.render_seed = snapshot.get("render_seed")
		ev = self.cs.event("undo")
		return ActionResult(True, self.action_type, events=[ev])


class CanvasRedo(CanvasAction):
	"""Re-apply the most recently undone canvas state."""

	action_type = "redo"

	def execute(self) -> ActionResult:
		"""Pop redo_stack onto canvas, push current state to undo_stack."""
		if not self.cs.redo_stack:
			raise self.error(ERROR_INVALID_ACTION, "Nothing to redo.")
		from canvas.canvas import Canvas
		snapshot = self.cs.redo_stack.pop()
		self.cs.undo_stack.append(self.cs._snapshot())
		self.cs.canvas = Canvas.from_dict(snapshot.get("canvas"))
		self.cs.render_seed = snapshot.get("render_seed")
		ev = self.cs.event("redo")
		return ActionResult(True, self.action_type, events=[ev])
