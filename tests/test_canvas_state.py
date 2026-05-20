"""Unit tests for canvas.state.CanvasState and its action subclasses.

Exercises the dispatch loop directly through cs.enact(...). The
CanvasRuntime layer is tested separately in test_canvas_runtime.py.
"""

from __future__ import annotations

from canvas.state import CANVAS_IDLE, HISTORY_LIMIT, CanvasState
from canvas.canvas import DEFAULT_PALETTE_ID, DEFAULT_SIZE, MAX_SIZE, MIN_SIZE


# =================================================================
# Construction + defaults
# =================================================================

def test_default_construction():
	"""Fresh CanvasState mirrors Canvas defaults and has an id + IDLE phase."""
	cs = CanvasState()
	assert cs.canvas_id
	assert cs.phase == CANVAS_IDLE
	assert cs.history == []
	assert cs.last_error is None
	assert cs.canvas.size == DEFAULT_SIZE
	assert cs.canvas.palette_id == DEFAULT_PALETTE_ID
	assert cs.canvas.layers == []


def test_explicit_canvas_id_preserved():
	"""An explicit canvas_id round-trips."""
	cs = CanvasState(canvas_id="abc123")
	assert cs.canvas_id == "abc123"


# =================================================================
# Per-action behavior
# =================================================================

def test_set_palette_changes_palette_and_emits_event():
	"""set_palette updates the canvas palette and adds a history event."""
	cs = CanvasState()
	r = cs.enact("set_palette", {"palette_id": "obsidian"})
	assert r.ok
	assert cs.canvas.palette_id == "obsidian"
	assert cs.history[-1]["type"] == "set_palette"


def test_add_layer_creation_then_transform():
	"""add_layer creation starts the chain; transform appends."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": "fractal", "kind": "creation"})
	assert len(cs.canvas.layers) == 1
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform", "controls": {"angle": 30}})
	assert len(cs.canvas.layers) == 2
	assert cs.canvas.layers[1]["controls"] == {"angle": 30}


def test_add_layer_rejects_unknown_kind():
	"""kind must be 'creation' or 'transform'."""
	cs = CanvasState()
	r = cs.enact("add_layer", {"skill_slug": "fractal", "kind": "bogus"})
	assert not r.ok
	assert cs.last_error is not None
	assert cs.canvas.layers == []


def test_add_layer_requires_skill_slug():
	"""Missing skill_slug fails cleanly."""
	cs = CanvasState()
	r = cs.enact("add_layer", {"kind": "creation"})
	assert not r.ok


def test_remove_layer_out_of_range_fails_cleanly():
	"""Out-of-range index produces an ActionResult.fail rather than raising."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": "fractal", "kind": "creation"})
	r = cs.enact("remove_layer", {"chain_index": 5})
	assert not r.ok
	assert cs.last_error is not None
	assert len(cs.canvas.layers) == 1


def test_move_layer_rejects_creation_displacement():
	"""Moving the creation off layer 0 must fail (Canvas.move_entry rule)."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": "fractal", "kind": "creation"})
	cs.enact("add_layer", {"skill_slug": "swirl", "kind": "transform"})
	r = cs.enact("move_layer", {"from_index": 1, "to_index": 0})
	assert not r.ok


def test_set_size_clamps_to_min_max():
	"""set_size is clamped to MIN_SIZE / MAX_SIZE."""
	cs = CanvasState()
	cs.enact("set_size", {"size": MAX_SIZE + 1000})
	assert cs.canvas.size == MAX_SIZE
	cs.enact("set_size", {"size": MIN_SIZE - 1})
	assert cs.canvas.size == MIN_SIZE


def test_set_control_updates_chain_entry():
	"""set_control mutates the named control on the target layer."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": "fractal", "kind": "creation", "controls": {"zoom": 1.0}})
	r = cs.enact("set_control", {"chain_index": 0, "name": "zoom", "value": 4.5})
	assert r.ok
	assert cs.canvas.layers[0]["controls"]["zoom"] == 4.5


def test_clear_resets_chain_but_preserves_palette_and_size():
	"""clear empties the chain; palette and size survive."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("set_size", {"size": 512})
	cs.enact("add_layer", {"skill_slug": "fractal", "kind": "creation"})
	cs.enact("clear", {})
	assert cs.canvas.layers == []
	assert cs.canvas.palette_id == "obsidian"
	assert cs.canvas.size == 512


def test_regenerate_records_intent_in_state_history():
	"""regenerate emits an event with needs_new_seed=True on CanvasState.history."""
	cs = CanvasState()
	cs.enact("add_layer", {"skill_slug": "fractal", "kind": "creation"})
	cs.enact("regenerate", {})
	assert any(e.get("type") == "regenerate" and e.get("needs_new_seed") for e in cs.history)


def test_unknown_action_type_returns_failure_not_exception():
	"""Unknown action_type routes to CanvasInvalidAction and returns ok=False."""
	cs = CanvasState()
	r = cs.enact("teleport", {})
	assert not r.ok
	assert cs.last_error is not None


# =================================================================
# Serialization + history bound
# =================================================================

def test_to_dict_from_dict_round_trip():
	"""Snapshot → restore yields equivalent state."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("add_layer", {"skill_slug": "fractal", "kind": "creation", "controls": {"z": 2}})
	cs.enact("set_size", {"size": 768})
	snap = cs.to_dict()
	restored = CanvasState.from_dict(snap)
	assert restored.canvas_id == cs.canvas_id
	assert restored.phase == cs.phase
	assert restored.canvas.palette_id == "obsidian"
	assert restored.canvas.size == 768
	assert restored.canvas.layers == cs.canvas.layers


def test_history_trims_to_limit():
	"""CanvasState.history doesn't grow unbounded."""
	cs = CanvasState()
	for _ in range(HISTORY_LIMIT + 50):
		cs.enact("set_palette", {"palette_id": "japandi"})
	assert len(cs.history) == HISTORY_LIMIT
