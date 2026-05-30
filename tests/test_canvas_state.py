"""Unit tests for canvas.state.CanvasState and its action subclasses.

Exercises the dispatch loop directly through cs.enact(...). The
CanvasRuntime layer is tested separately in test_canvas_runtime.py.
"""

from __future__ import annotations

from canvas.state import CANVAS_IDLE, HISTORY_LIMIT, UNDO_LIMIT, CanvasState
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


def test_add_layer_background_then_filter():
	"""add_layer background starts the chain; filter appends."""
	cs = CanvasState()
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background"})
	assert len(cs.canvas.layers) == 1
	cs.enact("add_layer", {"technique_slug": "swirl", "kind": "filter", "controls": {"angle": 30}})
	assert len(cs.canvas.layers) == 2
	assert cs.canvas.layers[1]["controls"] == {"angle": 30}


def test_add_layer_rejects_unknown_kind():
	"""kind must be 'background', 'filter', or 'object'."""
	cs = CanvasState()
	r = cs.enact("add_layer", {"technique_slug": "fractal", "kind": "bogus"})
	assert not r.ok
	assert cs.last_error is not None
	assert cs.canvas.layers == []


def test_add_layer_accepts_object_kind_after_background():
	"""object layers append onto an existing chain like filters."""
	cs = CanvasState()
	r1 = cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background"})
	assert r1.ok
	r2 = cs.enact("add_layer", {"technique_slug": "typography", "kind": "object",
	                            "controls": {"phrase": "hi"}})
	assert r2.ok
	assert len(cs.canvas.layers) == 2
	assert cs.canvas.layers[1]["kind"] == "object"
	assert cs.canvas.layers[1]["controls"] == {"phrase": "hi"}


def test_add_layer_requires_technique_slug():
	"""Missing technique_slug fails cleanly."""
	cs = CanvasState()
	r = cs.enact("add_layer", {"kind": "background"})
	assert not r.ok


def test_remove_layer_out_of_range_fails_cleanly():
	"""Out-of-range index produces an ActionResult.fail rather than raising."""
	cs = CanvasState()
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background"})
	r = cs.enact("remove_layer", {"chain_index": 5})
	assert not r.ok
	assert cs.last_error is not None
	assert len(cs.canvas.layers) == 1


def test_remove_background_clears_chain():
	"""Removing layer 0 clears dependent layers too, avoiding invalid chains."""
	cs = CanvasState()
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background"})
	cs.enact("add_layer", {"technique_slug": "swirl", "kind": "filter"})
	r = cs.enact("remove_layer", {"chain_index": 0})
	assert r.ok
	assert cs.canvas.layers == []


def test_move_layer_rejects_background_displacement():
	"""Moving the background off layer 0 must fail (Canvas.move_entry rule)."""
	cs = CanvasState()
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background"})
	cs.enact("add_layer", {"technique_slug": "swirl", "kind": "filter"})
	before = list(cs.canvas.layers)
	r = cs.enact("move_layer", {"from_index": 1, "to_index": 0})
	assert not r.ok
	assert cs.canvas.layers == before


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
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background", "controls": {"zoom": 1.0}})
	r = cs.enact("set_control", {"chain_index": 0, "name": "zoom", "value": 4.5})
	assert r.ok
	assert cs.canvas.layers[0]["controls"]["zoom"] == 4.5


def test_clear_resets_chain_but_preserves_palette_and_size():
	"""clear empties the chain; palette and size survive."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("set_size", {"size": 512})
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background"})
	cs.enact("clear", {})
	assert cs.canvas.layers == []
	assert cs.canvas.palette_id == "obsidian"
	assert cs.canvas.size == 512


def test_regenerate_records_intent_in_state_history():
	"""regenerate records whether the caller requested a fresh seed."""
	cs = CanvasState()
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background"})
	cs.enact("regenerate", {})
	assert cs.history[-1]["type"] == "regenerate"
	assert cs.history[-1]["needs_new_seed"] is False
	cs.enact("regenerate", {"force_new_seed": True})
	assert cs.history[-1]["needs_new_seed"] is True


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
	cs.enact("add_layer", {"technique_slug": "fractal", "kind": "background", "controls": {"z": 2}})
	cs.enact("set_size", {"size": 768})
	cs.render_seed = 456
	snap = cs.to_dict()
	restored = CanvasState.from_dict(snap)
	assert restored.canvas_id == cs.canvas_id
	assert restored.phase == cs.phase
	assert restored.canvas.palette_id == "obsidian"
	assert restored.canvas.size == 768
	assert restored.canvas.layers == cs.canvas.layers
	assert restored.render_seed == 456


def test_history_trims_to_limit():
	"""CanvasState.history doesn't grow unbounded."""
	cs = CanvasState()
	for _ in range(HISTORY_LIMIT + 50):
		cs.enact("set_palette", {"palette_id": "japandi"})
	assert len(cs.history) == HISTORY_LIMIT


# =================================================================
# Undo / Redo
# =================================================================

def test_undo_restores_prior_palette():
	"""Undo after set_palette restores the previous palette."""
	cs = CanvasState()
	original = cs.canvas.palette_id
	cs.enact("set_palette", {"palette_id": "obsidian"})
	assert cs.canvas.palette_id == "obsidian"
	r = cs.enact("undo", None)
	assert r.ok
	assert cs.canvas.palette_id == original


def test_redo_reapplies_undone_state():
	"""Redo after undo replays the most recent state."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("undo", None)
	r = cs.enact("redo", None)
	assert r.ok
	assert cs.canvas.palette_id == "obsidian"


def test_undo_then_new_action_clears_redo_stack():
	"""A new mutating action after undo discards the forward history."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("undo", None)
	assert cs.redo_stack  # redo is available
	# A different palette change (real state change, not a no-op) should
	# clear the forward history.
	cs.enact("set_palette", {"palette_id": "monochrome"})
	assert cs.redo_stack == []


def test_noop_after_undo_preserves_redo():
	"""A no-op mutation after undo leaves redo intact (no path branched)."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("undo", None)
	# Setting palette to the current (post-undo) value is a no-op.
	current = cs.canvas.palette_id
	cs.enact("set_palette", {"palette_id": current})
	# Redo path should still be available — the user changed nothing.
	assert cs.redo_stack
	r = cs.enact("redo", None)
	assert r.ok
	assert cs.canvas.palette_id == "obsidian"


def test_undo_empty_stack_fails():
	"""Undo with nothing to undo returns a failed ActionResult."""
	cs = CanvasState()
	r = cs.enact("undo", None)
	assert not r.ok
	assert cs.canvas.layers == []  # unchanged


def test_redo_empty_stack_fails():
	"""Redo with nothing to redo returns a failed ActionResult."""
	cs = CanvasState()
	r = cs.enact("redo", None)
	assert not r.ok


def test_noop_action_does_not_push_undo_entry():
	"""Setting the palette to its current value is a no-op for undo."""
	cs = CanvasState()
	current = cs.canvas.palette_id
	cs.enact("set_palette", {"palette_id": current})
	assert cs.undo_stack == []


def test_undo_restores_render_seed():
	"""Snapshot includes render_seed — undo brings it back."""
	cs = CanvasState(render_seed=111)
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.render_seed = 222  # simulate render minting a new seed
	# Pretend that change went through a mutating action by manually pushing
	# the snapshot — actually let's just check undo restores the snapshotted
	# seed from the set_palette call.
	cs.enact("undo", None)
	assert cs.render_seed == 111


def test_undo_redo_round_trip_via_dict():
	"""Stacks survive to_dict / from_dict."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("undo", None)
	# Now undo_stack is empty, redo_stack has one entry.
	restored = CanvasState.from_dict(cs.to_dict())
	assert restored.undo_stack == cs.undo_stack
	assert restored.redo_stack == cs.redo_stack
	# Redo on the restored state still works.
	r = restored.enact("redo", None)
	assert r.ok
	assert restored.canvas.palette_id == "obsidian"


def test_undo_stack_trims_to_limit():
	"""undo_stack is bounded by UNDO_LIMIT."""
	cs = CanvasState()
	palettes = ["obsidian", "japandi"]
	# Each set_palette to a different value pushes a snapshot.
	for i in range(UNDO_LIMIT + 20):
		cs.enact("set_palette", {"palette_id": palettes[i % 2]})
	assert len(cs.undo_stack) == UNDO_LIMIT


def test_undo_redo_themselves_not_undoable():
	"""Undo/redo don't push snapshots of their own execution."""
	cs = CanvasState()
	cs.enact("set_palette", {"palette_id": "obsidian"})
	cs.enact("set_palette", {"palette_id": "japandi"})
	# undo_stack now has 2 entries.
	cs.enact("undo", None)
	# undo_stack should have 1 entry (the first set_palette's pre-state),
	# not 3 (which would happen if undo pushed itself).
	assert len(cs.undo_stack) == 1
