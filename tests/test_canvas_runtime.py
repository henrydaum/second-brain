"""End-to-end tests for canvas.runtime.CanvasRuntime.

Exercises the public API: create_canvas, get, delete, snapshot,
handle_action — including the labeled enact site that wraps cs.enact.
"""

from __future__ import annotations

from canvas.runtime import CanvasRuntime


def test_create_canvas_returns_id_and_registers():
	"""create_canvas returns a non-empty id; get returns the same state."""
	rt = CanvasRuntime()
	cid = rt.create_canvas()
	assert cid
	cs = rt.get(cid)
	assert cs is not None
	assert cs.canvas_id == cid


def test_handle_action_unknown_canvas_returns_failure_not_exception():
	"""handle_action on a non-existent canvas_id is a clean failure."""
	rt = CanvasRuntime()
	r = rt.handle_action("does-not-exist", "clear", {})
	assert not r.ok
	assert r.error is not None


def test_full_sequence_through_runtime():
	"""Drive a realistic edit session through handle_action."""
	rt = CanvasRuntime()
	cid = rt.create_canvas()

	r = rt.handle_action(cid, "add_layer", {"skill_slug": "fractal", "kind": "background", "controls": {"zoom": 1.0}})
	assert r.ok
	r = rt.handle_action(cid, "add_layer", {"skill_slug": "swirl", "kind": "filter", "controls": {"angle": 30}})
	assert r.ok
	r = rt.handle_action(cid, "set_control", {"chain_index": 1, "name": "angle", "value": 90})
	assert r.ok
	r = rt.handle_action(cid, "move_layer", {"from_index": 1, "to_index": 1})  # no-op move
	assert r.ok
	r = rt.handle_action(cid, "regenerate", {})
	assert r.ok

	cs = rt.get(cid)
	assert cs is not None
	assert len(cs.canvas.layers) == 2
	assert cs.canvas.layers[1]["controls"]["angle"] == 90

	r = rt.handle_action(cid, "clear", {})
	assert r.ok
	assert cs.canvas.layers == []


def test_two_canvases_are_isolated():
	"""Actions on one canvas don't leak into another."""
	rt = CanvasRuntime()
	a = rt.create_canvas()
	b = rt.create_canvas()
	rt.handle_action(a, "add_layer", {"skill_slug": "fractal", "kind": "background"})
	rt.handle_action(b, "set_palette", {"palette_id": "obsidian"})

	cs_a = rt.get(a)
	cs_b = rt.get(b)
	assert len(cs_a.canvas.layers) == 1
	assert len(cs_b.canvas.layers) == 0
	assert cs_b.canvas.palette_id == "obsidian"
	# Canvas A's palette is untouched (still the default).
	assert cs_a.canvas.palette_id != "obsidian"


def test_delete_removes_canvas_and_subsequent_action_fails():
	"""After delete, handle_action returns a failure for that id."""
	rt = CanvasRuntime()
	cid = rt.create_canvas()
	rt.delete(cid)
	assert rt.get(cid) is None
	r = rt.handle_action(cid, "clear", {})
	assert not r.ok


def test_snapshot_returns_state_dict():
	"""snapshot returns the same shape as CanvasState.to_dict."""
	rt = CanvasRuntime()
	cid = rt.create_canvas()
	rt.handle_action(cid, "set_palette", {"palette_id": "obsidian"})
	snap = rt.snapshot(cid)
	assert snap is not None
	assert snap["canvas_id"] == cid
	assert snap["canvas"]["palette_id"] == "obsidian"


def test_snapshot_unknown_canvas_returns_none():
	"""snapshot on an unknown id is None, not an exception."""
	rt = CanvasRuntime()
	assert rt.snapshot("nope") is None


def test_unknown_action_type_via_runtime_returns_failure():
	"""Bad action_type funnels through CanvasInvalidAction."""
	rt = CanvasRuntime()
	cid = rt.create_canvas()
	r = rt.handle_action(cid, "teleport", {})
	assert not r.ok


def test_for_session_creates_canvas_on_first_call():
	"""First call mints a canvas, second call returns the same one."""
	rt = CanvasRuntime()
	first = rt.for_session("sess-A")
	second = rt.for_session("sess-A")
	assert first is second
	# Different session gets its own canvas.
	other = rt.for_session("sess-B")
	assert other.canvas_id != first.canvas_id


def test_for_session_recovers_if_bound_canvas_was_deleted():
	"""If the bound canvas is gone, for_session mints a new one."""
	rt = CanvasRuntime()
	first = rt.for_session("sess-X")
	rt.delete(first.canvas_id)
	second = rt.for_session("sess-X")
	assert second.canvas_id != first.canvas_id


def test_bind_session_links_existing_canvas():
	"""bind_session ties a session to an existing canvas (share-link open path)."""
	rt = CanvasRuntime()
	cid = rt.create_canvas()
	rt.bind_session("sess-share", cid)
	assert rt.for_session("sess-share").canvas_id == cid


def test_bind_session_rejects_unknown_canvas():
	"""Trying to bind a session to a non-existent canvas is a clean error."""
	import pytest
	rt = CanvasRuntime()
	with pytest.raises(KeyError):
		rt.bind_session("sess-bad", "does-not-exist")


def test_unbind_session_does_not_delete_canvas():
	"""unbind_session forgets the mapping but leaves the canvas in the registry."""
	rt = CanvasRuntime()
	cs = rt.for_session("sess-Z")
	rt.unbind_session("sess-Z")
	# Canvas still exists; for_session would now mint a NEW one.
	assert rt.get(cs.canvas_id) is cs
	new_cs = rt.for_session("sess-Z")
	assert new_cs.canvas_id != cs.canvas_id


def test_register_existing_canvas_state():
	"""register() adds a pre-built CanvasState into the registry."""
	from canvas.state import CanvasState
	rt = CanvasRuntime()
	cs = CanvasState(canvas_id="seeded")
	cs.enact("set_palette", {"palette_id": "obsidian"})
	rt.register(cs)
	assert rt.get("seeded") is cs
	assert rt.snapshot("seeded")["canvas"]["palette_id"] == "obsidian"
