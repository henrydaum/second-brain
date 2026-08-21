"""The cue ladder: what invalidates what, and where the text lands.

``prompt_cues`` is a table, and a table's failures are silent — a prompt that
refreshes too rarely reads exactly like a plugin with nothing new to say. So
the properties are driven rather than described: every rung is fired in a loop
and the set it moves is compared against the set the ladder claims.
"""

import itertools
from types import SimpleNamespace

import pytest

import prompt_cues as cues

CACHEABLE = [c for c in cues.LADDER if c != cues.CALL]


def _stamps(ctx=None):
    return {cue: cues.stamp(cue, ctx) for cue in CACHEABLE}


def test_the_ladder_is_a_threshold_not_a_set_of_triggers():
    """Firing a rung invalidates it and everything finer, and nothing coarser.

    This is the whole basis for ordering contributions by cue: if a rarer event
    did not invalidate the finer rungs, the ladder would be an unordered set of
    independent triggers and "rarest first" would mean nothing.
    """
    for fired in cues.COUNTED:
        before = _stamps()
        cues.fire(fired)
        after = _stamps()
        moved = {cue for cue in CACHEABLE if before[cue] != after[cue]}
        expected = {cue for cue in CACHEABLE
                    if cues.rank(cue) >= cues.RANK[fired]}
        assert moved == expected, f"fire({fired}) moved {moved}, want {expected}"


def test_a_turn_does_not_invalidate_a_config_cued_plugin():
    """The headline claim, stated alone because it is the one to keep.

    A prompt that only follows configuration should cost nothing per turn. The
    inverse — a config save refreshing a turn-cued prompt — is the monotonicity
    above, and is the direction that has to hold for safety.
    """
    before = _stamps()
    cues.fire(cues.TURN)
    after = _stamps()
    assert before[cues.CONFIG] == after[cues.CONFIG]
    assert before[cues.LOAD] == after[cues.LOAD]
    assert before[cues.SESSION] == after[cues.SESSION]
    assert before[cues.TURN] != after[cues.TURN]
    assert before[cues.WRITE] != after[cues.WRITE]


def test_a_write_invalidates_only_the_default():
    """The common case, and the reason ``write`` is the widest cached rung."""
    before = _stamps()
    cues.fire(cues.WRITE)
    after = _stamps()
    assert before[cues.WRITE] != after[cues.WRITE]
    assert all(before[cue] == after[cue]
               for cue in CACHEABLE if cue != cues.WRITE)


def test_session_facts_key_a_session_cued_prompt_with_nothing_fired():
    """A fact, not an event: changing one moves the stamp on its own.

    Each fact is varied alone, because a tuple keyed on only one of them would
    pass a test that changed them together.
    """
    base = SimpleNamespace(**{name: "same" for name in cues.SESSION_FACTS})
    for fact in cues.SESSION_FACTS:
        other = SimpleNamespace(**{**vars(base), fact: "moved"})
        assert cues.stamp(cues.SESSION, base) != cues.stamp(cues.SESSION, other), (
            f"{fact} does not key the session stamp")


def test_a_session_cued_prompt_is_not_keyed_on_transient_session_state():
    """``busy`` is always true while a prompt is built; keying on it is churn."""
    quiet = SimpleNamespace(session_key="chat", busy=False, phase="idle",
                            attended=False)
    busy = SimpleNamespace(session_key="chat", busy=True, phase="calling_tool",
                           attended=True)
    assert cues.stamp(cues.SESSION, quiet) == cues.stamp(cues.SESSION, busy)


def test_the_session_tuple_names_every_fact_a_prompt_can_actually_see():
    """What a guest reads and what the kernel keys on must be one set.

    ``sdk.session.get()`` is the only way a sandboxed ``agent_prompt`` learns
    anything about its session, so a key that handler answers and this table
    does not know about is a fact nothing invalidates — stale with no symptom.
    Driven off the handler's own source, so adding a field there fails here.
    """
    import ast
    import inspect

    from sandbox.handlers import kernel

    tree = ast.parse(inspect.getsource(kernel._session_get))
    answered = {
        key.value
        for node in ast.walk(tree) if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    answered |= {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name) and node.value.id == "data"
        and isinstance(node.slice, ast.Constant)
    }
    known = set(cues.SESSION_GET_KEYS) | set(cues.TRANSIENT)
    assert answered, "could not read the answer keys"
    assert answered <= known, (
        f"session.get answers {sorted(answered - known)}, which prompt_cues "
        "neither keys on nor names as transient")


def test_a_call_cued_prompt_is_never_cached():
    """Two stamps taken back to back, with nothing fired, must not match."""
    assert cues.stamp(cues.CALL) != cues.stamp(cues.CALL)


@pytest.mark.parametrize("cue", cues.LADDER)
def test_the_placement_threshold_is_config(cue):
    """Parametrized over the whole ladder so a new rung forces a decision."""
    assert cues.stable(cue) is (cues.rank(cue) <= cues.RANK[cues.STABLE_THROUGH])


@pytest.mark.parametrize("cue", cues.LADDER)
def test_only_a_session_or_finer_cue_is_lent_the_session(cue):
    """The tier, enforced: below ``session`` there is no session key to lend."""
    ctx = SimpleNamespace(session_key="chat")
    lent = cues.session_for(cue, ctx)
    assert bool(lent) is (cues.rank(cue) >= cues.RANK[cues.SESSION])


def test_never_is_not_declarable():
    """Derived from the shape, and a method declaring it is the bug.

    A method that never recomputes is exactly the permanent cache the write
    counter was written to kill, so the vocabulary simply cannot say it.
    """
    assert cues.NEVER not in cues.DECLARABLE
    assert set(cues.DECLARABLE) == set(cues.LADDER) - {cues.NEVER}


def test_a_string_shape_is_never_whatever_it_declares():
    """The shape decides first, and a declaration cannot override it."""
    plugin = SimpleNamespace(agent_prompt="fixed text",
                             agent_prompt_refresh="turn")
    assert cues.of(plugin) == cues.NEVER


def test_an_undeclared_method_defaults_to_write():
    """Silence keeps the widest cached rung — the safe direction, stated."""
    plugin = SimpleNamespace(agent_prompt=lambda ctx: "live")
    assert cues.of(plugin) == cues.WRITE
    assert cues.DEFAULT == cues.WRITE


def test_an_unknown_cue_falls_back_rather_than_raising():
    """The validator reports it at load; assembly stays conservative."""
    plugin = SimpleNamespace(agent_prompt=lambda ctx: "live",
                             agent_prompt_refresh="sesion")
    assert cues.of(plugin) == cues.DEFAULT
    assert cues.rank("sesion") == cues.RANK[cues.DEFAULT]


def test_every_declared_cue_is_read_back_unchanged():
    for cue in cues.DECLARABLE:
        plugin = SimpleNamespace(agent_prompt=lambda ctx: "live",
                                 agent_prompt_refresh=cue)
        assert cues.of(plugin) == cue


def test_ranks_are_distinct_and_ordered():
    assert [cues.RANK[c] for c in cues.LADDER] == list(range(len(cues.LADDER)))


def test_rendering_never_ticks_the_write_counter():
    """Moved wholesale from the epoch, and still why the set is named.

    A streaming reply sends one of these per token; counting any of them would
    undo the caching entirely, with no symptom beyond being slow.
    """
    ok = SimpleNamespace(ok=True)
    for kind in cues.RENDERING:
        assert kind in cues.UNCOUNTED, f"{kind} must not tick"
        assert not cues.counts(SimpleNamespace(type=kind), ok)


def test_a_failed_effect_changed_nothing():
    from sandbox.guest.requests import FS_WRITE

    assert cues.counts(SimpleNamespace(type=FS_WRITE), SimpleNamespace(ok=True))
    assert not cues.counts(SimpleNamespace(type=FS_WRITE),
                           SimpleNamespace(ok=False))


def test_stamps_for_different_cues_never_collide():
    """The cue leads the tuple, so a key says what it is keyed on."""
    ctx = SimpleNamespace(session_key="chat")
    for a, b in itertools.combinations(CACHEABLE, 2):
        assert cues.stamp(a, ctx) != cues.stamp(b, ctx)


# ────────────────────────────────────────────────────────────────────
# The fire sites: a rung nobody bumps is a rung that does nothing
# ────────────────────────────────────────────────────────────────────


def test_both_config_files_invalidate_a_config_cued_prompt():
    """The funnel is the announcement, not either writer.

    ``save`` writes the kernel's settings and ``save_plugin_config`` a plugin's
    own; a plugin rendering the value it was configured with cares about the
    second, and firing in the first would have missed it entirely.
    """
    from config.config_manager import _emit_config_changed

    for scope in ("core", "plugin"):
        before = cues.stamp(cues.CONFIG)
        _emit_config_changed(scope, ["some_setting"])
        assert cues.stamp(cues.CONFIG) != before, f"{scope} did not fire"


def test_a_save_that_changed_nothing_does_not_invalidate():
    """``save`` merges DEFAULTS and announces unconditionally.

    Materializing a default is a change to a file and to nothing else, so a
    boot would otherwise cost every config-cued prompt a recompute for a change
    nobody made — the same rule ``_record_config_save`` already follows.
    """
    from config.config_manager import _emit_config_changed

    before = cues.stamp(cues.CONFIG)
    _emit_config_changed("core", [])
    _emit_config_changed("core", None)
    assert cues.stamp(cues.CONFIG) == before


def test_starting_a_turn_fires_the_turn_rung():
    from runtime.hooks import HookRegistry

    before = cues.stamp(cues.TURN)
    HookRegistry().start_turn(SimpleNamespace(history=[], key="chat"))
    assert cues.stamp(cues.TURN) != before


def test_a_turn_scoped_mode_is_a_session_fact_not_a_turn_one():
    """Set for one turn and cleared at its end, with no counter involved.

    ``HookRegistry.finish_turn`` clears ``turn_security_mode`` by writing the
    field directly rather than through the runtime, which is exactly the kind
    of site a fire-based design forgets. The mode reaches the stamp because
    ``PromptContext.security_mode`` is read fresh from the runtime's own
    precedence reader, so there is nothing to forget.
    """
    from runtime.hooks import HookRegistry

    session = SimpleNamespace(history=[], key="chat", turn_security_mode="yolo",
                              staged_attachments=None,
                              pending_agent_actions=None)
    during = cues.stamp(cues.SESSION,
                        SimpleNamespace(session_key="chat",
                                        security_mode=session.turn_security_mode))
    HookRegistry().finish_turn(session)
    assert session.turn_security_mode is None
    after = cues.stamp(cues.SESSION,
                       SimpleNamespace(session_key="chat", security_mode="ask"))
    assert during != after


def test_changing_the_plugin_population_fires_the_load_rung():
    """A package install writes from kernel code and never reaches _settle.

    So without this rung a prompt describing what is installed was invalidated
    only by coincidence — whenever something unrelated happened to write.
    """
    from plugins.plugin_discovery import unload_plugin

    before = cues.stamp(cues.LOAD)
    unload_plugin("tool", "nothing-by-this-name")
    assert cues.stamp(cues.LOAD) != before
