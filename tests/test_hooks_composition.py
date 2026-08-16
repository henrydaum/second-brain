"""Two hooks at one doorway, and hooks that meet each other across doorways.

Every existing hook test stands *one* hook at *one* doorway. That is the shape
the contract is written in, but it is not the shape a real install has: a
turn meets whatever the user happened to install, in load order, and the
interesting failures are between them — an escort wrapping another escort, a
shaper hiding the tool its own doorman then demands, a doorman that stops
being consulted because an earlier one said yes.

These run **native and in-process**. The sandboxed half of the claim is
``tests/test_hooks_live_turn.py``, which proves one boxed probe writes the
same journal a native one does; past that point, whether a hook came from a
box is not what these tests are about, and paying a subprocess per case would
buy nothing but wall clock.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Import the state_machine package before runtime modules to settle the
# package-init circular import.
import state_machine  # noqa: F401

from runtime.hooks import (Allow, PermissionVerdict, Redrive, RequireTool,
                           SendBack, TurnEnding)
from tests.support import (FakeLLM, ToolChoiceLLM, echo_tool, loop_rig,
                           moments_in, native_probe, response, tool_call)

_OVERFLOW = "prompt tokens exceed model token limit"


class _OverflowLLM(FakeLLM):
    """Raises a context-limit error for the first ``overflows`` calls."""

    def __init__(self, responses=None, overflows=1):
        """Queue answers, and how many times to overflow first."""
        super().__init__(responses)
        self.overflows = overflows

    def chat(self, request, on_delta=None, on_call=None):
        """Overflow, then behave."""
        if self.overflows > 0:
            self.overflows -= 1
            self.calls.append(list(request.messages))
            raise RuntimeError(_OVERFLOW)
        return super().chat(request, on_delta)


class _Compactor:
    """A compactor that counts and answers with a fixed summary."""

    loaded = True

    def __init__(self):
        """Start at zero."""
        self.calls = 0

    def compact(self, **kwargs):
        """Summarize."""
        self.calls += 1
        return "Earlier summary."


# ──────────────────────────────────────────────────────────────────────
# llm_call — two escorts on one call
# ──────────────────────────────────────────────────────────────────────

def test_the_first_registered_escort_is_the_outermost():
    """Escorts nest like an onion, outermost first.

    The order is not cosmetic: the outer escort sees the inner one's response
    and can act on it, which is the whole reason the onion is built with
    ``reversed()`` in ``HookRegistry.wrap_llm_call``.
    """
    rig = loop_rig(llm=FakeLLM([response(content="hi")]))
    order = []

    def outer(ctx, request, proceed):
        order.append("outer:before")
        out = proceed(request)
        order.append("outer:after")
        return out

    def inner(ctx, request, proceed):
        order.append("inner:before")
        out = proceed(request)
        order.append("inner:after")
        return out

    rig.hooks.add("llm_call", outer)
    rig.hooks.add("llm_call", inner)

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert order == ["outer:before", "inner:before",
                     "inner:after", "outer:after"]


def test_an_outer_escort_sees_what_the_inner_one_answered():
    """The outer escort may rewrite the inner one's response."""
    rig = loop_rig(llm=FakeLLM([response(content="inner text")]))

    def outer(ctx, request, proceed):
        out = proceed(request)
        out.content = out.content.upper()
        return out

    def inner(ctx, request, proceed):
        return proceed(request)

    rig.hooks.add("llm_call", outer)
    rig.hooks.add("llm_call", inner)

    reply, _, _ = rig.loop.drive(rig.cs, "agent",
                                 [{"role": "user", "content": "go"}])

    assert reply == "INNER TEXT"


def test_an_abstaining_escort_does_not_cost_a_second_backend_call():
    """One escort abstaining and another dialing is still one round trip.

    The abstention path reuses whatever the inner layers already fetched
    (``_escort_layer``'s ``last["called"]``), so a stack of polite escorts
    cannot multiply the bill.
    """
    llm = FakeLLM([response(content="once")])
    rig = loop_rig(llm=llm)

    rig.hooks.add("llm_call", lambda ctx, req, proceed: None)
    rig.hooks.add("llm_call", lambda ctx, req, proceed: proceed(req))
    rig.hooks.add("llm_call", lambda ctx, req, proceed: None)

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert len(llm.calls) == 1


def test_a_raising_escort_does_not_stop_the_one_inside_it():
    """An escort that raises is skipped; the rest of the onion is intact."""
    llm = FakeLLM([response(content="survived")])
    rig = loop_rig(llm=llm)
    reached = []

    def broken(ctx, request, proceed):
        raise RuntimeError("no")

    def inner(ctx, request, proceed):
        reached.append(True)
        return proceed(request)

    rig.hooks.add("llm_call", broken)
    rig.hooks.add("llm_call", inner)

    reply, _, _ = rig.loop.drive(rig.cs, "agent",
                                 [{"role": "user", "content": "go"}])

    assert reply == "survived"
    assert reached == [True], "the inner escort was skipped with the outer one"
    assert len(llm.calls) == 1, "the transparent retry dialed twice"


# ──────────────────────────────────────────────────────────────────────
# end_turn — two doormen at one exit
# ──────────────────────────────────────────────────────────────────────

def test_the_first_decisive_doorman_wins_and_the_rest_are_not_asked():
    """First non-None answer wins; later doormen are never consulted."""
    rig = loop_rig(llm=FakeLLM([response(content="done"),
                                response(content="after note")]))
    asked = []

    def first(ctx, ending):
        asked.append("first")
        return SendBack("go back", ephemeral=True) if not asked[1:] else None

    def second(ctx, ending):
        asked.append("second")
        return None

    rig.hooks.add("end_turn", first)
    rig.hooks.add("end_turn", second)

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert asked[0] == "first"
    assert asked[1] != "second", "a decisive verdict still consulted the next"


def test_a_leading_allow_short_circuits_a_later_sendback():
    """``Allow`` is not the same as abstaining — it positively ends the turn.

    A policy that waves the agent through has to be able to beat a doorman
    registered after it, or ordering would be the only way to express
    precedence.
    """
    llm = FakeLLM([response(content="done")])
    rig = loop_rig(llm=llm)
    asked = []

    rig.hooks.add("end_turn", lambda ctx, e: (asked.append("allow"), Allow())[1])
    rig.hooks.add("end_turn",
                  lambda ctx, e: (asked.append("send"), SendBack("back"))[1])

    reply, _, _ = rig.loop.drive(rig.cs, "agent",
                                 [{"role": "user", "content": "go"}])

    assert asked == ["allow"], "the SendBack doorman was consulted anyway"
    assert reply == "done"
    assert len(llm.calls) == 1, "the turn went back inside despite Allow"


def test_a_raising_doorman_is_skipped_and_the_next_one_decides():
    """A broken doorman abstains; it does not silence the queue behind it."""
    rig = loop_rig(llm=FakeLLM([response(content="first"),
                                response(content="second try")]))

    def broken(ctx, ending):
        raise RuntimeError("no")

    rig.hooks.add("end_turn", broken)
    rig.hooks.add("end_turn", lambda ctx, e: (
        SendBack("say something", ephemeral=True)
        if e.doorman_fires == 0 else None))

    reply, _, _ = rig.loop.drive(rig.cs, "agent",
                                 [{"role": "user", "content": "go"}])

    assert reply == "second try"


def test_two_doormen_share_one_fire_budget():
    """The cap is per turn, not per doorman.

    Two stubborn doormen must not get ``DOORMAN_FIRE_LIMIT`` each — the budget
    exists so a turn cannot be trapped, and per-doorman accounting would make
    it scale with how many plugins are installed.
    """
    llm = FakeLLM()
    rig = loop_rig(llm=llm)
    limit = rig.loop.DOORMAN_FIRE_LIMIT

    rig.hooks.add("end_turn", lambda ctx, e: (
        SendBack("a", ephemeral=True) if e.doorman_fires % 2 == 0 else None))
    rig.hooks.add("end_turn", lambda ctx, e: (
        SendBack("b", ephemeral=True) if e.doorman_fires % 2 == 1 else None))

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert len(llm.calls) == limit + 1


# ──────────────────────────────────────────────────────────────────────
# shape_scope — the one doorway that folds
# ──────────────────────────────────────────────────────────────────────

def test_shapers_fold_so_the_second_sees_the_first_ones_answer():
    """Unlike a verdict, every shaper runs and each one narrows further."""
    from runtime.hooks import HookRegistry

    hooks = HookRegistry()
    seen = []

    def drop_admin(ctx, registry):
        seen.append(list(registry))
        return [n for n in registry if not n.startswith("admin_")]

    def drop_shell(ctx, registry):
        seen.append(list(registry))
        return [n for n in registry if "shell" not in n]

    hooks.add("shape_scope", drop_admin)
    hooks.add("shape_scope", drop_shell)

    out = hooks.shape_scope(SimpleNamespace(key="s"),
                            ["echo", "admin_wipe", "run_shell"])

    assert seen[0] == ["echo", "admin_wipe", "run_shell"]
    assert seen[1] == ["echo", "run_shell"], "the second shaper saw the original"
    assert out == ["echo"]


def _scope_rig(shaper):
    """A runtime whose global registry is real, with one shaper standing."""
    from agent.tool_registry import ToolRegistry
    from runtime.hooks import HookRegistry

    registry = ToolRegistry(None, {}, {})
    registry.tools = {"a": 1, "b": 2, "c": 3}
    hooks = HookRegistry()
    hooks.add("shape_scope", shaper)
    runtime = SimpleNamespace(tool_registry=registry, hooks=hooks, db=None,
                              config={}, is_attended=lambda key: True)
    session = SimpleNamespace(key="s", extra_tool_instances=[], cs=None,
                              profile_override=None, user_id=1,
                              frontend_name=None)
    return runtime, session, registry


def test_a_shaper_is_never_handed_the_global_registry():
    """Narrowing one session must not narrow the process.

    ``active_tool_registry`` layers global → profile-scoped → session-pinned
    and returns the deepest one that applies — so with no profile scope and no
    pinned extras, the "deepest layer" *is* ``runtime.tool_registry`` itself.
    Handing that to a shaper was handing it the one object every session reads,
    and ``narrow_scope`` writes ``visible_tool_names`` in place.
    """
    from sandbox.hooks import narrow_scope

    runtime, session, registry = _scope_rig(
        lambda ctx, reg: narrow_scope(reg, ["a"]))

    shaped = _cfg().active_tool_registry(runtime, session)

    assert shaped is not registry, "the shaper was handed the global registry"
    assert sorted(shaped.visible_tool_names) == ["a"]
    assert registry.visible_tool_names is None, (
        "a per-session narrowing escaped onto the global registry")


def test_a_shaper_can_widen_back_after_narrowing():
    """The shaper's answer is the answer, every time it is asked.

    ``narrow_scope`` intersects its answer with the registry's existing
    ``visible_tool_names``, which is right *within* one consultation — that is
    how two shapers fold. Across consultations on a shared object it became a
    ratchet: the visible set could only ever shrink, so a shaper that varies
    (the template's example keys on ``ctx.attended``, and one turn asks this
    doorway once per model call) walked its own scope toward empty and could
    never recover it.
    """
    wanted = {"keep": ["a", "b"]}
    from sandbox.hooks import narrow_scope

    runtime, session, registry = _scope_rig(
        lambda ctx, reg: narrow_scope(reg, wanted["keep"]))
    cfg = _cfg()

    seen = []
    for keep in (["a", "b"], ["a"], ["a", "b", "c"]):
        wanted["keep"] = keep
        shaped = cfg.active_tool_registry(runtime, session)
        seen.append(sorted(shaped.visible_tool_names))

    assert seen == [["a", "b"], ["a"], ["a", "b", "c"]], (
        "narrowing ratcheted; the shaper could not widen back")
    assert registry.visible_tool_names is None


def test_no_shaper_means_no_copy_is_made():
    """The detach is paid for only by installs that actually shape scope.

    Every install today has an empty ``shape_scope`` socket, so the common path
    must still hand back the global registry by identity — otherwise this fix
    would cost a ``ToolRegistry`` clone on every consultation, several times a
    turn, for a feature nobody uses.
    """
    from runtime.hooks import HookRegistry

    runtime, session, registry = _scope_rig(lambda ctx, reg: None)
    runtime.hooks = HookRegistry()          # nobody standing

    assert _cfg().active_tool_registry(runtime, session) is registry


def _cfg():
    """The module under test, imported late to settle the runtime cycle."""
    from runtime import runtime_config

    return runtime_config


def test_a_raising_shaper_leaves_the_fold_where_it_was():
    """A broken shaper is a no-op, not a reset to the unshaped registry."""
    from runtime.hooks import HookRegistry

    hooks = HookRegistry()

    def broken(ctx, registry):
        raise RuntimeError("no")

    hooks.add("shape_scope", lambda ctx, r: [n for n in r if n != "admin_wipe"])
    hooks.add("shape_scope", broken)
    hooks.add("shape_scope", lambda ctx, r: list(r))

    out = hooks.shape_scope(SimpleNamespace(key="s"), ["echo", "admin_wipe"])

    assert out == ["echo"], "a raising shaper undid an earlier narrowing"


# ──────────────────────────────────────────────────────────────────────
# vet_permission — two gates at one question
# ──────────────────────────────────────────────────────────────────────

def _gate(log, name, verdict):
    """A gate that records being asked and answers ``verdict``."""
    def gate(ctx, query):
        log.append(name)
        return verdict
    return gate


def test_a_refusal_beats_an_allow_registered_before_it():
    """Every gate is asked, and a deny wins however late it arrives.

    Under first-wins this returned the allow and never asked ``refuses`` at
    all — so which answer the kernel got depended on plugin load order, i.e.
    on filenames.
    """
    from runtime.hooks import HookRegistry

    hooks = HookRegistry()
    asked = []

    def broken(ctx, query):
        asked.append("broken")
        raise RuntimeError("no")

    hooks.add("vet_permission", broken)
    hooks.add("vet_permission",
              _gate(asked, "allows", PermissionVerdict(True, "fine")))
    hooks.add("vet_permission",
              _gate(asked, "refuses", PermissionVerdict(False, "nope")))

    verdict = hooks.vet_permission(SimpleNamespace(key="s"), "shell", "rm -rf /")

    assert asked == ["broken", "allows", "refuses"], (
        "a gate after an allow was not consulted")
    assert verdict.allow is False
    assert verdict.reason == "nope"


def test_the_verdict_does_not_depend_on_the_order_gates_loaded_in():
    """The property the change exists for, stated directly.

    Same two gates, both orders, same answer. This is the whole point: a
    security decision must not be a function of which file discovery reached
    first.
    """
    from runtime.hooks import HookRegistry

    allow = PermissionVerdict(True, "fine")
    deny = PermissionVerdict(False, "nope")
    answers = []
    for pair in ((allow, deny), (deny, allow)):
        hooks = HookRegistry()
        for verdict in pair:
            hooks.add("vet_permission", _gate([], "g", verdict))
        answers.append(hooks.vet_permission(
            SimpleNamespace(key="s"), "shell", "rm -rf /").allow)

    assert answers == [False, False]


def test_a_refusal_stops_the_walk():
    """A deny short-circuits, so the expensive case is the permissive one.

    Worth pinning because it is the mitigation for the cost this change
    introduces: gates are box round trips, and asking all of them is only paid
    when nobody objects.
    """
    from runtime.hooks import HookRegistry

    hooks = HookRegistry()
    asked = []
    hooks.add("vet_permission",
              _gate(asked, "refuses", PermissionVerdict(False, "nope")))
    hooks.add("vet_permission",
              _gate(asked, "after", PermissionVerdict(True, "fine")))

    hooks.vet_permission(SimpleNamespace(key="s"), "shell", "rm -rf /")

    assert asked == ["refuses"], "the walk continued past a refusal"


def test_an_answer_that_is_not_a_verdict_is_an_abstention():
    """Malformed answers abstain rather than being believed.

    ``sandbox.hooks.rebuild`` already draws this line for the sandboxed side.
    It matters more here than it looks: the approver reads ``verdict.allow``
    without guarding it, so an object lacking that attribute used to reach it
    and raise.
    """
    from runtime.hooks import HookRegistry

    hooks = HookRegistry()
    hooks.add("vet_permission", lambda ctx, query: "yes please")
    hooks.add("vet_permission",
              _gate([], "real", PermissionVerdict(True, "fine")))

    verdict = hooks.vet_permission(SimpleNamespace(key="s"), "shell", "ls")

    assert verdict.allow is True, "the string was read as a verdict"


# ──────────────────────────────────────────────────────────────────────
# Doorways meeting each other
# ──────────────────────────────────────────────────────────────────────

def test_a_required_tool_is_forced_only_when_the_drive_brain_supports_it():
    """``RequireTool`` pins ``tool_choice`` against the *drive's* brain.

    This is deliberate and documented at ``conversation_loop.py:1332`` — an
    escort that swaps ``request.llm`` per call keeps the pin only if its brain
    also honours ``tool_choice``. Pinned here because it is exactly the kind
    of cross-doorway assumption that reads like a bug later.
    """
    tools, schemas = echo_tool()
    # The drive's brain does NOT support tool_choice.
    llm = FakeLLM([
        response(content="done"),
        response(tool_calls=[tool_call("echo")]),
        response(content="finished"),
    ])
    rig = loop_rig(tools=tools, schemas=schemas, llm=llm)

    rig.hooks.add("end_turn", lambda ctx, e: (
        RequireTool("echo", note="call it") if e.doorman_fires == 0 else None))

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    forced = [rec["kwargs"].get("tool_choice") for rec in rig.llm.records]
    assert not any(forced), (
        "tool_choice was pinned against a backend that does not support it")
    # The note still went, so the demand degrades rather than disappearing.
    sent = [m for call in rig.llm.calls for m in call]
    assert any("call it" in str(m.get("content", "")) for m in sent)


def test_an_escort_sees_the_call_a_doorman_sent_back_for():
    """A doorman's comeback call goes through the escort onion like any other."""
    llm = FakeLLM([response(content="first"), response(content="second")])
    rig = loop_rig(llm=llm)
    escorted = []

    def escort(ctx, request, proceed):
        escorted.append(len(request.messages))
        return proceed(request)

    rig.hooks.add("llm_call", escort)
    rig.hooks.add("end_turn", lambda ctx, e: (
        SendBack("again please", ephemeral=True)
        if e.doorman_fires == 0 else None))

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert len(escorted) == 2, "the comeback call skipped the escort"
    assert escorted[1] > escorted[0], "the ephemeral note never reached the call"


def test_the_over_budget_wrapup_call_is_escorted_too():
    """The kernel's own default doorman still dials through the onion."""
    tools, schemas = echo_tool()
    llm = FakeLLM([response(tool_calls=[tool_call("echo", call_id=f"c{i}")])
                   for i in range(30)])
    rig = loop_rig(tools=tools, schemas=schemas, llm=llm, max_tool_calls=2)
    reasons = []

    rig.hooks.add("llm_call", lambda ctx, r, p: p(r))
    rig.hooks.add("end_turn", lambda ctx, e: reasons.append(e.reason))

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert "budget_exhausted" in reasons


# ──────────────────────────────────────────────────────────────────────
# The escort and the kernel's own layers
# ──────────────────────────────────────────────────────────────────────

def test_an_escort_sees_one_response_for_a_compaction_retry():
    """The compaction layer sits *inside* the escorts, so a retry is invisible.

    An escort counting its own ``proceed`` calls therefore under-counts real
    backend round trips. That is the documented stacking order — context
    safety must not be something a plugin can wrap and skip — but it means an
    escort cannot be used to meter provider spend.
    """
    llm = _OverflowLLM([response(content="Recovered.")], overflows=1)
    rig = loop_rig(llm=llm)
    rig.runtime.services = {"compactor": _Compactor()}
    dials = []

    def escort(ctx, request, proceed):
        out = proceed(request)
        dials.append(out)
        return out

    rig.hooks.add("llm_call", escort)

    reply, _, _ = rig.loop.drive(
        rig.cs, "agent",
        [{"role": "user", "content": "a"},
         {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}])

    assert reply == "Recovered."
    assert len(dials) == 1, "the escort saw the compaction retry"
    assert len(llm.calls) == 2, "the backend was not actually retried"


def test_a_compaction_retry_discards_an_escorts_message_rewrite():
    """What an escort added to ``messages`` does not survive an overflow.

    ``_compaction_layer.rebuilt`` rebuilds the prompt from ``history``, so the
    retry is the *kernel's* messages, not the escort's. Recorded because it is
    silent: the escort is never told, and its injection simply is not there.
    """
    llm = _OverflowLLM([response(content="Recovered.")], overflows=1)
    rig = loop_rig(llm=llm)
    rig.runtime.services = {"compactor": _Compactor()}

    def escort(ctx, request, proceed):
        request.messages = request.messages + [
            {"role": "user", "content": "ESCORT MARKER"}]
        return proceed(request)

    rig.hooks.add("llm_call", escort)

    rig.loop.drive(
        rig.cs, "agent",
        [{"role": "user", "content": "a"},
         {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}])

    first, retry = llm.calls[0], llm.calls[1]
    assert any("ESCORT MARKER" in str(m.get("content", "")) for m in first)
    assert not any("ESCORT MARKER" in str(m.get("content", "")) for m in retry)


def test_an_escort_brain_swap_survives_a_compaction_retry():
    """``rebuilt()`` keeps ``llm``/``tools``/``params`` even as it drops messages.

    The counterpart to the test above: the retry loses the escort's *content*
    edits but must not silently retarget the call to a different model.
    """
    llm = _OverflowLLM([response(content="Recovered.")], overflows=1)
    rig = loop_rig(llm=llm)
    rig.runtime.services = {"compactor": _Compactor()}
    names = []

    def escort(ctx, request, proceed):
        request.params = {**(request.params or {}), "marker": "kept"}
        names.append(request.llm)
        return proceed(request)

    rig.hooks.add("llm_call", escort)

    rig.loop.drive(
        rig.cs, "agent",
        [{"role": "user", "content": "a"},
         {"role": "assistant", "content": "b"},
         {"role": "user", "content": "c"}])

    kwargs = [rec["kwargs"] for rec in llm.records]
    assert kwargs and all(k.get("marker") == "kept" for k in kwargs), (
        "provider params set by an escort were dropped on the retry")


# ──────────────────────────────────────────────────────────────────────
# turn_start's three effects, all at once
# ──────────────────────────────────────────────────────────────────────

def test_a_queued_action_runs_before_the_model_is_ever_consulted():
    """``pending_agent_actions`` is drained ahead of the LLM call.

    Already true for one hook; asserted here alongside the others because the
    three ``turn_start`` effects share a session and have never been exercised
    together.
    """
    calls = []
    tools, schemas = echo_tool(calls)
    llm = FakeLLM([response(content="done")])
    rig = loop_rig(tools=tools, schemas=schemas, llm=llm)
    rig.session.pending_agent_actions.append(
        {"name": "echo", "args": {"from": "hook"}, "forced_by": "test"})

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert calls == [{"from": "hook"}], "the queued action did not run"
    assert len(llm.calls) == 1


def test_a_queued_action_does_not_outlive_the_turn_that_queued_it():
    """``pending_agent_actions`` is cleared at turn end, like its twin.

    ``runtime/session.py`` had always documented this field as sharing
    ``staged_attachments``' lifecycle — "ephemeral, cleared at turn end" — and
    ``finish_turn`` cleared only the attachments. Most turns drain the queue at
    a loop boundary and never arrive here holding anything; a turn that ended
    some other way (failed action, cancel, priority handoff) left the action
    to fire on somebody's *next* turn, against state the queuing hook never
    saw.
    """
    from runtime.hooks import HookRegistry, TurnOutcome

    session = SimpleNamespace(key="s", pending_agent_actions=[
        {"name": "echo", "args": {}}], staged_attachments=[object()],
        turn_security_mode=None)

    HookRegistry().finish_turn(session, TurnOutcome())

    assert session.staged_attachments == []
    assert session.pending_agent_actions == []


# ──────────────────────────────────────────────────────────────────────
# Everyone at once
# ──────────────────────────────────────────────────────────────────────

def test_five_doorways_in_one_turn_are_visited_in_ritual_order():
    """The whole ritual, with a probe standing at every loop-level doorway.

    ``turn_start``/``turn_finish`` live on the runtime and are covered in
    ``tests/test_hooks_turn_paths.py``; this is the three the loop owns.
    """
    journal = []
    tools, schemas = echo_tool()
    llm = FakeLLM([response(tool_calls=[tool_call("echo")]),
                   response(content="done")])
    rig = loop_rig(tools=tools, schemas=schemas, llm=llm)
    probes = native_probe(journal, ["llm_call", "end_turn"])
    for moment, fn in probes.items():
        rig.hooks.add(moment, fn)

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert moments_in(journal) == ["llm_call", "llm_call", "end_turn"]
    assert journal[-1]["reason"] == "model_finished"
    assert journal[-1]["final_text"] == "done"
