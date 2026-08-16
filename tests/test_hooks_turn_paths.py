"""Which doorways a turn visits, on each of the ways a turn can end.

``ConversationLoop.drive`` has nine distinct exits and ``end_turn`` is
consulted on only two of them. That table lives in docstrings and is asserted
nowhere, which makes it the easiest thing in the hook system to break by
accident: a doorman that stops being consulted is silent, and looks exactly
like a doorman with no opinion.

So each test here drives a turn out by one specific route and asserts *the
complete set of moments that fired*. Native probes, in-process — the sandboxed
half is ``tests/test_hooks_live_turn.py``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

# Import the state_machine package before runtime modules to settle the
# package-init circular import.
import state_machine  # noqa: F401

from runtime.hooks import Redrive, SendBack
from state_machine.errors import ActionResult
from tests.support import (FakeLLM, FakeRegistry, echo_tool, loop_rig,
                           make_runtime, moments_in, native_probe, response,
                           tool_call)

ALL = ["turn_start", "shape_scope", "vet_permission", "llm_call", "end_turn",
       "turn_finish"]


def _runtime_rig(tmp_path, responses, moments=ALL, **kwargs):
    """A real runtime with probes standing at ``moments``."""
    kwargs.setdefault("tool_registry", FakeRegistry([]))
    rt, session, llm = make_runtime(tmp_path, responses, **kwargs)
    journal = []
    for moment, fn in native_probe(journal, moments).items():
        rt.hooks.add(moment, fn)
    return rt, session, llm, journal


def visited(journal):
    """The moments that fired, deduplicated and in first-seen order.

    ``shape_scope`` fires several times per turn (see
    :func:`test_a_shaper_is_consulted_four_times_for_one_turn`), which is real
    but is not what these tests are about.
    """
    seen = []
    for moment in moments_in(journal):
        if moment not in seen:
            seen.append(moment)
    return seen


# ──────────────────────────────────────────────────────────────────────
# The ordinary way out
# ──────────────────────────────────────────────────────────────────────

def test_a_finished_turn_visits_every_doorway_in_ritual_order(tmp_path):
    """The baseline: model answers, doorman lets it go, observers see it."""
    rt, _, _, journal = _runtime_rig(tmp_path, [response(content="hi")])

    rt.handle_action("s", "send_text", "hello")

    assert visited(journal) == ["shape_scope", "turn_start", "llm_call",
                                "end_turn", "turn_finish"]
    assert journal[-1]["ok"] is True
    assert journal[-1]["cancelled"] is False
    assert journal[-1]["final_text"] == "hi"
    assert journal[-1]["reason"] == "model_finished"


def test_the_envelope_identifies_the_real_session(tmp_path):
    """Every doorway is told whose turn it is, from the live session."""
    rt, session, _, journal = _runtime_rig(tmp_path, [response(content="hi")])
    session.user_id = 7

    rt.handle_action("s", "send_text", "hello")

    for entry in journal:
        assert entry["session_key"] == "s"
        assert entry["user_id"] == 7
        assert entry["conversation_id"] == session.conversation_id
        assert entry["attended"] is True


def test_a_shaper_is_consulted_once_per_model_call_plus_three(tmp_path):
    """``shape_scope`` is not once-per-turn, and the cost is per consultation.

    Four consultations here, and the arithmetic is ``3 + N`` where N is the
    number of model calls:

    - ``refresh_specs`` — **2**, via ``tool_specs_for`` and
      ``scoped_tool_names``, once per dispatch.
    - ``build_loop`` — **1**, once per drive.
    - the ``_session_prompt`` closure (``runtime_config.py:349``) — **1 per
      model call**, because the prompt renders the visible tool list.

    So a six-tool-call turn reaches this doorway ten times, not four. Each
    caller genuinely needs current scope, so the count is correct; what is
    wrong is ``templates/hook_template.py`` telling an author they pay "a box
    round trip" per hook per turn. A sandboxed shaper pays one *per
    consultation*, and the number grows with the turn.

    This docstring used to name ``new_state`` as the fourth. It is a real
    consultation site but it runs at session open and conversation load, never
    inside a turn — on this rig it does not fire at all, and the fourth is the
    prompt closure. The count was right and the attribution was wrong, which is
    the kind of comment that sends the next reader to the wrong function.
    """
    rt, _, _, journal = _runtime_rig(tmp_path, [response(content="hi")],
                                     moments=["shape_scope"])

    rt.handle_action("s", "send_text", "hello")

    assert moments_in(journal).count("shape_scope") == 4


def test_a_shaper_is_also_consulted_outside_any_turn(tmp_path):
    """Loading a conversation walks to ``shape_scope`` before a turn exists.

    ``load_conversation`` refreshes tool specs, which consults the doorway —
    so a shaper runs at moments when there is no turn, and ``ctx.attended`` is
    ``False`` there because no session is active yet.

    That matters for the shaper the template teaches you to write: one that
    narrows the toolbox when nobody is watching sees ``attended=False`` during
    an ordinary attended conversation load. The value is truthful about the
    instant it describes; it is just not a statement about the turn, because
    at that point there is no turn.
    """
    from pipeline.database import Database
    from runtime.conversation_runtime import ConversationRuntime

    db = Database(str(tmp_path / "load.db"))
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db,
                             services={"llm": FakeLLM([response(content="hi")])},
                             config={}, tool_registry=FakeRegistry([]))
    journal = []
    for moment, fn in native_probe(journal, ["shape_scope"]).items():
        rt.hooks.add(moment, fn)

    rt.load_conversation("s", cid)

    assert journal, "loading a conversation consulted no shaper at all"
    assert all(entry["attended"] is False for entry in journal), (
        "attendance outside a turn is no longer False; check what changed")

    before = len(journal)
    rt.handle_action("s", "send_text", "hello")

    assert all(entry["attended"] is True for entry in journal[before:]), (
        "the same doorway disagreed about attendance inside the turn")


# ──────────────────────────────────────────────────────────────────────
# The doorman's own exits
# ──────────────────────────────────────────────────────────────────────

def test_a_sendback_keeps_the_turn_inside_and_still_finishes_once(tmp_path):
    """The agent goes back in, comes out again, and the turn ends once."""
    rt, _, llm, journal = _runtime_rig(
        tmp_path, [response(content="first"), response(content="second")])
    rt.hooks.add("end_turn", lambda ctx, e: (
        SendBack("go back", ephemeral=True) if e.doorman_fires == 0 else None))

    rt.handle_action("s", "send_text", "hello")

    assert moments_in(journal).count("turn_start") == 1
    assert moments_in(journal).count("turn_finish") == 1
    assert moments_in(journal).count("end_turn") == 2
    assert len(llm.calls) == 2


def test_a_redrive_is_the_same_logical_turn(tmp_path):
    """``Redrive`` re-drives without re-starting: one start, one finish.

    Two drives, two ``end_turn`` consultations, but the turn began once and
    ended once — which is the whole distinction between a drive and a turn.
    """
    fired = {"n": 0}

    def once(ctx, ending):
        fired["n"] += 1
        return Redrive() if fired["n"] == 1 else None

    rt, _, _, journal = _runtime_rig(
        tmp_path, [response(content="first"), response(content="second")])
    rt.hooks.add("end_turn", once)

    rt.handle_action("s", "send_text", "hello")

    assert moments_in(journal).count("turn_start") == 1, (
        "the re-drive re-ran the turn starters")
    assert moments_in(journal).count("turn_finish") == 1, (
        "the observers fired on a turn that had not ended")
    assert fired["n"] == 2
    assert journal[-1]["reason"] == "model_finished", (
        "the observer was told about the drive, not the turn")


def test_a_doorman_that_always_redrives_is_bounded_by_the_drive_budget(tmp_path):
    """``Redrive`` consumes no fire, so only the outer drive cap stops it.

    ``_doorman_fires`` resets per *drive*, and a ``Redrive`` verdict never
    increments it — so the fire budget is no defence here at all. What bounds
    it is ``drives < 5`` in the runtime's own loop. Worth pinning: it is the
    one verdict a stubborn doorman can repeat for free.
    """
    fired = {"n": 0}

    def always(ctx, ending):
        fired["n"] += 1
        return Redrive()

    rt, _, _, journal = _runtime_rig(
        tmp_path, [response(content=f"r{i}") for i in range(10)])
    rt.hooks.add("end_turn", always)

    rt.handle_action("s", "send_text", "hello")

    assert fired["n"] == 5, "the drive budget did not bound a redriving doorman"
    assert moments_in(journal).count("turn_finish") == 1
    # The one case where an observer sees ``"redrive"`` at all: the turn asked
    # to go round again and the drive budget said no. On a redrive that *is*
    # allowed, the observers wait for the drive that ends the turn.
    assert journal[-1]["reason"] == "redrive"


# ──────────────────────────────────────────────────────────────────────
# The exits where no doorman is consulted
# ──────────────────────────────────────────────────────────────────────

def test_a_cancelled_turn_skips_the_doorman_and_still_finishes(tmp_path):
    """Cancellation ends the turn without asking anybody's permission.

    ``turn_finish`` still fires, with ``cancelled=True`` — an observer is told
    the turn is over even when nothing else was.
    """
    rt, session, _, journal = _runtime_rig(tmp_path, [response(content="hi")])

    def cancel_mid_call(ctx, request, proceed):
        session.cancel_event.set()
        return proceed(request)

    rt.hooks.add("llm_call", cancel_mid_call)

    rt.handle_action("s", "send_text", "hello")

    moments = moments_in(journal)
    assert "end_turn" not in moments, "a cancelled turn consulted a doorman"
    assert moments.count("turn_finish") == 1
    assert journal[-1]["cancelled"] is True
    assert journal[-1]["reason"] == "cancelled"


def test_a_priority_handoff_leaves_without_consulting_a_doorman():
    """Handing priority back mid-turn skips the whole end-of-drive block.

    ``drive``'s closing block is guarded by ``turn_priority == actor_id``, so
    a tool that hands priority to the user takes the barrier and the doormen
    with it. Loop-level, because the handoff is a state-machine act.
    """
    journal = []
    seen = []

    def hand_off(cs, actor, args):
        from plugins.native.tool import ToolResult
        cs.set_priority("user")
        seen.append(True)
        return ToolResult(llm_summary="handed off", data={})

    tools, schemas = echo_tool()
    tools["echo"].handler = hand_off
    llm = FakeLLM([response(tool_calls=[tool_call("echo")]),
                   response(content="never reached")])
    rig = loop_rig(tools=tools, schemas=schemas, llm=llm)
    for moment, fn in native_probe(journal, ["llm_call", "end_turn"]).items():
        rig.hooks.add(moment, fn)

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert seen == [True]
    assert "end_turn" not in moments_in(journal)
    assert rig.loop._exit_reason == "priority_handoff"


def test_a_failed_action_suppresses_the_over_budget_wrapup():
    """A failed action ends the turn without the "you hit the limit" nudge.

    The closing block runs the wrap-up only when the loop genuinely ran out of
    budget — a failed action reaching it would make that premise false, and
    the agent would be told it had exhausted a budget it never touched. So the
    doorman is consulted once at ``model_finished`` (the enact that then
    failed) and never again at ``budget_exhausted``.
    """
    journal = []
    rig = loop_rig(llm=FakeLLM([response(content="done")]))
    for moment, fn in native_probe(journal, ["end_turn"]).items():
        rig.hooks.add(moment, fn)

    real = rig.loop._enact_logged

    def failing(cs, action_type, content, actor_id):
        if action_type == "end_turn":
            return ActionResult.fail("end_turn", "nope", code="broken")
        return real(cs, action_type, content, actor_id)

    rig.loop._enact_logged = failing

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    reasons = [entry["reason"] for entry in journal]
    assert reasons == ["model_finished"], (
        "a failed action was told it had exhausted its budget")
    assert rig.loop._exit_reason == "action_failed"


def test_a_tool_asking_for_a_restart_exits_without_a_doorman():
    """``session.restart_turn`` is the mid-turn spelling of ``Redrive``.

    Set by a tool it exits the drive at ``_restart_requested()``, before the
    closing block — so no doorman, no barrier, no ``end_turn`` enact.
    """
    journal = []

    def ask_restart(cs, actor, args):
        from plugins.native.tool import ToolResult
        return ToolResult(llm_summary="restarting", data={})

    tools, schemas = echo_tool()
    tools["echo"].handler = ask_restart
    llm = FakeLLM([response(tool_calls=[tool_call("echo")]),
                   response(content="unreached")])
    rig = loop_rig(tools=tools, schemas=schemas, llm=llm)
    for moment, fn in native_probe(journal, ["llm_call", "end_turn"]).items():
        rig.hooks.add(moment, fn)

    # A tool sets this on the session; simulate by setting it after the call.
    real = rig.loop._enact_logged

    def then_restart(cs, action_type, content, actor_id):
        out = real(cs, action_type, content, actor_id)
        if action_type == "call_tool":
            rig.session.restart_turn = True
        return out

    rig.loop._enact_logged = then_restart

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    assert "end_turn" not in moments_in(journal), (
        "a restart-requesting tool still went past a doorman")
    assert rig.loop._exit_reason == "redrive"


# ──────────────────────────────────────────────────────────────────────
# Running out of budget
# ──────────────────────────────────────────────────────────────────────

def test_budget_exhaustion_reaches_the_doorman_with_its_own_reason():
    """The kernel's wrap-up is a doorman consultation, at a different reason."""
    journal = []
    tools, schemas = echo_tool()
    llm = FakeLLM([response(tool_calls=[tool_call("echo", call_id=f"c{i}")])
                   for i in range(30)])
    rig = loop_rig(tools=tools, schemas=schemas, llm=llm, max_tool_calls=2)
    for moment, fn in native_probe(journal, ["end_turn"]).items():
        rig.hooks.add(moment, fn)

    rig.loop.drive(rig.cs, "agent", [{"role": "user", "content": "go"}])

    reasons = [e["reason"] for e in journal]
    assert "budget_exhausted" in reasons
    assert reasons[-1] == "budget_exhausted"
    assert rig.loop._exit_reason == "budget_exhausted"


def test_a_crashed_turn_is_named_by_the_runtime_not_the_loop(tmp_path):
    """A crash is the one ending the loop cannot label itself.

    ``drive`` raised, so it never reached the assignment at any of its exits —
    the reason has to come from the caller, which is also why ``loop`` is
    hoisted above the ``try``: assigned inside it, the name is unbound on
    exactly the path that most needs an answer.
    """
    rt, _, llm, journal = _runtime_rig(tmp_path, [response(content="hi")],
                                       moments=["turn_finish"])

    def explode(*_args, **_kwargs):
        """A backend that fails outright."""
        raise ValueError("the model exploded")

    # Break the backend rather than a hook: a raising *escort* is swallowed by
    # the onion, which is the whole point of that layer, so it would never
    # reach the runtime as a crash.
    llm.chat = explode

    rt.handle_action("s", "send_text", "hello")

    assert journal, "a crashed turn told its observers nothing"
    assert journal[-1]["ok"] is False
    assert journal[-1]["reason"] == "crashed"


def test_every_exit_names_itself(tmp_path):
    """No ending may reach an observer unlabelled.

    The blank default is deliberately readable rather than guessed at, so it
    would be easy for a new exit to ship saying nothing — and an observer
    branching on ``reason`` would silently fall through its every case. This
    is the guard: whatever the turn did, it says what it did.
    """
    from runtime.hooks import Redrive, SendBack

    cases = {
        "plain": (None, "model_finished"),
        "sent back": (lambda rt, s: rt.hooks.add("end_turn", lambda c, e: (
            SendBack("again", ephemeral=True) if e.doorman_fires == 0
            else None)), "model_finished"),
        "cancelled": (lambda rt, s: rt.hooks.add(
            "llm_call",
            lambda c, r, p: (s.cancel_event.set(), p(r))[1]), "cancelled"),
        "redrive denied": (lambda rt, s: rt.hooks.add(
            "end_turn", lambda c, e: Redrive()), "redrive"),
    }
    for name, (setup, expected) in cases.items():
        rt, session, _, journal = _runtime_rig(
            tmp_path, [response(content=f"r{i}") for i in range(10)],
            moments=["turn_finish"], name=f"{name.replace(' ', '_')}.db")
        if setup is not None:
            setup(rt, session)

        rt.handle_action("s", "send_text", "hello")

        assert journal, f"{name}: no observer fired"
        assert journal[-1]["reason"] == expected, name


# ──────────────────────────────────────────────────────────────────────
# Turns nobody is watching
# ──────────────────────────────────────────────────────────────────────

def test_a_background_drive_visits_the_same_doorways(tmp_path):
    """``iterate_agent_turn`` is the same ritual, not a reduced one.

    Background drives are where a hook matters most (nobody is watching), so
    a doorway that quietly did not fire there would be the worst version of
    this bug.
    """
    rt, _, _, journal = _runtime_rig(tmp_path, [response(content="bg")])

    rt.iterate_agent_turn("s", "do the thing")

    assert visited(journal) == ["shape_scope", "turn_start", "llm_call",
                                "end_turn", "turn_finish"]


def test_an_unattended_session_is_reported_as_unattended(tmp_path):
    """``ctx.attended`` is what a hook branches on to decide how to behave.

    The template's worked example uses it to tell an agent not to ask
    questions, so a doorway that always reported ``True`` would make every
    background turn look supervised.
    """
    rt, session, _, journal = _runtime_rig(tmp_path, [response(content="hi")])
    rt.set_session_attended("s", False)

    rt.handle_action("s", "send_text", "hello")

    assert journal, "no doorway was visited"
    assert all(entry["attended"] is False for entry in journal)
