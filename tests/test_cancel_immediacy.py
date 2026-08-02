"""``/cancel`` ends the turn where it stands.

Cancellation used to be a flag nobody watched. ``session.cancel_event`` is read
*between* actions, and everything slow lives inside one, so a cancel was only
ever as immediate as the current model or tool call — and for a streaming model
call it was not immediate at all, because a backend pushing tokens makes no
Request the kernel could refuse.

These tests pin the two halves of the fix. First that the *stoppers* exist and
reach the right thing: a session's interrupt registry, a brain evicting the box
it just killed, a sandbox cancelling only the runs belonging to one session.
Second, and just as load-bearing, that **nothing from the agent is rendered
after the cancel lands** — the narration, the ✕, the final reply, the tail
message and the whole extra turn the subagent barrier used to force.

The failure they guard against is silent in the worst direction: every one of
those paths looks like the system working, and the person is left watching an
agent they told to stop.
"""

import threading

import pytest

from state_machine.conversation_phases import BASE_PHASE

from runtime.conversation_loop import ConversationLoop
from runtime.session import RuntimeSession

from tests.support import FakeLLM, FakeRegistry, agent_state, response


# ── The registry ─────────────────────────────────────────────────────

def _session(key="s"):
    return RuntimeSession(key=key, cs=agent_state())


def test_an_armed_stopper_fires_on_interrupt():
    session = _session()
    fired = []

    with session.interruptible() as slot:
        assert slot.arm(lambda: fired.append(True)) is True
        assert session.interrupt() == 1

    assert fired == [True]


def test_a_slot_that_never_armed_stops_nothing():
    """Reaching the block is not the same as reaching the arming point.

    A model call queued behind a busy box has no box to kill yet, and the
    registry must say so rather than counting the slot.
    """
    session = _session()
    with session.interruptible():
        assert session.interrupt() == 0


def test_arming_is_refused_once_the_cancel_has_already_fired():
    """The race the whole mechanism exists to close.

    A cancel landing between the loop's last check and the next call would
    otherwise park a stopper nobody is left to fire — and the turn would block
    on exactly the thing it was just told to stop.
    """
    session = _session()
    session.cancel_event.set()

    with session.interruptible() as slot:
        assert slot.arm(lambda: pytest.fail("must not be armed")) is False
        assert slot.armed is False
        assert session.interrupt() == 0


def test_slots_leave_by_identity_so_nested_calls_do_not_corrupt_the_registry():
    """Compaction retries re-enter the model call, so slots nest.

    Removal by position would be invalidated by whichever slot happens to
    leave first — and the survivor's arm would land on somebody else's entry
    or off the end of the list.
    """
    session = _session()
    fired = []
    with session.interruptible() as outer:
        with session.interruptible() as inner:
            inner.arm(lambda: fired.append("inner"))
        outer.arm(lambda: fired.append("outer"))
        assert session.interrupt() == 1
    assert fired == ["outer"]
    assert session._interrupts == []


def test_a_stopper_that_raises_does_not_take_the_cancel_down():
    session = _session()
    fired = []
    with session.interruptible() as first, session.interruptible() as second:
        first.arm(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        second.arm(lambda: fired.append(True))
        assert session.interrupt() == 2
    assert fired == [True]


# ── The model-call stopper ───────────────────────────────────────────

class _DeadBox:
    """A box that records being interrupted."""

    def __init__(self, name="box"):
        self.name = name
        self.alive = True
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True
        self.alive = False


def test_interrupting_a_brain_evicts_the_box_from_the_pool_not_just_the_idle_list():
    """The half that is easy to miss, because ``_release`` hides it.

    ``_grow`` counts ``_boxes`` against the ceiling and ``_lease`` hands back
    ``_boxes[0]`` once there, so a dead box left in that list means the pool
    never reopens and starts leasing a corpse — every later call failing with
    "box is not running", one cancel after the pool filled up.
    """
    from llm.registry import Brain

    brain = Brain("p", {})
    box = _DeadBox()
    brain._boxes.append(box)
    brain._idle.append(box)

    brain._interrupt(box)

    assert box.interrupted is True
    assert box not in brain._boxes
    assert box not in brain._idle


def test_the_loop_arms_the_box_the_brain_leased():
    """``on_call`` is the only way the cancel path learns what to stop."""
    session = _session()
    armed = []

    class _Brain(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            assert on_call is not None
            on_call(lambda: armed.append("stopped"))
            # Armed *now*, mid-call, which is the whole point of the slot.
            assert session.interrupt() == 1
            return response(content="too late")

    loop = ConversationLoop(_Brain(), FakeRegistry([]), {}, "prompt")
    loop.runtime = _FakeRuntime(session)
    loop.session_key = session.key
    loop.drive(session.cs, "agent", [{"role": "user", "content": "hi"}])

    assert armed == ["stopped"]


class _FakeRuntime:
    """Just enough runtime for the loop to find its session."""

    def __init__(self, session):
        self.sessions = {session.key: session}
        self.hooks = None
        self.pushed = []

    def push_message(self, key, text):
        self.pushed.append(text)


# ── Nothing renders after the cancel ─────────────────────────────────

def _cancelled_loop(llm, session=None):
    session = session or _session()
    loop = ConversationLoop(llm, FakeRegistry([]), {}, "prompt")
    runtime = _FakeRuntime(session)
    loop.runtime = runtime
    loop.session_key = session.key
    loop.cancel_event = session.cancel_event
    return loop, session, runtime


def test_a_reply_that_arrives_after_the_cancel_is_dropped():
    """The model answered; the person had already said stop."""
    streamed = []

    class _Late(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            session.cancel_event.set()
            return response(content="here is that answer you cancelled")

    session = _session()
    loop, session, _ = _cancelled_loop(_Late(), session)
    loop.on_delta = streamed.append

    reply, new_messages, _ = loop.drive(
        session.cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply is None
    assert not [m for m in new_messages if m.get("role") == "assistant"]
    assert not any(frame.get("delta") for frame in streamed)


def test_narration_alongside_a_tool_call_is_dropped_too():
    """The mid-turn "let me just..." line, pushed straight to the frontend.

    This is the one that actually lands on screen a beat after ``/cancel``.
    """
    class _Late(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            session.cancel_event.set()
            return response(
                content="Right — let me just check one more thing.",
                tool_calls=[{"id": "t1", "name": "noop", "arguments": "{}"}])

    session = _session()
    loop, session, runtime = _cancelled_loop(_Late(), session)

    loop.drive(session.cs, "agent", [{"role": "user", "content": "hi"}])

    assert runtime.pushed == []


def test_a_cancelled_turn_does_not_run_the_subagent_barrier():
    """A collecting barrier sets ``restarting``, and a re-drive is a whole
    fresh agent turn arriving after the person said stop — uncancelled, too,
    since the flag is cleared on the way past."""
    session = _session()
    loop, session, _ = _cancelled_loop(FakeLLM([response(content="hi")]), session)
    asked = []

    class _Subagents:
        def barrier(self, _session):
            asked.append(True)
            return True

    loop.runtime.subagents = _Subagents()
    session.cancel_event.set()

    loop.drive(session.cs, "agent", [{"role": "user", "content": "hi"}])

    assert asked == []
    assert session.restart_turn is False


def test_an_interrupted_tool_reads_as_interrupted_rather_than_as_a_dead_box():
    """What the box reports is a corpse's error. Handing the model "box died
    during 'run'" next turn invites a retry of work the person just stopped."""
    session = _session()
    loop, session, _ = _cancelled_loop(FakeLLM([]), session)
    session.cancel_event.set()

    from state_machine.errors import ActionResult

    text, paths = loop._format_tool_result(
        "edit_file",
        ActionResult.fail("call_tool", "box 'tool_edit_file' died during 'run'"),
        {})

    assert "Interrupted by user." in text
    assert "died during" not in text
    assert paths == []


def test_a_cancelled_turn_renders_no_tool_status():
    session = _session()
    loop, session, _ = _cancelled_loop(FakeLLM([]), session)
    session.cancel_event.set()
    seen = []
    loop.on_tool_result = lambda *a: seen.append(a)

    loop._tool_finished(("edit_file", "t1"), result=None, error="killed")
    assert seen == []

    session.cancel_event.clear()
    loop._tool_finished(("edit_file", "t1"), result=None, error="killed")
    assert len(seen) == 1


# ── End to end, through the real runtime ─────────────────────────────

def test_cancel_returns_while_the_model_is_still_blocked(conv_runtime):
    """The whole point, stated once: ``/cancel`` does not wait for the model.

    The fake blocks in ``chat`` exactly as a real backend blocks reading its
    box's pipe. Before the interrupt registry the cancel could not return
    until the model did.
    """
    entered = threading.Event()
    release = threading.Event()

    class _Blocking(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            if on_call is not None:
                on_call(release.set)
            entered.set()
            if not release.wait(timeout=10):
                raise AssertionError("never interrupted")
            raise RuntimeError("box 'llm_0_0' died during '__chat__'")

    rt, session, _ = conv_runtime()
    rt.services["llm"] = _Blocking()

    turn = threading.Thread(
        target=rt.handle_action, args=(session.key, "send_text", "go"),
        daemon=True)
    turn.start()
    assert entered.wait(timeout=10), "the model call never started"

    out = rt.handle_action(session.key, "cancel")

    assert out.messages == ["Cancelled."]
    turn.join(timeout=10)
    assert not turn.is_alive()


def test_the_interrupted_turn_reports_cancellation_rather_than_an_error(conv_runtime):
    """Killing the box is the mechanism working, not a fault.

    Left unhandled it puts ``Error: box 'llm_0_0' died during '__chat__'`` on
    screen immediately after ``Cancelled.`` — alarming, and the exact thing
    the change promised not to do.
    """
    class _Interrupted(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            session.cancel_event.set()
            raise RuntimeError("box 'llm_0_0' died during '__chat__'")

    rt, session, _ = conv_runtime()
    rt.services["llm"] = _Interrupted()

    out = rt.handle_action(session.key, "send_text", "go")

    assert out.error is None
    assert out.messages == []
    assert session.cs.turn_priority == "user"
    assert session.cs.phase == BASE_PHASE


def test_exactly_one_cancelled_reaches_the_person(conv_runtime):
    """The ``/cancel`` action answers; the interrupted turn stays quiet."""
    entered = threading.Event()
    release = threading.Event()
    results = []

    class _Blocking(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            if on_call is not None:
                on_call(release.set)
            entered.set()
            release.wait(timeout=10)
            raise RuntimeError("interrupted")

    rt, session, _ = conv_runtime()
    rt.services["llm"] = _Blocking()

    turn = threading.Thread(
        target=lambda: results.append(
            rt.handle_action(session.key, "send_text", "go")),
        daemon=True)
    turn.start()
    assert entered.wait(timeout=10)

    cancel_out = rt.handle_action(session.key, "cancel")
    turn.join(timeout=10)

    said = cancel_out.messages + [m for out in results for m in out.messages]
    assert said.count("Cancelled.") == 1


def test_a_normal_turn_still_says_what_it_always_said(conv_runtime):
    """The guards are keyed on cancellation and must not leak into the
    ordinary paths — an uncancelled turn with no reply still explains itself.

    Two empty responses, because the loop's empty-response nudge retries once
    before it accepts one.
    """
    rt, session, _ = conv_runtime([response(content=""), response(content="")])

    out = rt.handle_action(session.key, "send_text", "go")

    assert out.messages == ["(The agent ended its turn without a reply.)"]


# ── Telling the model it was cancelled ───────────────────────────────

def test_the_next_turn_is_told_the_last_one_was_cancelled(conv_runtime):
    """A cancelled turn used to leave no trace in the transcript at all.

    The last rows are the agent's own tool calls; the next user message
    simply follows. Nothing says the turn was stopped, so the model reads
    its own plan and carries on executing it.
    """
    rt, session, _ = conv_runtime()
    rt.handle_action(session.key, "cancel")

    assert session.pending_user_messages == []

    session.busy = True
    rt.handle_action(session.key, "cancel")

    assert len(session.pending_user_messages) == 1
    notice = session.pending_user_messages[0]
    assert "cancelled your previous turn" in notice
    assert "did not finish" in notice


def test_the_notice_survives_the_queue_being_cleared(conv_runtime):
    """Cancel drops queued user messages, and the notice is queued after.

    Reversed, the notice would be swallowed by the very clear that makes
    room for it — and the failure is silent, because an empty queue is
    exactly what an uncancelled session has.
    """
    rt, session, _ = conv_runtime()
    session.busy = True
    session.pending_user_messages.extend(["stale one", "stale two"])

    rt.handle_action(session.key, "cancel")

    assert len(session.pending_user_messages) == 1
    assert "stale" not in session.pending_user_messages[0]


def test_stopped_subagents_are_named_as_producing_nothing(conv_runtime):
    """The reported symptom: "I'll wait for the four subagents to finish",
    said about agents cancelled minutes earlier. Saying the turn stopped is
    not enough — it leaves "but the background work might still land" open."""
    rt, session, _ = conv_runtime()
    session.busy = True
    rt.subagents.cancel_for = lambda owner: 4

    rt.handle_action(session.key, "cancel")

    notice = session.pending_user_messages[0]
    assert "none are coming" in notice
    assert "offer to wait for them" in notice
    assert "4" not in notice, "a count the model would repeat must be right"


def test_a_turn_with_no_subagents_says_nothing_about_them(conv_runtime):
    rt, session, _ = conv_runtime()
    session.busy = True

    rt.handle_action(session.key, "cancel")

    assert "background agents" not in session.pending_user_messages[0]


def test_the_notice_reaches_history_on_the_next_turn(conv_runtime):
    """End to end: queued as an agent-facing message, drained into the
    transcript at the next turn's first loop boundary, and therefore in
    front of the model before it says anything."""
    rt, session, _ = conv_runtime([response(content="understood")])
    session.busy = True
    rt.handle_action(session.key, "cancel")
    session.busy = False

    rt.handle_action(session.key, "send_text", "different question")

    prompts = "\n".join(
        str(m.get("content") or "")
        for m in rt.services["llm"].calls[0])
    assert "cancelled your previous turn" in prompts


# ── The sandbox half ─────────────────────────────────────────────────

def test_interrupt_session_cancels_only_that_session_s_runs(sandbox_box):
    """``bridge._root_for`` roots an agent tool call at its session key, so
    ``chain_session`` is an exact filter rather than a guess."""
    from sandbox.policy import Chain

    class _Run:
        def __init__(self, root):
            self.chain = Chain(root=root)
            self.done = False
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def wait(self, timeout=None):
            """The fixture's shutdown drains ``_runs``; answer it."""
            return None

    mine, theirs, finished = _Run("s"), _Run("other"), _Run("s")
    finished.done = True
    sandbox_box._runs.extend([mine, theirs, finished])

    assert sandbox_box.interrupt_session("s") == 1
    assert mine.cancelled is True
    assert theirs.cancelled is False
    assert finished.cancelled is False


def test_interrupt_session_ignores_an_empty_key(sandbox_box):
    """A chain rooted at ``user`` names no session, and every root that is not
    a session answers "" — cancelling on that would cancel the world."""
    from sandbox.policy import Chain

    class _Run:
        chain = Chain(root="user")
        done = False
        cancelled = False

        def cancel(self):
            type(self).cancelled = True

        def wait(self, timeout=None):
            return None

    sandbox_box._runs.append(_Run())
    assert sandbox_box.interrupt_session("") == 0
    assert _Run.cancelled is False
