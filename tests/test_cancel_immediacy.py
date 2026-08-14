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

    loop._tool_finished(("edit_file", "t1", None), result=None, error="killed")
    assert seen == []

    session.cancel_event.clear()
    loop._tool_finished(("edit_file", "t1", None), result=None, error="killed")
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

    # State, not prose: the acknowledgement goes out as a notification (see
    # ``test_exactly_one_cancelled_reaches_the_person``), so what the action
    # itself answers with is whether it cancelled anything.
    assert out.data["cancelled"] is True
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
    """The ``/cancel`` action answers; the interrupted turn stays quiet.

    Counted across the notification *and* both text channels, because a person
    sees all three — a frontend that declares neither opt-in flattens the other
    two into the chat. Watching one would pass just as well if the duplicate
    moved to another, which is exactly what happened when the acknowledgement
    changed channels.
    """
    from events.event_bus import bus
    from events.event_channels import NOTIFICATION_PUSHED

    entered = threading.Event()
    release = threading.Event()
    results = []
    notices = []

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

    unsub = bus.subscribe(NOTIFICATION_PUSHED, notices.append)
    try:
        cancel_out = rt.handle_action(session.key, "cancel")
        turn.join(timeout=10)
    finally:
        unsub()

    def _said(out):
        return list(out.messages) + list(out.callable_output)

    said = _said(cancel_out) + [line for out in results for line in _said(out)]
    cancelled = ([n for n in notices if n.get("title") == "Cancelled"]
                 + [line for line in said if line.startswith("Cancelled")])
    assert len(cancelled) == 1
    # And it is the notification, not prose on a text channel.
    assert said == []


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

def test_a_cancelled_turn_leaves_a_row_saying_so(conv_runtime):
    """A cancelled turn used to leave no trace in the transcript at all.

    The last rows are the agent's own tool calls — five successful
    ``spawn_subagent`` results, say — and the next user message simply
    follows. The model reads its own plan back, sees no evidence anything
    ended, and offers to wait for results cancelled minutes ago.
    """
    class _Late(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            session.cancel_event.set()
            return response(content="ignored")

    session = _session()
    loop, session, _ = _cancelled_loop(_Late(), session)
    history = [{"role": "user", "content": "hi"}]

    loop.drive(session.cs, "agent", history)

    assert history[-1]["role"] == "user"
    assert "cancelled the previous turn" in history[-1]["content"]
    assert "nothing from it is still coming" in history[-1]["content"]


def test_an_uncancelled_turn_leaves_no_such_row(conv_runtime):
    session = _session()
    loop, session, _ = _cancelled_loop(FakeLLM([response(content="done")]), session)
    history = [{"role": "user", "content": "hi"}]

    loop.drive(session.cs, "agent", history)

    assert not any("cancelled the previous turn" in str(m.get("content") or "")
                   for m in history)


def test_the_notice_does_not_start_a_new_turn(conv_runtime):
    """The bug the first attempt shipped, and the reason this row is written
    by the loop rather than queued on ``pending_user_messages``.

    That list is a *drive trigger*: ``handle_action``'s closing-race check
    pops it and dispatches it as a fresh ``send_text``. So the notice started
    a whole new agent turn — one that was no longer cancelled, since the flag
    is cleared on the way out, and which happily re-ran the searches the
    person had just stopped.
    """
    entered = threading.Event()
    release = threading.Event()
    calls = []

    class _Blocking(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            calls.append(1)
            if len(calls) > 1:
                raise AssertionError(
                    "the cancel notice drove a second turn")
            if on_call is not None:
                on_call(release.set)
            entered.set()
            release.wait(timeout=10)
            raise RuntimeError("interrupted")

    rt, session, _ = conv_runtime()
    rt.services["llm"] = _Blocking()

    turn = threading.Thread(
        target=rt.handle_action, args=(session.key, "send_text", "go"),
        daemon=True)
    turn.start()
    assert entered.wait(timeout=10)

    rt.handle_action(session.key, "cancel")
    turn.join(timeout=10)

    assert not turn.is_alive()
    assert len(calls) == 1
    assert session.pending_user_messages == []


def test_the_notice_is_in_front_of_the_model_on_the_next_turn(conv_runtime):
    """End to end: recorded into history by the cancelled turn, so the next
    call carries it whether or not anything drained a queue."""
    class _Late(FakeLLM):
        def chat(self, request, on_delta=None, on_call=None):
            session.cancel_event.set()
            return response(content="ignored")

    rt, session, _ = conv_runtime()
    rt.services["llm"] = _Late()
    rt.handle_action(session.key, "send_text", "first question")

    second = FakeLLM([response(content="understood")])
    rt.services["llm"] = second
    rt.handle_action(session.key, "send_text", "different question")

    prompts = "\n".join(str(m.get("content") or "") for m in second.calls[0])
    assert "cancelled the previous turn" in prompts


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


def test_a_nested_run_is_still_matched_to_the_session_that_caused_it(sandbox_box):
    """What makes ``/cancel`` reach a *script*, and it is one line elsewhere.

    A script is not started by the session; it is started by a tool that was.
    ``Chain.push`` preserves ``root``, so the script's own Run carries the
    session key however deep it sits — which is why nothing had to be plumbed
    for scripts at all. It also covers a ``wait=False`` script, where
    ``_script_run``'s ``caller.abandoned`` poll never runs because nobody is
    waiting.
    """
    from sandbox.policy import Chain

    class _Run:
        def __init__(self, chain):
            self.chain = chain
            self.done = False
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def wait(self, timeout=None):
            return None

    deep = _Run(Chain(root="s").push("tool_something").push("some_script"))
    sandbox_box._runs.append(deep)

    assert sandbox_box.interrupt_session("s") == 1
    assert deep.cancelled is True


def test_cancelling_a_child_interrupts_the_work_it_was_doing():
    """The only route to a *subagent's* in-flight work, and it is one hop.

    A child's session key is ``spawn_subagent:<cid>``, so the parent's own
    ``interrupt_session`` cannot match anything the child started — a script
    the child is running is rooted at the child's key. Nothing reaches it
    except ``SubagentRegistry.cancel`` stepping into the child's session, and
    that hop depends on ``runtime.sessions`` still holding it. Missing, the
    flag would be set on nothing and the child's script would run on to its
    own ceiling with nobody waiting for it.
    """
    import threading as _threading

    from runtime.subagents import RUNNING, SESSION_PREFIX, Handle, SubagentRegistry

    child = _session(f"{SESSION_PREFIX}7")
    interrupted = []

    class _Runtime:
        sessions = {child.key: child}

        def _interrupt_work(self, session):
            interrupted.append(session.key)
            return 0

    registry = SubagentRegistry.__new__(SubagentRegistry)
    registry._lock = _threading.RLock()
    registry._handles = {}
    registry.runtime = _Runtime()
    registry._handles["h"] = Handle(
        id="h", conversation_id=7, owner="repl", title="c", timeout=30,
        state=RUNNING)

    assert registry.cancel("h") is True
    assert interrupted == [child.key], "the child's own work was left running"
    assert child.cancel_event.is_set()


def test_cancelling_a_child_whose_session_is_gone_does_not_raise():
    """Fail quiet, not loud: a cancel that raises leaves the rest of the
    children running."""
    import threading as _threading

    from runtime.subagents import RUNNING, Handle, SubagentRegistry

    class _Runtime:
        sessions = {}

    registry = SubagentRegistry.__new__(SubagentRegistry)
    registry._lock = _threading.RLock()
    registry._handles = {}
    registry.runtime = _Runtime()
    registry._handles["h"] = Handle(
        id="h", conversation_id=7, owner="repl", title="c", timeout=30,
        state=RUNNING)

    assert registry.cancel("h") is True


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
