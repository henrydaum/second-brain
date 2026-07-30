"""The subagent registry: the guards, the deadline, and the barrier.

Subagents are the part of the kernel where a mistake is quiet. A dropped
report reads as an agent that had nothing to say; a double-delivered one reads
as a model repeating itself; a stale success after a timeout reads as findings
nobody produced. None of those raise, so they are pinned here instead.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from runtime.subagents import (CANCELLED, DONE, FAILED, RUNNING,
                               SESSION_PREFIX, SubagentRegistry)


# ──────────────────────────────────────────────────────────────────────
# A runtime just real enough to spawn against.
# ──────────────────────────────────────────────────────────────────────

class FakeSession:
    """What the barrier and the cancel path actually touch on a session."""

    def __init__(self, key: str, conversation_id: int | None = None):
        self.key = key
        self.conversation_id = conversation_id
        self.busy = False
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.pending_user_messages: list[str] = []


class FakeDB:
    def __init__(self):
        self.conversations: dict[int, dict] = {}
        self.messages: list[tuple] = []

    def get_conversation(self, cid):
        return self.conversations.get(int(cid))

    def save_message(self, cid, role, content):
        self.messages.append((cid, role, content))


class FakeRuntime:
    """Drives whatever ``turn`` returns, and records what was asked of it."""

    def __init__(self, turn=None, config=None):
        self.db = FakeDB()
        self.config = config or {"max_concurrent_subagents": 4,
                                 "subagent_timeout_seconds": 300}
        self.services = {}
        self.sessions: dict[str, FakeSession] = {}
        self.active_conversation_id = None
        self.active_session_key = "repl"
        self._next_cid = 100
        self._turn = turn or (lambda key, prompt, **kw: SimpleNamespace(
            ok=True, messages=[f"did: {prompt}"], error=None))
        self.pushed: list[str] = []
        self.opened: list[str] = []
        self.closed: list[str] = []

    def create_conversation(self, title, *, kind="user", category=None,
                            user_id=1):
        self._next_cid += 1
        cid = self._next_cid
        self.db.conversations[cid] = {"id": cid, "title": title,
                                      "category": category}
        return cid

    def open_session(self, key, *, conversation_id=None, **kw):
        self.opened.append(key)
        session = self.sessions.setdefault(key, FakeSession(key))
        session.conversation_id = conversation_id
        return session

    def iterate_agent_turn(self, key, prompt, *, attachments=None, **kw):
        return self._turn(key, prompt, attachments=attachments)

    def close_session(self, key):
        self.closed.append(key)
        return True

    def push_message(self, key, text, **kw):
        self.pushed.append(text)


def registry_for(turn=None, config=None):
    """A registry over a fake runtime, ready to spawn."""
    runtime = FakeRuntime(turn=turn, config=config)
    return SubagentRegistry(runtime, runtime.config), runtime


def settle(registry, handle, timeout=5.0):
    """Block until a child reaches a terminal state, or fail the test."""
    assert handle._done.wait(timeout), f"subagent {handle.id} never finished"
    return registry.get(handle.id)


def eventually(predicate, timeout=5.0):
    """Wait for something that happens just after a handle settles.

    ``_done`` is set *before* the notice for an uncollectable failure goes
    out, so a waiter is released while that push is still in flight on the
    worker thread. Asserting straight after ``settle`` therefore passes when
    the suite is idle and loses the race under load — which is what made this
    look like an order-dependent test rather than a racing one.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# ──────────────────────────────────────────────────────────────────────
# The guards. Each one is a way a spawn can go badly wrong quietly.
# ──────────────────────────────────────────────────────────────────────

def test_a_subagent_cannot_spawn_a_subagent_at_the_default_depth():
    """max_subagent_depth is 1: a turn may spawn, a child may not."""
    registry, _ = registry_for()
    with pytest.raises(PermissionError, match="nest more than 1"):
        registry.spawn("go", owner=f"{SESSION_PREFIX}42")


def test_an_unidentifiable_spawner_is_treated_as_a_child_not_a_turn():
    """Fail closed. A subagent session whose handle the registry no longer
    holds — collected, or a restart under it — must not read as depth 0,
    which would buy unlimited nesting for the price of being forgotten."""
    registry, _ = registry_for(config={"max_subagent_depth": 2})
    # No handle exists for conversation 999, but the key says it is a child.
    child = registry.spawn("go", owner=f"{SESSION_PREFIX}999")
    settle(registry, child)
    assert registry.get(child.id).depth == 1

    registry2, _ = registry_for(config={"max_subagent_depth": 1})
    with pytest.raises(PermissionError):
        registry2.spawn("go", owner=f"{SESSION_PREFIX}999")


def test_depth_is_counted_down_the_lineage_when_nesting_is_allowed():
    registry, runtime = registry_for(config={"max_subagent_depth": 3,
                                             "max_concurrent_subagents": 4})
    parent = registry.spawn("parent work", owner="repl")
    # A grandchild spawned from inside the parent's own session.
    child = registry.spawn("child work",
                           owner=f"{SESSION_PREFIX}{parent.conversation_id}")
    assert (parent.depth, child.depth) == (0, 1)
    assert child.parent == parent.id


def test_cancelling_a_parent_takes_its_children_with_it():
    """A child of a stopped agent is work for a parent that is gone."""
    release = threading.Event()

    def turn(key, prompt, **kw):
        release.wait(5)
        return SimpleNamespace(ok=True, messages=["late"], error=None)

    registry, _ = registry_for(turn=turn, config={"max_subagent_depth": 3})
    parent = registry.spawn("parent", owner="repl")
    child = registry.spawn("child",
                           owner=f"{SESSION_PREFIX}{parent.conversation_id}")
    try:
        assert registry.cancel(parent.id) is True
        assert registry.get(child.id).state == CANCELLED
    finally:
        release.set()


def test_cancel_for_stops_everything_one_session_started():
    """What /cancel reaches for."""
    release = threading.Event()

    def turn(key, prompt, **kw):
        release.wait(5)
        return SimpleNamespace(ok=True, messages=["late"], error=None)

    registry, _ = registry_for(turn=turn)
    mine = [registry.spawn(f"job {i}", owner="repl") for i in range(3)]
    theirs = registry.spawn("other", owner="telegram")
    try:
        assert registry.cancel_for("repl") == 3
        assert all(registry.get(h.id).state == CANCELLED for h in mine)
        assert registry.get(theirs.id).state == RUNNING
    finally:
        release.set()


def test_a_queued_child_does_not_burn_its_deadline_while_it_waits():
    """A fan-out wider than the pool queues, and a deadline running in the
    queue cancels the tail before it has said a word."""
    started = threading.Event()
    release = threading.Event()

    def turn(key, prompt, **kw):
        started.set()
        release.wait(5)
        return SimpleNamespace(ok=True, messages=["done"], error=None)

    # One worker: the second spawn cannot start until the first finishes.
    registry, _ = registry_for(turn=turn,
                               config={"max_concurrent_subagents": 1,
                                       "subagent_timeout_seconds": 2})
    first = registry.spawn("first", owner="repl")
    queued = registry.spawn("queued", owner="repl")
    try:
        assert started.wait(5)
        # Its deadline has not started, so it cannot be overdue however long
        # the queue is.
        assert queued.deadline == float("inf")
        time.sleep(0.5)
        assert registry.get(queued.id).state == RUNNING
    finally:
        release.set()
    settle(registry, first)
    settle(registry, queued)
    assert registry.get(queued.id).state == DONE


def test_a_subagent_refuses_the_conversation_the_user_is_looking_at():
    """Two drivers on one conversation is a scrambled transcript."""
    registry, runtime = registry_for()
    cid = runtime.create_conversation("existing")
    runtime.active_conversation_id = cid
    with pytest.raises(PermissionError, match="active conversation"):
        registry.spawn("go", conversation_id=cid)


def test_one_conversation_runs_one_child_at_a_time():
    registry, runtime = registry_for()
    cid = runtime.create_conversation("busy one")
    session = FakeSession(f"{SESSION_PREFIX}{cid}", cid)
    session.busy = True
    runtime.sessions[session.key] = session
    with pytest.raises(PermissionError, match="already running"):
        registry.spawn("go", conversation_id=cid)


def test_an_empty_prompt_is_refused_before_a_conversation_is_opened():
    registry, runtime = registry_for()
    with pytest.raises(ValueError):
        registry.spawn("   ")
    assert not runtime.db.conversations


def test_a_missing_attachment_is_refused_while_there_is_a_caller_to_tell():
    registry, _ = registry_for()
    with pytest.raises(FileNotFoundError):
        registry.spawn("go", attachments=["nowhere/at/all.txt"])


# ──────────────────────────────────────────────────────────────────────
# Running one.
# ──────────────────────────────────────────────────────────────────────

def test_a_child_runs_its_turn_and_reports_back():
    registry, runtime = registry_for()
    handle = registry.spawn("summarise the docs", title="Docs")
    settle(registry, handle)

    report = registry.collect([handle.id])[0]
    assert report["state"] == DONE and report["ok"]
    assert "summarise the docs" in report["text"]
    assert report["title"] == "Docs"
    # The session is opened and closed around the turn, in that order — a
    # child that leaks its session blocks the next spawn into that
    # conversation forever.
    key = f"{SESSION_PREFIX}{handle.conversation_id}"
    assert runtime.opened == [key] and runtime.closed == [key]


def test_a_reporting_child_is_told_its_final_message_is_the_deliverable():
    """Without the framing a background agent writes as if someone will reply."""
    seen = {}

    def turn(key, prompt, **kw):
        seen["prompt"] = prompt
        return SimpleNamespace(ok=True, messages=["done"], error=None)

    registry, _ = registry_for(turn=turn)
    settle(registry, registry.spawn("find the bug", owner="repl"))
    assert "background agent" in seen["prompt"]


def test_a_child_with_nobody_waiting_is_not_given_the_framing():
    seen = {}

    def turn(key, prompt, **kw):
        seen["prompt"] = prompt
        return SimpleNamespace(ok=True, messages=["done"], error=None)

    registry, _ = registry_for(turn=turn)
    settle(registry, registry.spawn("find the bug"))
    assert "background agent" not in seen["prompt"]


def test_a_failing_turn_reports_the_failure_rather_than_an_empty_success():
    def turn(key, prompt, **kw):
        return SimpleNamespace(ok=False, messages=[],
                               error={"message": "the model refused"})

    registry, _ = registry_for(turn=turn)
    handle = settle(registry, registry.spawn("go"))
    assert handle.state == FAILED and handle.error == "the model refused"


def test_a_raising_turn_is_caught_and_carried_on_the_handle():
    def turn(key, prompt, **kw):
        raise RuntimeError("the box died")

    registry, runtime = registry_for(turn=turn)
    handle = settle(registry, registry.spawn("go"))
    assert handle.state == FAILED and "the box died" in handle.error
    # Still closed: the session must not leak past a crashing turn.
    assert runtime.closed


def test_a_scheduled_failure_is_pushed_because_nobody_will_collect_it():
    def turn(key, prompt, **kw):
        raise RuntimeError("nightly broke")

    registry, runtime = registry_for(turn=turn)
    settle(registry, registry.spawn("go", title="Nightly"))  # no owner
    assert eventually(
        lambda: any("Nightly" in message for message in runtime.pushed))


# ──────────────────────────────────────────────────────────────────────
# Collecting.
# ──────────────────────────────────────────────────────────────────────

def test_collect_with_no_ids_takes_everything_this_owner_started():
    registry, _ = registry_for()
    handles = [registry.spawn(f"job {i}", owner="repl") for i in range(3)]
    for handle in handles:
        settle(registry, handle)

    reports = registry.collect(owner="repl")
    assert len(reports) == 3
    assert all(r["state"] == DONE for r in reports)
    # Taken once. A second call has nothing left, which is what stops the
    # barrier re-delivering what the agent already read.
    assert registry.collect(owner="repl") == []


def test_collect_does_not_take_another_owners_children():
    registry, _ = registry_for()
    mine = registry.spawn("mine", owner="repl")
    theirs = registry.spawn("theirs", owner="telegram")
    settle(registry, mine)
    settle(registry, theirs)

    reports = registry.collect(owner="repl")
    assert [r["id"] for r in reports] == [mine.id]


def test_polling_with_timeout_zero_leaves_a_running_child_uncollected():
    """A poll must not consume a report that has not been produced yet."""
    release = threading.Event()

    def turn(key, prompt, **kw):
        release.wait(5)
        return SimpleNamespace(ok=True, messages=["late"], error=None)

    registry, _ = registry_for(turn=turn)
    handle = registry.spawn("slow", owner="repl")
    try:
        polled = registry.collect(owner="repl", timeout=0)
        assert polled[0]["state"] == RUNNING
        assert registry.pending_for("repl"), "a poll must not collect"
    finally:
        release.set()
    settle(registry, handle)
    assert registry.collect(owner="repl")[0]["text"] == "late"


def test_a_child_past_its_deadline_is_cancelled_rather_than_waited_on():
    """A deadline is a hard cutoff: no result, and the model is told so."""
    release = threading.Event()

    def turn(key, prompt, **kw):
        release.wait(5)
        return SimpleNamespace(ok=True, messages=["too late"], error=None)

    registry, _ = registry_for(turn=turn)
    handle = registry.spawn("slow", owner="repl", timeout_seconds=1)
    try:
        report = registry.collect([handle.id])[0]
        assert report["state"] == CANCELLED and not report["ok"]
        assert not report["text"]
        notice = registry.get(handle.id).notice()
        assert "TIMED OUT" in notice
        assert "do not report anything on its behalf" in notice
    finally:
        release.set()


def test_a_cancelled_child_never_reports_a_late_success():
    """The failure the model already saw must not be contradicted."""
    release = threading.Event()

    def turn(key, prompt, **kw):
        release.wait(5)
        return SimpleNamespace(ok=True, messages=["finished anyway"],
                               error=None)

    registry, _ = registry_for(turn=turn)
    handle = registry.spawn("slow", owner="repl")
    registry.cancel(handle.id)
    release.set()
    time.sleep(0.3)  # give the worker time to try to settle it

    assert registry.get(handle.id).state == CANCELLED
    assert "finished anyway" not in registry.get(handle.id).text


def test_cancelling_before_the_turn_starts_stops_it_starting():
    """A child cancelled while queued must not run work already reported failed."""
    started = []

    def turn(key, prompt, **kw):
        started.append(key)
        return SimpleNamespace(ok=True, messages=["ran"], error=None)

    # One worker, occupied, so the second spawn is still queued.
    registry, _ = registry_for(config={"max_concurrent_subagents": 1})
    hold = threading.Event()

    def occupy(key, prompt, **kw):
        hold.wait(5)
        return SimpleNamespace(ok=True, messages=["first"], error=None)

    registry.runtime._turn = occupy
    first = registry.spawn("first", owner="repl")
    registry.runtime._turn = turn
    second = registry.spawn("second", owner="repl")

    registry.cancel(second.id)
    hold.set()
    settle(registry, first)
    time.sleep(0.3)

    assert started == [], "a cancelled child must not start"
    assert registry.get(second.id).state == CANCELLED


# ──────────────────────────────────────────────────────────────────────
# The barrier.
# ──────────────────────────────────────────────────────────────────────

def test_the_barrier_queues_uncollected_reports_and_asks_for_a_redrive():
    registry, runtime = registry_for()
    session = FakeSession("repl", 7)
    runtime.sessions["repl"] = session
    for i in range(2):
        settle(registry, registry.spawn(f"job {i}", owner="repl",
                                        owner_conversation_id=7))

    assert registry.barrier(session) is True
    assert len(session.pending_user_messages) == 2
    assert all("Background agent" in m for m in session.pending_user_messages)


def test_the_barrier_abstains_on_the_redriven_half():
    """Firing twice would hold every turn open forever."""
    registry, runtime = registry_for()
    session = FakeSession("repl", 7)
    settle(registry, registry.spawn("job", owner="repl",
                                    owner_conversation_id=7))

    assert registry.barrier(session) is True
    session.pending_user_messages.clear()
    assert registry.barrier(session) is False
    assert session.pending_user_messages == []


def test_the_barrier_does_not_re_deliver_what_was_collected_explicitly():
    """One delivery, decided by whoever collects first."""
    registry, _ = registry_for()
    session = FakeSession("repl", 7)
    settle(registry, registry.spawn("job", owner="repl",
                                    owner_conversation_id=7))

    assert registry.collect(owner="repl")[0]["state"] == DONE
    assert registry.barrier(session) is False
    assert session.pending_user_messages == []


def test_a_report_is_dropped_when_the_session_moved_conversations():
    """The result still lives in the child's own conversation; it does not
    get to leak into whatever the user opened next."""
    registry, _ = registry_for()
    session = FakeSession("repl", 7)
    settle(registry, registry.spawn("job", owner="repl",
                                    owner_conversation_id=7))
    session.conversation_id = 9  # the user switched away

    assert registry.barrier(session) is False
    assert session.pending_user_messages == []


def test_the_barrier_takes_the_children_with_it_when_the_user_cancels():
    release = threading.Event()

    def turn(key, prompt, **kw):
        release.wait(5)
        return SimpleNamespace(ok=True, messages=["late"], error=None)

    registry, _ = registry_for(turn=turn)
    session = FakeSession("repl", 7)
    handle = registry.spawn("slow", owner="repl", owner_conversation_id=7)
    session.cancel_event.set()
    try:
        assert registry.barrier(session) is False
        assert registry.get(handle.id).state == CANCELLED
        assert session.pending_user_messages == []
    finally:
        release.set()


def test_the_barrier_survives_a_broken_session():
    """A barrier that breaks a turn is worse than one that misses a report."""
    registry, _ = registry_for()
    settle(registry, registry.spawn("job", owner="repl"))

    class Hostile:
        key = "repl"
        conversation_id = None
        cancel_event = None

        @property
        def lock(self):
            raise RuntimeError("no lock here")

    assert registry.barrier(Hostile()) is False


# ──────────────────────────────────────────────────────────────────────
# The barrier, from inside a real drive.
#
# The bug these exist for: the barrier used to live only inside the two
# end_turn doorways, so a turn leaving by any other route walked out past its
# own children and their reports were never delivered. It reproduces only with
# a real ConversationLoop, because a hand-called barrier is by definition on
# the path that works.
# ──────────────────────────────────────────────────────────────────────

def _turn_harness(tmp_path, responses, turn=None):
    """A real runtime whose agent gets scripted responses."""
    from agent.tool_registry import ToolRegistry
    from pipeline.database import Database
    from plugins.BaseTool import BaseTool, ToolResult
    from runtime.conversation_runtime import ConversationRuntime

    def _resp(content="", tool_calls=None):
        return SimpleNamespace(content=content, tool_calls=tool_calls or [],
                               has_tool_calls=bool(tool_calls),
                               is_error=False, prompt_tokens=0, error=None)

    seen = []

    class LLM:
        context_size = 0
        model_name = "fake"
        loaded = True

        def chat_with_tools(self, messages, tools=None, attachments=None):
            text = "\n".join(str(m.get("content") or "") for m in messages)
            if "[Note: you are a background agent" in text:
                return _resp(content="CHILD REPORT")
            seen.append(text)
            kind, value = responses[min(len(seen) - 1, len(responses) - 1)]
            if kind == "spawn":
                import json
                return _resp(tool_calls=[{
                    "id": f"c{len(seen)}", "name": "spawner",
                    "arguments": json.dumps({"prompt": value})}])
            if kind == "boom":
                return _resp(tool_calls=[{"id": "b", "name": "boom",
                                          "arguments": "{}"}])
            return _resp(content=value)

    class Spawner(BaseTool):
        name = "spawner"
        description = "spawn"
        parameters = {"type": "object",
                      "properties": {"prompt": {"type": "string"}}}
        requires_services = []
        max_calls = 20

        def run(self, context, **kwargs):
            handle = context.runtime.subagents.spawn(
                kwargs.get("prompt") or "x", title=kwargs.get("prompt") or "x",
                owner=context.session_key,
                owner_conversation_id=(context.runtime.sessions
                                       .get(context.session_key)
                                       .conversation_id))
            return ToolResult(True, data={"id": handle.id},
                              llm_summary="spawned")

    db = Database(str(tmp_path / "turn.db"))
    cid = db.create_conversation("parent")
    registry = ToolRegistry(db, {}, {})
    registry.register(Spawner())
    rt = ConversationRuntime(
        db=db, services={"llm": LLM()}, tool_registry=registry,
        config={"max_concurrent_subagents": 4,
                "subagent_timeout_seconds": 30})
    registry.runtime = rt
    if turn is not None:
        rt.subagents._config = rt.config
    rt.load_conversation("repl", cid)
    rt.active_session_key = "repl"
    return rt, seen


def test_a_turn_with_a_failing_tool_call_still_collects_its_children(tmp_path):
    """A failed tool call is feedback, not a turn-ender, so this one does
    reach a doorway. Pinned anyway: it is the shape the live failure was
    reported in, and it must keep working whichever barrier catches it."""
    rt, seen = _turn_harness(tmp_path, [
        ("spawn", "do the thing"),
        ("boom", None),              # a tool that does not exist -> failure
        ("text", "PARENT FINAL"),
    ])
    try:
        rt.iterate_agent_turn("repl", "go")
        history = "\n".join(seen)
        assert "[Background agent" in history, (
            "the child's report never reached the model")
        assert not rt.subagents.pending_for("repl")
    finally:
        rt.subagents.stop()


def test_children_are_collected_even_when_the_loop_never_reaches_a_doorway(tmp_path):
    """The backstop, exercised on a drive that leaves by no doorway at all.

    ``drive`` is stubbed to return the way a failed action, a priority handoff
    or an exhausted iteration budget leaves it: without enacting end_turn and
    without consulting either end_turn doorway. Before the backstop in
    ``_drive_agent_turn`` this abandoned every pending child silently.
    """
    rt, seen = _turn_harness(tmp_path, [("text", "unused")])
    session = rt.sessions["repl"]
    handle = rt.subagents.spawn("do the thing", title="Orphan", owner="repl",
                                owner_conversation_id=session.conversation_id)
    settle(rt.subagents, handle)

    import runtime.runtime_config as cfg
    from runtime.session import RuntimeResult

    class SilentLoop:
        def drive(self, *a, **kw):
            return "", [], []          # no doorway, no end_turn

    original, cfg.build_loop = cfg.build_loop, lambda *a, **kw: SilentLoop()
    try:
        rt._drive_agent_turn(session, RuntimeResult())
    finally:
        cfg.build_loop = original
        rt.subagents.stop()

    assert any("Background agent" in m for m in session.pending_user_messages), (
        "a drive that reached no doorway abandoned its child")


def test_a_normal_turn_still_collects_its_children(tmp_path):
    """The path that always worked, kept working."""
    rt, seen = _turn_harness(tmp_path, [
        ("spawn", "do the thing"),
        ("text", "spawned, ending"),
        ("text", "PARENT FINAL"),
    ])
    try:
        rt.iterate_agent_turn("repl", "go")
        assert "[Background agent" in "\n".join(seen)
    finally:
        rt.subagents.stop()


# ──────────────────────────────────────────────────────────────────────
# Scheduled spawns: the bus half.
# ──────────────────────────────────────────────────────────────────────

def test_a_timekeeper_job_on_the_spawn_channel_starts_a_child():
    """The only route a scheduled agent can arrive by."""
    from events.event_bus import bus
    from events.event_channels import SUBAGENT_SPAWN

    seen = {}

    def turn(key, prompt, **kw):
        seen["prompt"] = prompt
        return SimpleNamespace(ok=True, messages=["briefed"], error=None)

    registry, runtime = registry_for(turn=turn)
    registry.start()
    try:
        bus.emit(SUBAGENT_SPAWN, {
            "prompt": "write the morning brief",
            "title": "Morning brief",
            "_timekeeper": {"job_name": "brief", "one_time": False},
        })
        deadline = time.time() + 5
        while not seen and time.time() < deadline:
            time.sleep(0.05)
    finally:
        registry.stop()

    assert seen.get("prompt") == "write the morning brief"
    # No report channel, so no framing and nothing to collect — a scheduled
    # child's delivery surface is the user-facing push from its own turn.
    assert "background agent" not in seen["prompt"]
    assert registry.pending_for("repl") == []
    cid, = runtime.db.conversations
    assert runtime.db.conversations[cid]["category"] == "Scheduled"


def test_a_one_time_job_files_its_conversation_separately():
    from events.event_bus import bus
    from events.event_channels import SUBAGENT_SPAWN

    registry, runtime = registry_for()
    registry.start()
    try:
        bus.emit(SUBAGENT_SPAWN, {
            "prompt": "remind me",
            "_timekeeper": {"job_name": "once", "one_time": True},
        })
        deadline = time.time() + 5
        while not runtime.db.conversations and time.time() < deadline:
            time.sleep(0.05)
    finally:
        registry.stop()

    cid, = runtime.db.conversations
    assert runtime.db.conversations[cid]["category"] == "Scheduled (one-time)"


def test_a_recurring_job_pins_its_conversation_so_one_transcript_accumulates():
    """Without this, every firing opens a new conversation and the job's
    history scatters across dozens of them."""
    from events.event_bus import bus
    from events.event_channels import SUBAGENT_SPAWN

    patched = {}

    class Keeper:
        def get_job(self, name):
            return {"payload": {"prompt": "brief"}}

        def update_job(self, name, patch):
            patched[name] = patch

    registry, runtime = registry_for()
    runtime.services["timekeeper"] = Keeper()
    registry.start()
    try:
        bus.emit(SUBAGENT_SPAWN, {
            "prompt": "brief",
            "_timekeeper": {"job_name": "nightly", "one_time": False},
        })
        deadline = time.time() + 5
        while "nightly" not in patched and time.time() < deadline:
            time.sleep(0.05)
    finally:
        registry.stop()

    cid, = runtime.db.conversations
    assert patched["nightly"]["payload"]["conversation_id"] == cid
    assert patched["nightly"]["payload"]["prompt"] == "brief"


def test_a_scheduled_spawn_that_cannot_start_tells_the_user():
    """Nobody is collecting, so a refusal that is not pushed is one nobody
    ever learns about."""
    from events.event_bus import bus
    from events.event_channels import SUBAGENT_SPAWN

    registry, runtime = registry_for()
    registry.start()
    try:
        bus.emit(SUBAGENT_SPAWN, {"prompt": "   "})  # refused: no prompt
        deadline = time.time() + 5
        while not runtime.pushed and time.time() < deadline:
            time.sleep(0.05)
    finally:
        registry.stop()

    assert runtime.pushed and "did not start" in runtime.pushed[0]


# ──────────────────────────────────────────────────────────────────────
# agent.schedule: the Request that writes the job.
# ──────────────────────────────────────────────────────────────────────

class RecordingKeeper:
    """Just enough Timekeeper to see what job definition was written."""

    def __init__(self):
        self.jobs = {}

    def create_job(self, name, job):
        self.jobs[name] = job
        return job


def _schedule(args):
    """Call the agent.schedule handler with a recording Timekeeper."""
    from sandbox.guest.requests import AGENT_SCHEDULE
    from sandbox.handlers import HANDLERS
    from tests.support import call_handler

    keeper = RecordingKeeper()
    ctx = SimpleNamespace(services={"timekeeper": keeper}, runtime=None)
    return call_handler(AGENT_SCHEDULE, ctx, args), keeper


def test_scheduling_writes_a_job_on_the_kernels_own_spawn_channel():
    """The channel and payload shape are the kernel's, which is the whole
    reason this is a Request rather than a hand-built cron.create."""
    from events.event_channels import SUBAGENT_SPAWN

    result, keeper = _schedule({"prompt": "morning brief", "cron": "0 9 * * *",
                                "title": "Brief"})
    assert result.ok
    job = keeper.jobs["brief"]
    assert job["channel"] == SUBAGENT_SPAWN
    assert job["cron"] == "0 9 * * *" and job["one_time"] is False
    assert job["run_at"] is None and job["enabled"] is True
    assert job["payload"] == {"title": "Brief", "prompt": "morning brief",
                              "attachments": []}


def test_a_one_time_schedule_resolves_the_cron_to_an_absolute_time():
    """The Timekeeper wants a run_at, not a cron, for a job that fires once."""
    result, keeper = _schedule({"prompt": "remind me", "cron": "0 9 * * *",
                                "one_time": True, "title": "Reminder"})
    assert result.ok
    job = keeper.jobs["reminder"]
    assert job["one_time"] is True and job["cron"] is None
    assert job["run_at"] and "T" in job["run_at"]


def test_scheduling_refuses_a_malformed_cron_rather_than_writing_it():
    result, keeper = _schedule({"prompt": "go", "cron": "not a cron"})
    assert not result.ok and "cron" in result.error
    assert keeper.jobs == {}


def test_scheduling_needs_both_a_prompt_and_a_cron():
    assert not _schedule({"cron": "0 9 * * *"})[0].ok
    assert not _schedule({"prompt": "go"})[0].ok


def test_the_documented_orchestration_script_actually_runs(tmp_path):
    """The SDK's fan-out example, executed exactly as written.

    The whole reason spawning is in the SDK is that an agent should reach for
    a script instead of authoring a tool — and it will only do that if the one
    example it is shown works. So the example is run rather than trusted, the
    same rule ``tests/test_sdk_docs.py`` applies to the rest of the document.
    """
    import re
    from pathlib import Path

    import sandbox  # noqa: F401  - installs the ``guest`` package alias
    from guest.loader import unload_box
    from sandbox import Sandbox

    doc = Path(__file__).resolve().parents[1] / "docs/SDK.md"
    source = next(
        block for block in re.findall(
            r"```python\n(.*?)```", doc.read_text(encoding="utf-8"), re.S)
        if "def main(sdk, questions)" in block)
    script = tmp_path / "research.py"
    script.write_text(source, encoding="utf-8")

    # One child answers, one overruns its deadline — the example has to handle
    # both, and a version that only handled success would still pass a
    # happy-path test.
    slow = threading.Event()

    def turn(key, prompt, **kw):
        if "never" in prompt:
            slow.wait(5)
        return SimpleNamespace(ok=True, messages=[f"answer to {prompt[-6:]}"],
                               error=None)

    # The example asks for 600s; the configured ceiling caps it, which is how
    # a one-second deadline is imposed on a five-second child from out here.
    registry, runtime = registry_for(
        turn=turn, config={"max_concurrent_subagents": 4,
                           "subagent_timeout_seconds": 1})
    runtime.subagents = registry
    ctx = SimpleNamespace(runtime=runtime, session_key=None, user_id=1,
                          services={}, config=runtime.config, db=runtime.db,
                          root_dir=str(tmp_path))

    box = Sandbox()
    try:
        result = box.run(script, "main",
                         kwargs={"questions": ["question one", "never ends"]},
                         context=ctx)
    finally:
        slow.set()
        unload_box("research")
        box.shutdown()

    assert result.ok, result.error
    assert "answer to" in result.data["report"]
    assert result.data["lost"] and "cancelled" in result.data["lost"][0]


def test_the_schedule_command_names_the_kernels_spawn_channel():
    """A command cannot import the kernel, so it spells the channel out.

    That leaves exactly one way for the two to drift, and this closes it.
    """
    from events.event_channels import SUBAGENT_SPAWN
    from sandbox.validator import validate_file

    source = validate_file("plugins/commands/command_schedule.py").source
    assert f'SUBAGENT_CHANNEL = "{SUBAGENT_SPAWN}"' in source


def test_a_long_report_says_how_much_was_cut():
    """A preview the model mistakes for the whole report is worse than an
    obvious truncation."""
    from runtime.subagents import NOTICE_CAP

    def turn(key, prompt, **kw):
        return SimpleNamespace(ok=True, messages=["x" * (NOTICE_CAP + 500)],
                               error=None)

    registry, _ = registry_for(turn=turn)
    handle = settle(registry, registry.spawn("go", owner="repl"))
    notice = handle.notice()
    assert "report truncated" in notice
    assert f"{NOTICE_CAP + 500:,} chars total" in notice
