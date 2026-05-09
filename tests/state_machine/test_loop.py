"""Smoke tests for the state-machine refactor.

Three tight tests covering the three real flows:
    1. Chat path        — user text → agent text → end turn
    2. Tool-call path   — user text → agent calls a tool → tool result → agent text → end turn
    3. Form path        — user runs `/cmd` → fills two text fields → command runs

These intentionally avoid the runtime + DB + frontend layers so the state
machine itself can be exercised in isolation. They're the equivalent of
running a single PokerMonster game with two scripted players.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.tool_registry import ToolRegistry
from plugins.BaseTool import BaseTool, ToolResult
from plugins.commands.command_conversations import ConversationsCommand, NewCommand
from plugins.commands.command_tools import ToolsCommand
from plugins.services.llmService import LLMResponse
from plugins.services import timekeeperService as timekeeper_module
from plugins.services.timekeeperService import TimekeeperService
from plugins.tools.tool_schedule_subagent import SCHEDULED, SCHEDULED_ONCE, ScheduleSubagent
from plugins.tasks.task_spawn_subagent import SpawnSubagent
from state_machine.action_map import create_action
from state_machine.conversation import CallableSpec, ConversationState, FormStep, Participant
from runtime.conversation_loop import ConversationLoop
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST
from state_machine.serialization import latest_state, save_state_marker
from runtime.conversation_runtime import ConversationRuntime
from events.event_channels import CHAT_MESSAGE_PUSHED, COMMAND_CALL_FINISHED, SESSION_CLOSED, SESSION_CREATED, SESSION_TURN_COMPLETED, TOOL_CALL_FINISHED, TOOL_CALL_STARTED
from events.event_bus import bus


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class FakeResponse(SimpleNamespace):
    """Mimics the response shape ConversationLoop expects from an LLM."""

    @classmethod
    def text(cls, content: str) -> "FakeResponse":
        return cls(content=content, has_tool_calls=False, tool_calls=[], is_error=False, prompt_tokens=0)

    @classmethod
    def tool(cls, calls: list[dict]) -> "FakeResponse":
        return cls(content=None, has_tool_calls=True, tool_calls=calls, is_error=False, prompt_tokens=0)


class FakeLLM:
    """Returns a scripted sequence of FakeResponse objects per call."""

    context_size = 0

    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls = 0
        self.seen = []

    def chat_with_tools(self, messages, tools, attachments=None):
        self.seen.append((messages, tools, attachments))
        if self.calls >= len(self.responses):
            raise AssertionError("FakeLLM ran out of scripted responses")
        r = self.responses[self.calls]
        self.calls += 1
        return r


class FakeToolResult(SimpleNamespace):
    """Matches the success/error/data/llm_summary shape ConversationLoop reads."""

    success: bool = True


class FakeToolRegistry:
    """Minimal stand-in for the real tool registry."""

    def __init__(self, schemas: list[dict] | None = None, tool_results: dict | None = None):
        self._schemas = schemas or []
        self._tool_results = tool_results or {}
        self.tools = {name: SimpleNamespace(max_calls=3) for name in self._tool_results}
        self.max_tool_calls = 5
        self.called: list[tuple[str, dict]] = []

    def get_all_schemas(self) -> list[dict]:
        return self._schemas

    def call(self, name: str, **kwargs):
        self.called.append((name, kwargs))
        return self._tool_results.get(name, FakeToolResult(success=True, data={"echo": kwargs}, llm_summary=f"ran {name}", attachment_paths=[]))


class FakeConversationDB:
    def __init__(self):
        self.conversations = {}
        self.messages = {}
        self.next_id = 1
        self.replaced_history = None

    def create_conversation(self, title="New conversation", kind="user", category=None):
        cid = self.next_id; self.next_id += 1
        self.conversations[cid] = {"id": cid, "title": title, "kind": kind, "category": category}
        self.messages[cid] = []
        return cid

    def get_conversation(self, conversation_id):
        return self.conversations.get(conversation_id)

    def save_message(self, conversation_id, role, content, tool_call_id=None, tool_name=None):
        self.messages.setdefault(conversation_id, []).append({"role": role, "content": content, "tool_call_id": tool_call_id, "tool_name": tool_name})

    def replace_conversation_messages(self, conversation_id, history):
        self.replaced_history = list(history)
        self.messages[conversation_id] = []
        for msg in history:
            self.save_message(conversation_id, msg.get("role"), msg.get("content") or "", msg.get("tool_call_id"), msg.get("name"))

    def get_conversation_messages(self, conversation_id):
        return list(self.messages.get(conversation_id, []))

    def get_system_stats(self):
        return {"files": {}, "tasks": {}}

    def list_conversations_page(self, offset=0, limit=10, category=None):
        rows = list(self.conversations.values())
        if category == "":
            rows = [r for r in rows if r.get("category") in (None, "")]
        elif category is not None:
            rows = [r for r in rows if r.get("category") == category]
        return rows[offset:offset + limit], len(rows) > offset + limit

    def list_conversation_categories(self):
        out = []
        for r in self.conversations.values():
            c = r.get("category")
            v = None if c in (None, "") else c
            if v not in out:
                out.append(v)
        return out

    def set_conversation_category(self, conversation_id, category):
        self.conversations[conversation_id]["category"] = category


class FakeTimekeeper:
    loaded = True

    def __init__(self, jobs=None):
        self.jobs = dict(jobs or {})

    def get_job(self, name):
        return self.jobs.get(name)

    def list_jobs(self):
        return dict(self.jobs)

    def create_job(self, name, job_def):
        if name in self.jobs:
            raise ValueError(f"Job '{name}' already exists.")
        self.jobs[name] = dict(job_def)
        return self.jobs[name]

    def update_job(self, name, patch):
        self.jobs[name].update(patch)
        return self.jobs[name]

    def remove_job(self, name):
        return self.jobs.pop(name, None) is not None

    def cron_to_text(self, expr):
        if expr == "bad cron":
            raise ValueError("Invalid cron expression")
        return expr


def make_cs(commands: dict[str, CallableSpec] | None = None, tools: dict[str, CallableSpec] | None = None) -> ConversationState:
    return ConversationState([
        Participant("user", "user", commands=commands or {}),
        Participant("agent", "agent", tools=tools or {}),
    ])


# ──────────────────────────────────────────────────────────────────────────
# 1. Chat path
# ──────────────────────────────────────────────────────────────────────────


def test_chat_path_user_text_then_agent_reply():
    cs = make_cs()
    # User sends text; SendText auto-hands priority to the agent.
    result = create_action(cs, "send_text", "hello", "user").enact()
    assert result.ok
    assert cs.turn_priority == "agent"
    assert cs.phase == "awaiting_input"

    llm = FakeLLM([FakeResponse.text("hi there")])
    registry = FakeToolRegistry()
    loop = ConversationLoop(llm, registry, {}, "")

    history: list[dict] = [{"role": "user", "content": "hello"}]
    final_text, new_messages, attachments = loop.drive(cs, "agent", history)

    assert final_text == "hi there"
    assert [m["role"] for m in new_messages] == ["assistant"]
    assert new_messages[0]["content"] == "hi there"
    assert attachments == []
    # EndTurn handed priority back to the user.
    assert cs.turn_priority == "user"
    assert cs.phase == "awaiting_input"


# ──────────────────────────────────────────────────────────────────────────
# 2. Tool-call path
# ──────────────────────────────────────────────────────────────────────────


def test_tool_call_path_records_assistant_then_tool_then_final_text():
    events = []
    schema = {"function": {"name": "echo", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
    registry = FakeToolRegistry(
        schemas=[schema],
        tool_results={"echo": FakeToolResult(success=True, data={"q": "x"}, llm_summary="echoed: x", attachment_paths=[])},
    )
    def tool_handler(_cs, _actor, args):
        assert events == [("start", "echo", "tc1", {"q": "x"})]
        return registry.call("echo", **args)

    # Build a CallableSpec for the tool so cs.spec("agent", "call_tool", "echo") succeeds.
    tool_spec = CallableSpec(
        name="echo",
        handler=tool_handler,
        form=[FormStep("q", required=True)],
    )
    cs = make_cs(tools={"echo": tool_spec})

    # Pretend the user already sent text and priority is the agent's.
    cs.set_priority("agent")

    llm = FakeLLM([
        FakeResponse.tool([{"id": "tc1", "name": "echo", "arguments": json.dumps({"q": "x"})}]),
        FakeResponse.text("done"),
    ])
    loop = ConversationLoop(
        llm, registry, {}, "",
        on_tool_start=lambda name, call_id, args: events.append(("start", name, call_id, args)),
        on_tool_result=lambda name, call_id, result, error: events.append(("finish", name, call_id, result.ok, error)),
    )

    history: list[dict] = [{"role": "user", "content": "do the thing"}]
    final_text, new_messages, attachments = loop.drive(cs, "agent", history)

    assert final_text == "done"
    roles = [m["role"] for m in new_messages]
    assert roles == ["assistant", "tool", "assistant"]
    assert new_messages[0].get("tool_calls"), "first assistant row carries the tool_calls"
    assert new_messages[1]["name"] == "echo"
    assert new_messages[1]["tool_call_id"] == "tc1"
    assert new_messages[2]["content"] == "done"
    assert events == [("start", "echo", "tc1", {"q": "x"}), ("finish", "echo", "tc1", True, None)]
    assert cs.turn_priority == "user"


def test_tool_budget_blocks_before_second_execution():
    schema = {"function": {"name": "echo", "parameters": {"type": "object", "properties": {}, "required": []}}}
    registry = FakeToolRegistry([schema], {"echo": FakeToolResult(success=True, data={}, llm_summary="ok", attachment_paths=[])})
    registry.tools["echo"].max_calls = 1
    cs = make_cs(tools={"echo": CallableSpec("echo", handler=lambda _cs, _actor, args: registry.call("echo", **args))})
    cs.set_priority("agent")
    llm = FakeLLM([
        FakeResponse.tool([{"id": "tc1", "name": "echo", "arguments": "{}"}]),
        FakeResponse.tool([{"id": "tc2", "name": "echo", "arguments": "{}"}]),
        FakeResponse.text("stopped"),
    ])

    final_text, new_messages, _ = ConversationLoop(llm, registry, {}, "").drive(cs, "agent", [{"role": "user", "content": "loop"}])

    assert final_text == "stopped"
    assert len(registry.called) == 1
    assert "call limit" in new_messages[3]["content"]


def test_runtime_emits_session_scoped_tool_status_events():
    events = []
    schema = {"function": {"name": "echo", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
    registry = FakeToolRegistry(
        schemas=[schema],
        tool_results={"echo": FakeToolResult(success=True, data={"q": "x"}, llm_summary="echoed: x", attachment_paths=[])},
    )
    runtime = ConversationRuntime(
        services={"llm": FakeLLM([
            FakeResponse.tool([{"id": "tc1", "name": "echo", "arguments": json.dumps({"q": "x"})}]),
            FakeResponse.text("done"),
        ])},
        tool_registry=registry,
        emit_event=lambda channel, payload: events.append((channel, payload)),
    )

    result = runtime.handle_action("chat-1", "send_text", "do it")

    assert result.messages[-1] == "done"
    assert [(c, p["session_key"], p["call_id"], p["tool_name"]) for c, p in events] == [
        (TOOL_CALL_STARTED, "chat-1", "tc1", "echo"),
        (TOOL_CALL_FINISHED, "chat-1", "tc1", "echo"),
    ]
    assert events[0][1]["args"] == {"q": "x"}
    assert events[1][1]["ok"] is True


def test_runtime_can_chat_without_tool_registry():
    llm = FakeLLM([FakeResponse.text("plain reply")])
    result = ConversationRuntime(services={"llm": llm}).handle_action("chat", "send_text", "hello")

    assert result.messages[-1] == "plain reply"
    assert llm.seen[0][1] is None


def test_runtime_surfaces_llm_provider_error_to_chat():
    err = "your current token plan not support model, MiniMax-M2.7 (2061)"
    llm = FakeLLM([LLMResponse(error=err, error_code="provider_error")])

    result = ConversationRuntime(services={"llm": llm}).handle_action("chat", "send_text", "hello")

    assert not result.ok
    assert err in result.messages[-1]
    assert result.error["code"] == "agent_failed"


def test_runtime_attachment_bundle_reaches_llm():
    import tempfile, os
    fd, note_path = tempfile.mkstemp(suffix=".txt")
    try:
        os.write(fd, b"parsed file text")
        os.close(fd)

        llm = FakeLLM([FakeResponse.text("saw it")])
        runtime = ConversationRuntime(services={"llm": llm}, tool_registry=FakeToolRegistry())

        result = runtime.handle_action("chat-attach", "send_attachment", {"path": note_path, "extension": "txt", "caption": "see this"})

        assert result.messages[-1] == "saw it"
        bundle = llm.seen[0][2]
        assert bundle is not None and len(bundle) == 1
        attachment = list(bundle)[0]
        assert attachment.parsed_text == "parsed file text"
        assert attachment.modality == "text"
        # History row carries the caption + pointer line, not the parsed body.
        user_msg = llm.seen[0][0][1]
        assert "see this" in user_msg["content"]
    finally:
        try:
            os.unlink(note_path)
        except OSError:
            pass


def test_iterate_agent_turn_loads_persists_and_emits_completion():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron", category="Scheduled")
    db.save_message(conv_id, "user", "earlier")
    events = []
    runtime = ConversationRuntime(
        db=db,
        services={"llm": FakeLLM([FakeResponse.text("done")])},
        tool_registry=FakeToolRegistry(),
        emit_event=lambda channel, payload: events.append((channel, payload)),
    )
    pushed = []
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, pushed.append)

    try:
        runtime.load_conversation("job", conv_id)
        result = runtime.iterate_agent_turn("job", "wake up")
    finally:
        unsub()

    rows = [r for r in db.get_conversation_messages(conv_id) if r["role"] != "system"]
    assert result.messages[-1] == "done"
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "earlier"),
        ("user", "wake up"),
        ("assistant", "done"),
    ]
    assert db.replaced_history[-1] == {"role": "assistant", "content": "done"}
    assert events[-1][0] == SESSION_TURN_COMPLETED
    assert events[-1][1]["session_key"] == "job"
    assert events[-1][1]["conversation_id"] == conv_id
    assert events[-1][1]["final_text"] == "done"
    assert pushed[-1]["message"] == "done\n\nLoad this conversation: `/conversations Scheduled 1 'Load conversation'`"
    assert pushed[-1]["source_session_key"] == "job"
    assert "session_key" not in pushed[-1]


def test_load_history_restores_saved_agent_profile_and_history():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Builder chat")
    db.save_message(conv_id, "user", "earlier")
    save_state_marker(db, conv_id, {"active_agent_profile": "builder"})
    events = []
    unsubs = [
        bus.subscribe(SESSION_CLOSED, lambda payload: events.append((SESSION_CLOSED, payload))),
        bus.subscribe(SESSION_CREATED, lambda payload: events.append((SESSION_CREATED, payload))),
    ]
    runtime = ConversationRuntime(db=db, config={"active_agent_profile": "default"})
    try:
        runtime.get_session("chat").history.append({"role": "user", "content": "old"})
        result = runtime.load_history("chat", conv_id)
    finally:
        for unsub in unsubs:
            unsub()

    session = runtime.sessions["chat"]
    from runtime.runtime_config import profile_for, refresh_specs
    refresh_specs(runtime, session)

    assert session.history == [{"role": "user", "content": "earlier"}]
    assert session.profile_override == "builder"
    assert profile_for(runtime, session) == "builder"
    assert result.messages == ["Loaded conversation: Builder chat\nAgent: builder\nSwitched agent: default -> builder"]
    assert [name for name, _ in events[-2:]] == [SESSION_CLOSED, SESSION_CREATED]
    assert events[-1][1]["agent_profile"] == "builder"


def test_set_conversation_notification_mode_updates_stored_marker():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    save_state_marker(db, conv_id, {"active_agent_profile": "builder"})
    runtime = ConversationRuntime(db=db)

    assert runtime.set_conversation_notification_mode(conv_id, "IMPORTANT") == "on"
    assert latest_state(db.get_conversation_messages(conv_id))["notification_mode"] == "on"
    assert runtime.load_conversation("job", conv_id).notification_mode == "on"


def test_set_conversation_notification_mode_updates_live_session():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    runtime = ConversationRuntime(db=db)
    session = runtime.load_conversation("job", conv_id)

    assert session.notification_mode == "on"
    assert not any(getattr(t, "name", None) == "notify" for t in session.extra_tool_instances)
    assert runtime.set_conversation_notification_mode(conv_id, "off") == "off"
    assert session.notification_mode == latest_state(db.get_conversation_messages(conv_id))["notification_mode"] == "off"
    assert not any(getattr(t, "name", None) == "notify" for t in session.extra_tool_instances)


def test_notification_off_suppresses_background_final_answer_push():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    save_state_marker(db, conv_id, {"notification_mode": "off"})
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("quiet")])}, tool_registry=FakeToolRegistry())
    pushed = []
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, pushed.append)
    try:
        runtime.load_conversation("job", conv_id)
        result = runtime.iterate_agent_turn("job", "wake up")
    finally:
        unsub()

    assert result.ok
    assert pushed == []


def test_spawn_subagent_drives_inactive_conversation():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    db.save_message(conv_id, "user", "earlier")
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("done")])}, tool_registry=FakeToolRegistry())
    result = SpawnSubagent().run_event("run", {"conversation_id": conv_id, "prompt": "wake up"}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))

    assert result.success
    assert runtime.sessions[f"spawn_subagent:{conv_id}"].conversation_id == conv_id
    assert [(r["role"], r["content"]) for r in db.get_conversation_messages(conv_id) if r["role"] != "system"] == [
        ("user", "earlier"), ("user", "wake up"), ("assistant", "done")
    ]


def test_spawn_subagent_creates_missing_conversation():
    db = FakeConversationDB()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("done")])}, tool_registry=FakeToolRegistry())

    result = SpawnSubagent().run_event("run", {"title": "Morning Brief", "prompt": "wake up"}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))

    assert result.success
    cid = result.data["conversation_id"]
    assert db.get_conversation(cid)["title"] == "Morning Brief"
    assert db.get_conversation(cid)["category"] == SCHEDULED
    assert latest_state(db.get_conversation_messages(cid))["profile_override"] == "default"


def test_spawn_subagent_repairs_deleted_conversation_and_updates_recurring_job():
    db = FakeConversationDB()
    tk = FakeTimekeeper({"brief": {"payload": {"title": "Brief", "prompt": "wake up", "conversation_id": 999}}})
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("done")])}, tool_registry=FakeToolRegistry())
    payload = {"title": "Brief", "prompt": "wake up", "conversation_id": 999, "_timekeeper": {"job_name": "brief", "one_time": False}}

    result = SpawnSubagent().run_event("run", payload, SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}))

    assert result.success
    assert result.data["conversation_id"] != 999
    assert tk.jobs["brief"]["payload"]["conversation_id"] == result.data["conversation_id"]


def test_spawn_subagent_rejects_active_conversation():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Main")
    runtime = ConversationRuntime(db=db)
    runtime.load_conversation("chat", conv_id)
    runtime.active_session_key = "chat"
    pushed = []
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, pushed.append)
    try:
        result = SpawnSubagent().run_event("run", {"conversation_id": conv_id, "prompt": "wake up"}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))
    finally:
        unsub()

    assert not result.success
    assert "active conversation" in result.error
    assert pushed and "active conversation" in pushed[-1]["message"]


def test_spawn_subagent_rejects_busy_background_session():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    runtime = ConversationRuntime(db=db)
    runtime.load_conversation(f"spawn_subagent:{conv_id}", conv_id).busy = True

    result = SpawnSubagent().run_event("run", {"conversation_id": conv_id, "prompt": "wake up"}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))

    assert not result.success
    assert "already running" in result.error


def test_spawn_subagent_parses_attachment_paths():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    path = Path(".codex_spawn_subagent_attachment.txt")
    path.write_text("hello from a file", encoding="utf-8")
    llm = FakeLLM([FakeResponse.text("done")])
    runtime = ConversationRuntime(db=db, services={"llm": llm}, tool_registry=FakeToolRegistry())
    try:
        result = SpawnSubagent().run_event("run", {"conversation_id": conv_id, "prompt": "read", "attachments": [str(path)]}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))
    finally:
        path.unlink(missing_ok=True)

    assert result.success
    bundle = llm.seen[0][2]
    assert bundle and next(iter(bundle)).path == str(path)


def test_spawn_subagent_validates_payload_and_attachment():
    db = FakeConversationDB()
    runtime = ConversationRuntime(db=db)
    task = SpawnSubagent()
    ctx = SimpleNamespace(db=db, runtime=runtime, services={}, config={})
    conv_id = db.create_conversation("Cron")

    assert "prompt" in task.run_event("run", {"conversation_id": conv_id}, ctx).error
    assert "Attachment not found" in task.run_event("run", {"conversation_id": conv_id, "prompt": "x", "attachments": [".missing_spawn_attachment.txt"]}, ctx).error


def test_schedule_subagent_recurring_cron_requires_approval_and_creates_job():
    db = FakeConversationDB()
    tk = FakeTimekeeper()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    approvals = []
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda command, text: approvals.append((command, text)) or True)

    result = ScheduleSubagent().run(ctx, action="add", title="Morning Brief", prompt="brief me", cron="0 8 * * *")

    job = tk.jobs["morning_brief"]
    assert result.success
    assert approvals and approvals[0][0] == "schedule_subagent"
    assert db.conversations == {}
    assert job["channel"] == "subagent.spawn"
    assert job["cron"] == "0 8 * * *"
    assert job["one_time"] is False
    assert job["payload"] == {"title": "Morning Brief", "prompt": "brief me", "attachments": []}


def test_tools_command_user_initiated_schedule_creates_cron_without_second_approval():
    db = FakeConversationDB()
    tk = FakeTimekeeper()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    registry = ToolRegistry(db, {"tool_timeout": 10}, {"timekeeper": tk})
    registry.runtime = runtime
    registry.register(ScheduleSubagent())

    result = ToolsCommand().run({
        "tool_name": "schedule_subagent",
        "action": "call",
        "operation": "add",
        "title": "Nightly Wisdom",
        "prompt": "send wisdom",
        "cron": "0 20 * * *",
        "one_time": False,
    }, SimpleNamespace(tool_registry=registry))

    assert result == "Done: Scheduled subagent 'Nightly Wisdom'."
    assert tk.jobs["nightly_wisdom"]["cron"] == "0 20 * * *"
    assert tk.jobs["nightly_wisdom"]["one_time"] is False


def test_schedule_subagent_one_time_cron_computes_run_at():
    db = FakeConversationDB()
    tk = FakeTimekeeper()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: True)

    result = ScheduleSubagent().run(ctx, operation="add", title="One Shot", prompt="do it", cron="0 8 * * *", one_time=True)

    job = tk.jobs["one_shot"]
    assert result.success
    assert db.conversations == {}
    assert job["one_time"] is True
    assert "run_at" in job and job["run_at"]
    assert job["cron"] is None


def test_schedule_subagent_stores_attachments_in_payload():
    db = FakeConversationDB()
    tk = FakeTimekeeper()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: True)

    result = ScheduleSubagent().run(ctx, operation="add", title="With Files", prompt="read", cron="0 8 * * *", attachments=["a.txt", "b.txt"])

    assert result.success
    assert tk.jobs["with_files"]["payload"]["attachments"] == ["a.txt", "b.txt"]


def test_schedule_subagent_list_edit_and_remove():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Morning Brief")
    tk = FakeTimekeeper({
        "morning_brief": {"enabled": True, "channel": "subagent.spawn", "cron": "0 8 * * *", "one_time": False, "payload": {"title": "Morning Brief", "prompt": "old", "attachments": [], "conversation_id": conv_id}},
        "update_titles": {"enabled": True, "channel": "update_titles", "cron": "*/30 * * * *", "one_time": False, "payload": {}},
    })
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: True)

    listed = ScheduleSubagent().run(ctx, operation="list")
    assert listed.success
    assert listed.data["jobs"] == [{"title": "Morning Brief", "cron": "0 8 * * *", "run_at": None, "one_time": False, "enabled": True, "attachments": [], "conversation_id": conv_id}]

    edited = ScheduleSubagent().run(ctx, operation="edit", title="Morning Brief", prompt="new", attachments=["a.txt"])
    assert edited.success
    assert tk.jobs["morning_brief"]["payload"] == {"title": "Morning Brief", "prompt": "new", "attachments": ["a.txt"], "conversation_id": conv_id}

    removed = ScheduleSubagent().run(ctx, operation="remove", title="Morning Brief")
    assert removed.success
    assert "morning_brief" not in tk.jobs
    assert db.get_conversation(conv_id) is not None


def test_schedule_subagent_resolves_literal_job_key_or_payload_title():
    db = FakeConversationDB()
    tk = FakeTimekeeper({"Tester": {"enabled": True, "channel": "subagent.spawn", "cron": "0 8 * * *", "one_time": False, "payload": {"title": "Tester", "prompt": "old", "attachments": []}}})
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: True)

    edited = ScheduleSubagent().run(ctx, operation="edit", title="Tester", prompt="new")
    assert edited.success
    assert tk.jobs["Tester"]["payload"]["prompt"] == "new"

    removed = ScheduleSubagent().run(ctx, operation="remove", title="Tester")
    assert removed.success
    assert "Tester" not in tk.jobs


def test_schedule_subagent_edit_schedule_shapes():
    db = FakeConversationDB()
    tk = FakeTimekeeper({"brief": {"enabled": True, "channel": "subagent.spawn", "cron": "0 8 * * *", "one_time": False, "payload": {"title": "Brief", "prompt": "x", "attachments": []}}})
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: True)

    result = ScheduleSubagent().run(ctx, operation="edit", title="Brief", cron="0 9 * * *", one_time=True)

    assert result.success
    assert tk.jobs["brief"]["one_time"] is True
    assert "run_at" in tk.jobs["brief"] and tk.jobs["brief"]["run_at"]
    assert "cron" not in tk.jobs["brief"] or tk.jobs["brief"]["cron"] is None


def test_timekeeper_one_time_jobs_auto_delete_after_emit(monkeypatch):
    saved = {}
    monkeypatch.setattr(timekeeper_module.config_manager, "load_plugin_config", lambda: {})
    monkeypatch.setattr(timekeeper_module.config_manager, "save_plugin_config", saved.update)
    run_at = timekeeper_module._now_local().isoformat()
    tk = TimekeeperService({"scheduled_jobs": {"once": {"channel": "test.once", "run_at": run_at, "one_time": True, "payload": {}}}})

    tk._emit_job("once", tk.get_job("once"), timekeeper_module._now_local())

    assert tk.get_job("once") is None
    assert "once" not in saved["scheduled_jobs"]


def test_schedule_subagent_denied_approval_has_no_side_effects():
    db = FakeConversationDB()
    tk = FakeTimekeeper()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: False)

    result = ScheduleSubagent().run(ctx, operation="add", title="Nope", prompt="do it", cron="0 8 * * *")

    assert not result.success
    assert db.conversations == {}
    assert tk.jobs == {}


def test_schedule_subagent_validation_and_duplicate_title():
    db = FakeConversationDB()
    tk = FakeTimekeeper({"morning_brief": {"enabled": True}})
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: True)
    tool = ScheduleSubagent()

    assert "action" in tool.run(ctx, title="x", prompt="y", cron="0 8 * * *").error
    assert "title" in tool.run(ctx, operation="add", title="", prompt="x", cron="0 8 * * *").error
    assert "prompt" in tool.run(ctx, operation="add", title="x", prompt="", cron="0 8 * * *").error
    assert "cron expression" in tool.run(ctx, operation="add", title="x", prompt="y").error
    assert "Timekeeper" in tool.run(SimpleNamespace(db=db, runtime=runtime, services={}, config={}, approve_command=lambda *_: True), operation="add", title="x", prompt="y", cron="0 8 * * *").error

    result = tool.run(ctx, operation="add", title="Morning Brief", prompt="brief me", cron="0 9 * * *")
    assert not result.success
    assert "already exists" in result.error


def test_next_turn_after_history_load_sends_loaded_history_to_llm():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Old chat")
    db.save_message(conv_id, "user", "earlier")
    db.save_message(conv_id, "assistant", "previous answer")
    llm = FakeLLM([FakeResponse.text("next")])
    runtime = ConversationRuntime(db=db, services={"llm": llm}, tool_registry=FakeToolRegistry())

    runtime.load_history("chat", conv_id)
    result = runtime.handle_action("chat", "send_text", "continue")

    assert result.messages[-1] == "next"
    assert [(m["role"], m["content"]) for m in llm.seen[0][0]] == [
        ("system", ""),
        ("user", "earlier"),
        ("assistant", "previous answer"),
        ("user", "continue"),
    ]


def test_agent_tool_approval_uses_state_machine_phase_and_resumes():
    class NeedsApproval(BaseTool):
        name = "needs_approval"
        description = "Needs approval"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, context, **_kwargs):
            return ToolResult(data={"approved": context.approve_command("danger", "test approval")}, llm_summary="approved")

    registry = ToolRegistry(None, {"tool_timeout": 10})
    registry.register(NeedsApproval())
    runtime = ConversationRuntime(
        services={"llm": FakeLLM([
            FakeResponse.tool([{"id": "tc1", "name": "needs_approval", "arguments": "{}"}]),
            FakeResponse.text("done"),
        ])},
        tool_registry=registry,
    )
    registry.runtime = runtime
    seen = []
    worker = threading.Thread(target=lambda: seen.append(runtime.handle_action("chat", "send_text", "go")), daemon=True)

    worker.start()
    deadline = time.time() + 5
    while time.time() < deadline and runtime.sessions["chat"].cs.phase != PHASE_APPROVING_REQUEST:
        time.sleep(0.01)
    assert runtime.sessions["chat"].cs.phase == PHASE_APPROVING_REQUEST
    assert runtime.handle_action("chat", "answer_approval", {"value": True}).ok
    worker.join(timeout=5)

    assert seen and seen[0].messages[-1] == "done"
    assert runtime.sessions["chat"].cs.phase == "awaiting_input"


def test_stale_approval_request_id_does_not_answer_current_frame():
    runtime = ConversationRuntime()
    req = runtime.request_input("chat", "Pick", "Pick one", type="string")

    result = runtime.handle_action("chat", "answer_approval", {"request_id": "old", "value": "x"})

    assert not result.ok
    assert req.id in runtime._approval_requests
    assert runtime.sessions["chat"].cs.phase == PHASE_APPROVING_REQUEST


def test_inject_user_message_appends_without_driving_agent_turn():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Inbox")
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("should not run")])})

    runtime.inject_user_message("inbox:job", "later please", conversation_id=conv_id)

    session = runtime.sessions["inbox:job"]
    rows = [r for r in db.get_conversation_messages(conv_id) if r["role"] != "system"]
    assert session.cs.turn_priority == "user"
    assert session.history == [{"role": "user", "content": "later please"}]
    assert [(r["role"], r["content"]) for r in rows] == [("user", "later please")]


def test_new_conversation_action_uses_reset_lifecycle():
    events = []
    unsubs = [
        bus.subscribe(SESSION_CLOSED, lambda payload: events.append((SESSION_CLOSED, payload))),
        bus.subscribe(SESSION_CREATED, lambda payload: events.append((SESSION_CREATED, payload))),
    ]
    runtime = ConversationRuntime()

    try:
        runtime.get_session("chat").history.append({"role": "user", "content": "old"})
        result = runtime.handle_action("chat", "new_conversation")
    finally:
        for unsub in unsubs:
            unsub()

    assert result.messages == ["New conversation started. Agent: default."]
    assert runtime.sessions["chat"].history == []
    assert [name for name, _ in events[-2:]] == ["session_closed", "session_created"]


def test_new_command_starts_default_main_conversation():
    db = FakeConversationDB()
    runtime = ConversationRuntime(db=db)
    result = NewCommand().run({}, SimpleNamespace(db=db, runtime=runtime, session_key="chat"))

    assert result == "Started new conversation #1 under 'Main'.\nAgent: default"
    assert db.conversations[1]["category"] is None
    assert runtime.sessions["chat"].conversation_id == 1
    assert runtime.sessions["chat"].profile_override == "default"


def test_conversations_command_changes_category_after_selection():
    db = FakeConversationDB()
    cid = db.create_conversation("Old", category=None)
    result = ConversationsCommand().run({"category": "Main", "conversation_id": str(cid), "action": "Change category", "target_category": "Projects"}, SimpleNamespace(db=db, runtime=ConversationRuntime(db=db), session_key="chat"))

    assert result == "Conversation #1 moved to 'Projects'."
    assert db.conversations[cid]["category"] == "Projects"


def test_conversations_command_form_manages_existing_conversations_only():
    db = FakeConversationDB()
    cid = db.create_conversation("Old", category=None)
    cmd = ConversationsCommand()

    top = cmd.form({}, SimpleNamespace(db=db))[0]
    steps = cmd.form({"category": "Main", "conversation_id": str(cid)}, SimpleNamespace(db=db))

    assert "➕ New conversation" not in top.enum
    assert "Change category" in steps[-1].enum


def test_tool_budget_gets_one_final_model_summary():
    schema = {"function": {"name": "echo", "parameters": {"type": "object", "properties": {}, "required": []}}}
    registry = FakeToolRegistry([schema], {"echo": FakeToolResult(success=True, data={}, llm_summary="ok", attachment_paths=[])})
    cs = make_cs(tools={"echo": CallableSpec("echo", handler=lambda _cs, _actor, args: registry.call("echo", **args))})
    cs.set_priority("agent")
    llm = FakeLLM([FakeResponse.tool([{"id": f"tc{i}", "name": "echo", "arguments": "{}"}]) for i in range(24)] + [FakeResponse.text("here is what I found")])

    final_text, new_messages, _ = ConversationLoop(llm, registry, {}, "").drive(cs, "agent", [{"role": "user", "content": "loop"}])

    assert final_text == "here is what I found"
    assert new_messages[-1]["content"] == "here is what I found"


def test_busy_runtime_rejects_text_but_accepts_cancel_signal():
    runtime = ConversationRuntime()
    session = runtime.get_session("chat-busy")
    session.busy = True

    text_result = runtime.handle_action("chat-busy", "send_text", "hello?")
    cancel_result = runtime.handle_action("chat-busy", "cancel")

    assert text_result.messages == ["Not your turn - I'm still working. Send /cancel to interrupt."]
    assert cancel_result.messages == ["Cancelled."]
    assert session.cancel_event.is_set()


# ──────────────────────────────────────────────────────────────────────────
# 3. Form path
# ──────────────────────────────────────────────────────────────────────────


def test_form_path_collects_args_then_runs_command():
    captured: dict = {}

    def handler(_cs, _actor, args):
        captured.update(args)
        return f"ran with {args['name']}={args['count']}"

    cmd = CallableSpec(
        name="hello",
        handler=handler,
        form=[
            FormStep("name", prompt="Name?", required=True),
            FormStep("count", prompt="Count?", required=True, type="integer"),
        ],
    )
    cs = make_cs(commands={"hello": cmd})

    # Step 1: user invokes the command with no args. Enters form-filling phase.
    r1 = create_action(cs, "call_command", {"name": "hello"}, "user").enact()
    assert r1.ok
    assert cs.phase == "filling_command_form"

    # Step 2: submit first form value.
    r2 = create_action(cs, "submit_form_text", "world", "user").enact()
    assert r2.ok
    assert cs.phase == "filling_command_form"  # one more field still pending

    # Step 3: submit second form value. Replays the original CallCommand.
    r3 = create_action(cs, "submit_form_text", "3", "user").enact()
    assert r3.ok, r3.error
    assert cs.phase == "awaiting_input"
    assert captured == {"name": "world", "count": 3}
    assert (r3.data or {}).get("result") == "ran with world=3"


def test_invalid_integer_form_input_keeps_same_field_active():
    cmd = CallableSpec("count", handler=lambda *_: "ok", form=[FormStep("n", required=True, type="integer")])
    cs = make_cs(commands={"count": cmd})

    assert create_action(cs, "call_command", {"name": "count"}, "user").enact().ok
    result = create_action(cs, "submit_form_text", "nope", "user").enact()

    assert not result.ok
    assert result.error.code == "invalid_input"
    assert cs.frame.step.name == "n"


def test_new_command_supersedes_pending_form():
    events = []
    unsub = bus.subscribe(COMMAND_CALL_FINISHED, lambda payload: events.append(payload))
    cs = make_cs(commands={
        "old": CallableSpec("old", handler=lambda *_: None, form=[FormStep("x", required=True)]),
        "new": CallableSpec("new", handler=lambda *_: "ok", form=[FormStep("y", required=True)]),
    })
    try:
        assert create_action(cs, "call_command", {"name": "old"}, "user").enact().ok
        old_call_id = cs.frame.data["call_id"]
        result = create_action(cs, "call_command", {"name": "new"}, "user").enact()
    finally:
        unsub()

    assert result.ok
    assert cs.frame.name == "new"
    assert events[-1] == {"session_key": None, "call_id": old_call_id, "command_name": "old", "ok": False, "error": "superseded"}


def test_tool_optional_args_do_not_start_forms():
    captured: dict = {}

    def handler(_cs, _actor, args):
        captured.update(args)
        return "read"

    cmd = CallableSpec(
        name="read_file",
        handler=handler,
        form=[
            FormStep("path", required=True),
            FormStep("offset", required=False, type="integer", default=1),
            FormStep("limit", required=False, type="integer"),
        ],
    )
    cs = make_cs(tools={"read_file": cmd})
    cs.set_priority("agent")

    result = create_action(cs, "call_tool", {"name": "read_file", "args": {"path": "app.log"}}, "agent").enact()

    assert result.ok, result.error
    assert cs.phase == "awaiting_input"
    assert captured == {"path": "app.log"}
    assert (result.data or {}).get("result") == "read"


def test_optional_prompted_args_can_be_skipped():
    captured: dict = {}

    cmd = CallableSpec(
        name="llm",
        handler=lambda _cs, _actor, args: captured.update(args) or "ok",
        form=[
            FormStep("subcommand", required=True, enum=["list", "show"]),
            FormStep("args", required=False, default="", prompt_when_missing=True),
        ],
    )
    cs = make_cs(commands={"llm": cmd})

    r1 = create_action(cs, "call_command", {"name": "llm", "args": {}}, "user").enact()
    assert r1.ok and cs.frame.step.name == "subcommand"

    r2 = create_action(cs, "submit_form_text", "list", "user").enact()
    assert r2.ok and cs.frame.step.name == "args"

    r3 = create_action(cs, "skip_form", None, "user").enact()
    assert r3.ok, r3.error
    assert r3.message == "Skipped."
    assert captured == {"subcommand": "list", "args": ""}
    assert (r3.data or {}).get("result") == "ok"


def test_tools_command_prompts_optional_schema_args_before_calling_tool():
    registry = SimpleNamespace(tools={"schedule_subagent": ScheduleSubagent()})
    tools = ToolsCommand()
    cmd = CallableSpec(
        "tools",
        handler=lambda *_: "ok",
        form_factory=lambda args, _cs: tools.form(args, SimpleNamespace(tool_registry=registry)),
    )
    cs = make_cs(commands={"tools": cmd})

    assert create_action(cs, "call_command", {"name": "tools", "args": {"tool_name": "schedule_subagent", "action": "call"}}, "user").enact().ok
    assert cs.frame.step.name == "operation"
    assert create_action(cs, "submit_form_text", "add", "user").enact().ok
    assert cs.frame.step.name == "title"
    assert create_action(cs, "submit_form_text", "Nightly Wisdom", "user").enact().ok
    assert cs.frame.step.name == "prompt"
    assert create_action(cs, "submit_form_text", "send wisdom", "user").enact().ok
    assert cs.frame.step.name == "cron"


def test_cancel_and_skip_confirm_when_no_handler_message():
    cs = make_cs(commands={
        "need": CallableSpec("need", handler=lambda *_: None, form=[FormStep("x", required=True)]),
        "maybe": CallableSpec("maybe", handler=lambda *_: None, form=[FormStep("x", required=False, default="", prompt_when_missing=True)]),
    })

    assert create_action(cs, "call_command", {"name": "need"}, "user").enact().ok
    assert create_action(cs, "cancel", None, "user").enact().message == "Cancelled."
    assert create_action(cs, "call_command", {"name": "maybe"}, "user").enact().ok
    assert create_action(cs, "skip_form", None, "user").enact().message == "Skipped."


def test_dynamic_form_steps_follow_collected_args():
    captured: dict = {}

    def form(args, _cs):
        steps = [FormStep("subcommand", required=True, enum=["list", "add"])]
        if args.get("subcommand") == "add":
            steps += [
                FormStep("model", required=True),
                FormStep("endpoint", required=False, default="", prompt_when_missing=True),
            ]
        return steps

    cmd = CallableSpec(
        name="llm",
        handler=lambda _cs, _actor, args: captured.update(args) or "ok",
        form_factory=form,
    )
    cs = make_cs(commands={"llm": cmd})

    assert create_action(cs, "call_command", {"name": "llm", "args": {}}, "user").enact().ok
    assert cs.frame.step.name == "subcommand"
    assert create_action(cs, "submit_form_text", "add", "user").enact().ok
    assert cs.frame.step.name == "model"
    assert create_action(cs, "submit_form_text", "gpt", "user").enact().ok
    assert cs.frame.step.name == "endpoint"
    result = create_action(cs, "skip_form", None, "user").enact()

    assert result.ok, result.error
    assert captured == {"subcommand": "add", "model": "gpt", "endpoint": ""}


def test_back_form_revisits_static_step_and_uses_replacement_value():
    captured = {}
    cs = make_cs(commands={"hello": CallableSpec("hello", handler=lambda _cs, _actor, args: captured.update(args) or "ok", form=[FormStep("name", required=True), FormStep("count", required=True, type="integer")])})

    assert create_action(cs, "call_command", {"name": "hello", "args": {}}, "user").enact().ok
    assert create_action(cs, "submit_form_text", "old", "user").enact().ok
    assert create_action(cs, "back_form", None, "user").enact().ok
    assert cs.frame.step.name == "name"
    assert create_action(cs, "submit_form_text", "new", "user").enact().ok
    assert create_action(cs, "submit_form_text", "2", "user").enact().ok
    assert captured == {"name": "new", "count": 2}


def test_back_form_follows_dynamic_form_factory():
    def form(args, _cs):
        return [FormStep("subcommand", required=True, enum=["list", "add"])] + ([FormStep("model", required=True), FormStep("endpoint", required=False, default="", prompt_when_missing=True)] if args.get("subcommand") == "add" else [])

    cs = make_cs(commands={"llm": CallableSpec("llm", handler=lambda *_: "ok", form_factory=form)})

    assert create_action(cs, "call_command", {"name": "llm", "args": {}}, "user").enact().ok
    assert create_action(cs, "submit_form_text", "add", "user").enact().ok
    assert create_action(cs, "submit_form_text", "gpt", "user").enact().ok
    assert cs.frame.step.name == "endpoint"
    assert create_action(cs, "back_form", None, "user").enact().ok
    assert cs.frame.step.name == "model"
    assert cs.frame.data["args"] == {"subcommand": "add"}


def test_back_form_can_undo_skipped_optional_before_next_field():
    captured = {}
    cs = make_cs(commands={"setup": CallableSpec("setup", handler=lambda _cs, _actor, args: captured.update(args) or "ok", form=[FormStep("name", required=True), FormStep("note", required=False, default="", prompt_when_missing=True), FormStep("count", required=True, type="integer")])})

    assert create_action(cs, "call_command", {"name": "setup", "args": {}}, "user").enact().ok
    assert create_action(cs, "submit_form_text", "job", "user").enact().ok
    assert create_action(cs, "skip_form", None, "user").enact().ok
    assert cs.frame.step.name == "count"
    assert create_action(cs, "back_form", None, "user").enact().ok
    assert cs.frame.step.name == "note"
    assert create_action(cs, "submit_form_text", "changed", "user").enact().ok
    assert create_action(cs, "submit_form_text", "7", "user").enact().ok
    assert captured == {"name": "job", "note": "changed", "count": 7}


def test_back_form_at_first_step_fails_without_mutating_prompt():
    cs = make_cs(commands={"hello": CallableSpec("hello", handler=lambda *_: "ok", form=[FormStep("name", required=True)])})

    assert create_action(cs, "call_command", {"name": "hello", "args": {}}, "user").enact().ok
    result = create_action(cs, "back_form", None, "user").enact()

    assert not result.ok
    assert result.error.message == "Nothing to go back to."
    assert cs.frame.step.name == "name"


def test_back_form_does_not_remove_prefilled_command_args():
    cs = make_cs(commands={"hello": CallableSpec("hello", handler=lambda *_: "ok", form=[FormStep("name", required=True), FormStep("count", required=True, type="integer")])})

    assert create_action(cs, "call_command", {"name": "hello", "args": {"name": "prefilled"}}, "user").enact().ok
    assert cs.frame.step.name == "count"
    result = create_action(cs, "back_form", None, "user").enact()

    assert not result.ok
    assert cs.frame.data["args"] == {"name": "prefilled"}
    assert cs.frame.step.name == "count"


# ──────────────────────────────────────────────────────────────────────────
# 4. legal_actions_in_phase smoke
# ──────────────────────────────────────────────────────────────────────────


def test_legal_actions_in_phase_returns_registered_types():
    from state_machine.action_map import legal_actions_in_phase
    base = legal_actions_in_phase("awaiting_input")
    assert "send_text" in base
    assert "call_command" in base
    assert "call_tool" in base
    assert "end_turn" in base

    form = legal_actions_in_phase("filling_command_form")
    assert "call_command" in form
    assert "submit_form_text" in form
    assert "back_form" in form
    assert "cancel" in form
