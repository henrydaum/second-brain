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

from agent.system_prompt import build_prompt_sections
from agent.tool_registry import ToolRegistry
from plugins.BaseTool import BaseTool, ToolResult
from plugins.commands.command_plan import PlanCommand
from plugins.commands.command_conversations import ConversationsCommand, NewCommand
from plugins.commands.command_tools import ToolsCommand
from plugins.services.service_llm import LLMResponse
from plugins.services import service_timekeeper as timekeeper_module
from plugins.services.service_timekeeper import TimekeeperService
from plugins.tools.tool_schedule_subagent import SCHEDULED, SCHEDULED_ONCE, ScheduleSubagent
from plugins.tools.tool_ask_user_question import AskUserQuestion
from plugins.tools.tool_edit_file import EditFile
from plugins.tools.tool_propose_plan import ProposePlan
from plugins.tools.tool_run_command import RunCommand
from plugins.tasks.task_spawn_subagent import SpawnSubagent
from state_machine.action_map import create_action
from state_machine.conversation import CallableSpec, ConversationState, FormStep, Participant
from runtime.conversation_loop import ConversationLoop
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST
from state_machine.serialization import latest_state, save_state_marker
from runtime.conversation_runtime import ConversationRuntime
from runtime.context import PLAN_MODE_PERMISSION_DENIED
from runtime.runtime_config import session_system_prompt
from runtime.session import RuntimeSession
from events.event_channels import CHAT_MESSAGE_PUSHED, COMMAND_CALL_FINISHED, SESSION_CLOSED, SESSION_CREATED, SESSION_TURN_COMPLETED, TOOL_CALL_FINISHED, TOOL_CALL_STARTED
from events.event_bus import bus


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────


class FakeResponse(SimpleNamespace):
    """Mimics the response shape ConversationLoop expects from an LLM."""

    @classmethod
    def text(cls, content: str) -> "FakeResponse":
        """Handle text."""
        return cls(content=content, has_tool_calls=False, tool_calls=[], is_error=False, prompt_tokens=0)

    @classmethod
    def tool(cls, calls: list[dict]) -> "FakeResponse":
        """Handle tool."""
        return cls(content=None, has_tool_calls=True, tool_calls=calls, is_error=False, prompt_tokens=0)


class FakeLLM:
    """Returns a scripted sequence of FakeResponse objects per call."""

    context_size = 0

    def __init__(self, responses: list[FakeResponse]):
        """Initialize the fake LLM."""
        self.responses = list(responses)
        self.calls = 0
        self.seen = []

    def chat_with_tools(self, messages, tools, attachments=None):
        """Handle chat with tools."""
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
        """Initialize the fake tool registry."""
        self._schemas = schemas or []
        self._tool_results = tool_results or {}
        self.tools = {name: SimpleNamespace(max_calls=3) for name in self._tool_results}
        self.max_tool_calls = 5
        self.called: list[tuple[str, dict]] = []

    def get_all_schemas(self) -> list[dict]:
        """Get all schemas."""
        return self._schemas

    def call(self, name: str, **kwargs):
        """Call fake tool registry."""
        self.called.append((name, kwargs))
        return self._tool_results.get(name, FakeToolResult(success=True, data={"echo": kwargs}, llm_summary=f"ran {name}", attachment_paths=[]))


class FakeConversationDB:
    """Test double for fake conversation DB."""
    def __init__(self):
        """Initialize the fake conversation DB."""
        self.conversations = {}
        self.messages = {}
        self.next_id = 1
        self.replaced_history = None

    def create_conversation(self, title="New conversation", kind="user", category=None):
        """Create conversation."""
        cid = self.next_id; self.next_id += 1
        self.conversations[cid] = {"id": cid, "title": title, "kind": kind, "category": category}
        self.messages[cid] = []
        return cid

    def get_conversation(self, conversation_id):
        """Get conversation."""
        return self.conversations.get(conversation_id)

    def save_message(self, conversation_id, role, content, tool_call_id=None, tool_name=None):
        """Save message."""
        self.messages.setdefault(conversation_id, []).append({"role": role, "content": content, "tool_call_id": tool_call_id, "tool_name": tool_name})

    def replace_conversation_messages(self, conversation_id, history):
        """Handle replace conversation messages."""
        self.replaced_history = list(history)
        self.messages[conversation_id] = []
        for msg in history:
            self.save_message(conversation_id, msg.get("role"), msg.get("content") or "", msg.get("tool_call_id"), msg.get("name"))

    def get_conversation_messages(self, conversation_id):
        """Get conversation messages."""
        return list(self.messages.get(conversation_id, []))

    def get_system_stats(self):
        """Get system stats."""
        return {"files": {}, "tasks": {}}

    def list_conversations_page(self, offset=0, limit=10, category=None):
        """List conversations page."""
        rows = list(self.conversations.values())
        if category == "":
            rows = [r for r in rows if r.get("category") in (None, "")]
        elif category is not None:
            rows = [r for r in rows if r.get("category") == category]
        return rows[offset:offset + limit], len(rows) > offset + limit

    def list_conversation_categories(self):
        """List conversation categories."""
        out = []
        for r in self.conversations.values():
            c = r.get("category")
            v = None if c in (None, "") else c
            if v not in out:
                out.append(v)
        return out

    def set_conversation_category(self, conversation_id, category):
        """Set conversation category."""
        self.conversations[conversation_id]["category"] = category


class FakeTimekeeper:
    """Test double for fake timekeeper."""
    loaded = True

    def __init__(self, jobs=None):
        """Initialize the fake timekeeper."""
        self.jobs = dict(jobs or {})

    def get_job(self, name):
        """Get job."""
        return self.jobs.get(name)

    def list_jobs(self):
        """List jobs."""
        return dict(self.jobs)

    def create_job(self, name, job_def):
        """Create job."""
        if name in self.jobs:
            raise ValueError(f"Job '{name}' already exists.")
        self.jobs[name] = dict(job_def)
        return self.jobs[name]

    def update_job(self, name, patch):
        """Update job."""
        self.jobs[name].update(patch)
        return self.jobs[name]

    def remove_job(self, name):
        """Remove job."""
        return self.jobs.pop(name, None) is not None

    def cron_to_text(self, expr):
        """Handle cron to text."""
        if expr == "bad cron":
            raise ValueError("Invalid cron expression")
        return expr


def make_cs(commands: dict[str, CallableSpec] | None = None, tools: dict[str, CallableSpec] | None = None) -> ConversationState:
    """Build cs."""
    return ConversationState([
        Participant("user", "user", commands=commands or {}),
        Participant("agent", "agent", tools=tools or {}),
    ])


def sectioned_prompt():
    """Build a tiny sectioned system prompt for ordering tests."""
    return [
        {"role": "system", "content": "[STATIC SYSTEM PROMPT]\nstatic"},
        {"role": "system", "content": "[SEMI-STABLE TOOL/SCHEMA INFO]\nsemi"},
        {"role": "system", "content": "[DYNAMIC RUNTIME CONTEXT]\ndynamic"},
    ]


def test_session_system_prompt_includes_conversation_metadata():
    """Verify session system prompt includes conversation metadata."""
    db = FakeConversationDB()
    cid = db.create_conversation("Build Runtime Prompt", category="Projects")
    session = RuntimeSession("chat", make_cs(), conversation_id=cid)
    session.system_prompt_extras["pin"] = "Pinned runtime note."
    runtime = SimpleNamespace(db=db, system_prompt=lambda: [
        {"role": "system", "content": "[STATIC SYSTEM PROMPT]\nbase"},
        {"role": "system", "content": "[SEMI-STABLE TOOL/SCHEMA INFO]\ntools"},
        {"role": "system", "content": "[DYNAMIC RUNTIME CONTEXT]\nbase dynamic"},
    ], active_session_key="chat")

    prompt = session_system_prompt(runtime, session)()
    dynamic = prompt[-1]["content"]

    assert "## Current conversation" in dynamic
    assert f"Number: {cid}" in dynamic
    assert "Category: Projects" in dynamic
    assert "Title: Build Runtime Prompt" in dynamic
    assert "Pinned runtime note." in dynamic


def test_build_prompt_sections_places_stable_and_volatile_content():
    """Verify prompt builder separates cacheable and volatile prompt content."""
    db = FakeConversationDB()
    registry = FakeToolRegistry([{"function": {"name": "demo", "description": "Demo tool."}}])
    sections = build_prompt_sections(
        db, None, registry, {"llm": SimpleNamespace(loaded=True, model_name="gpt-test")},
        commands={"new": CallableSpec("new", form=[FormStep("title", required=False)])},
        config={"sync_directories": ["C:/sync"]},
        conversation_metadata={"id": 7, "category": "Projects", "title": "Cache Work"},
        prompt_extras={"warning": "Volatile warning."},
    )

    static, semi, dynamic = [m["content"] for m in sections]
    assert static.startswith("[STATIC SYSTEM PROMPT]")
    assert "Core Identity" in static
    assert "Current date and time" not in static
    assert "Memory (from memory.md)" not in static
    assert semi.startswith("[SEMI-STABLE TOOL/SCHEMA INFO]")
    assert "demo: Demo tool." in semi
    assert "/new [title]" in semi
    assert "Current conversation" not in semi
    assert dynamic.startswith("[DYNAMIC RUNTIME CONTEXT]")
    assert "Current date and time" in dynamic
    assert "Current model: gpt-test." in dynamic
    assert "C:/sync" in dynamic
    assert "Title: Cache Work" in dynamic
    assert "Volatile warning." in dynamic


def test_loop_messages_put_dynamic_context_before_current_user_turn():
    """Verify dynamic runtime context sits after prior history."""
    loop = ConversationLoop(FakeLLM([]), FakeToolRegistry(), {}, sectioned_prompt)
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new"},
    ]

    messages = loop._messages(history)

    assert [(m["role"], m["content"]) for m in messages] == [
        ("system", "[STATIC SYSTEM PROMPT]\nstatic"),
        ("system", "[SEMI-STABLE TOOL/SCHEMA INFO]\nsemi"),
        ("user", "old"),
        ("assistant", "old reply"),
        ("system", "[DYNAMIC RUNTIME CONTEXT]\ndynamic"),
        ("user", "new"),
    ]


def test_loop_messages_preserve_current_tool_turn_adjacency():
    """Verify assistant/tool-call rows stay in the current turn tail."""
    loop = ConversationLoop(FakeLLM([]), FakeToolRegistry(), {}, sectioned_prompt)
    tool_call = {"id": "tc1", "function": {"name": "echo", "arguments": "{}"}}
    history = [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": "tc1", "name": "echo", "content": "{}"},
    ]

    messages = loop._messages(history)

    assert [m["role"] for m in messages] == ["system", "system", "system", "user", "assistant", "tool"]
    assert messages[3]["content"] == "run"
    assert messages[4]["tool_calls"] == [tool_call]
    assert messages[5]["tool_call_id"] == "tc1"


def test_loop_messages_keep_legacy_string_prompt_compatibility():
    """Verify old one-string system prompts still work."""
    messages = ConversationLoop(FakeLLM([]), FakeToolRegistry(), {}, "legacy")._messages([
        {"role": "system", "content": "stored marker"},
        {"role": "user", "content": "hi"},
    ])

    assert messages == [{"role": "system", "content": "legacy"}, {"role": "user", "content": "hi"}]


# ──────────────────────────────────────────────────────────────────────────
# 1. Chat path
# ──────────────────────────────────────────────────────────────────────────


def test_chat_path_user_text_then_agent_reply():
    """Verify chat path user text then agent reply."""
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
    """Verify tool call path records assistant then tool then final text."""
    events = []
    schema = {"function": {"name": "echo", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}
    registry = FakeToolRegistry(
        schemas=[schema],
        tool_results={"echo": FakeToolResult(success=True, data={"q": "x"}, llm_summary="echoed: x", attachment_paths=[])},
    )
    def tool_handler(_cs, _actor, args):
        """Handle tool handler."""
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
    assert "result" not in next(e for e in cs.history if e["type"] == "call_tool")
    assert "result" not in str(cs.to_dict())
    assert events == [("start", "echo", "tc1", {"q": "x"}), ("finish", "echo", "tc1", True, None)]
    assert cs.turn_priority == "user"


def test_tool_budget_blocks_before_second_execution():
    """Verify tool budget blocks before second execution."""
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
    """Verify runtime emits session scoped tool status events."""
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
    """Verify runtime can chat without tool registry."""
    llm = FakeLLM([FakeResponse.text("plain reply")])
    result = ConversationRuntime(services={"llm": llm}).handle_action("chat", "send_text", "hello")

    assert result.messages[-1] == "plain reply"
    assert llm.seen[0][1] is None


def test_runtime_surfaces_llm_provider_error_to_chat():
    """Verify runtime surfaces LLM provider error to chat."""
    err = "your current token plan not support model, MiniMax-M2.7 (2061)"
    llm = FakeLLM([LLMResponse(error=err, error_code="provider_error")])

    result = ConversationRuntime(services={"llm": llm}).handle_action("chat", "send_text", "hello")

    assert not result.ok
    assert err in result.messages[-1]
    assert result.error["code"] == "agent_failed"


def test_runtime_attachment_bundle_reaches_llm():
    """Verify runtime attachment bundle reaches LLM."""
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
    """Verify iterate agent turn loads persists and emits completion."""
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
    """Verify load history restores saved agent profile and history."""
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
    """Verify set conversation notification mode updates stored marker."""
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    save_state_marker(db, conv_id, {"active_agent_profile": "builder"})
    runtime = ConversationRuntime(db=db)

    assert runtime.set_conversation_notification_mode(conv_id, "IMPORTANT") == "on"
    assert latest_state(db.get_conversation_messages(conv_id))["notification_mode"] == "on"
    assert runtime.load_conversation("job", conv_id).notification_mode == "on"


def test_set_conversation_notification_mode_updates_live_session():
    """Verify set conversation notification mode updates live session."""
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
    """Verify notification off suppresses background final answer push."""
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
    """Verify spawn subagent drives inactive conversation."""
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    db.save_message(conv_id, "user", "earlier")
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("done")])}, tool_registry=FakeToolRegistry())
    result = SpawnSubagent().run_event("run", {"conversation_id": conv_id, "prompt": "wake up"}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))

    assert result.success
    assert f"spawn_subagent:{conv_id}" not in runtime.sessions
    assert [(r["role"], r["content"]) for r in db.get_conversation_messages(conv_id) if r["role"] != "system"] == [
        ("user", "earlier"), ("user", "wake up"), ("assistant", "done")
    ]


def test_spawn_subagent_creates_missing_conversation():
    """Verify spawn subagent creates missing conversation."""
    db = FakeConversationDB()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("done")])}, tool_registry=FakeToolRegistry())

    result = SpawnSubagent().run_event("run", {"title": "Morning Brief", "prompt": "wake up"}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))

    assert result.success
    cid = result.data["conversation_id"]
    assert db.get_conversation(cid)["title"] == "Morning Brief"
    assert db.get_conversation(cid)["category"] == SCHEDULED
    assert latest_state(db.get_conversation_messages(cid))["profile_override"] == "default"


def test_spawn_subagent_repairs_deleted_conversation_and_updates_recurring_job():
    """Verify spawn subagent repairs deleted conversation and updates recurring job."""
    db = FakeConversationDB()
    tk = FakeTimekeeper({"brief": {"payload": {"title": "Brief", "prompt": "wake up", "conversation_id": 999}}})
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([FakeResponse.text("done")])}, tool_registry=FakeToolRegistry())
    payload = {"title": "Brief", "prompt": "wake up", "conversation_id": 999, "_timekeeper": {"job_name": "brief", "one_time": False}}

    result = SpawnSubagent().run_event("run", payload, SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}))

    assert result.success
    assert result.data["conversation_id"] != 999
    assert tk.jobs["brief"]["payload"]["conversation_id"] == result.data["conversation_id"]


def test_spawn_subagent_rejects_active_conversation():
    """Verify spawn subagent rejects active conversation."""
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
    """Verify spawn subagent rejects busy background session."""
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    runtime = ConversationRuntime(db=db)
    runtime.load_conversation(f"spawn_subagent:{conv_id}", conv_id).busy = True

    result = SpawnSubagent().run_event("run", {"conversation_id": conv_id, "prompt": "wake up"}, SimpleNamespace(db=db, runtime=runtime, services={}, config={}))

    assert not result.success
    assert "already running" in result.error


def test_spawn_subagent_parses_attachment_paths():
    """Verify spawn subagent parses attachment paths."""
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
    """Verify spawn subagent validates payload and attachment."""
    db = FakeConversationDB()
    runtime = ConversationRuntime(db=db)
    task = SpawnSubagent()
    ctx = SimpleNamespace(db=db, runtime=runtime, services={}, config={})
    conv_id = db.create_conversation("Cron")

    assert "prompt" in task.run_event("run", {"conversation_id": conv_id}, ctx).error
    assert "Attachment not found" in task.run_event("run", {"conversation_id": conv_id, "prompt": "x", "attachments": [".missing_spawn_attachment.txt"]}, ctx).error


def test_schedule_subagent_recurring_cron_requires_approval_and_creates_job():
    """Verify schedule subagent recurring cron requires approval and creates job."""
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
    """Verify tools command user initiated schedule creates cron without second approval."""
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
    """Verify schedule subagent one time cron computes run at."""
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
    """Verify schedule subagent stores attachments in payload."""
    db = FakeConversationDB()
    tk = FakeTimekeeper()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: True)

    result = ScheduleSubagent().run(ctx, operation="add", title="With Files", prompt="read", cron="0 8 * * *", attachments=["a.txt", "b.txt"])

    assert result.success
    assert tk.jobs["with_files"]["payload"]["attachments"] == ["a.txt", "b.txt"]


def test_schedule_subagent_list_edit_and_remove():
    """Verify schedule subagent list edit and remove."""
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
    """Verify schedule subagent resolves literal job key or payload title."""
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
    """Verify schedule subagent edit schedule shapes."""
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
    """Verify timekeeper one time jobs auto delete after emit."""
    saved = {}
    monkeypatch.setattr(timekeeper_module.config_manager, "load_plugin_config", lambda: {})
    monkeypatch.setattr(timekeeper_module.config_manager, "save_plugin_config", saved.update)
    run_at = timekeeper_module._now_local().isoformat()
    tk = TimekeeperService({"scheduled_jobs": {"once": {"channel": "test.once", "run_at": run_at, "one_time": True, "payload": {}}}})

    tk._emit_job("once", tk.get_job("once"), timekeeper_module._now_local())

    assert tk.get_job("once") is None
    assert "once" not in saved["scheduled_jobs"]


def test_schedule_subagent_denied_approval_has_no_side_effects():
    """Verify schedule subagent denied approval has no side effects."""
    db = FakeConversationDB()
    tk = FakeTimekeeper()
    runtime = ConversationRuntime(db=db, services={"llm": FakeLLM([])}, tool_registry=FakeToolRegistry())
    ctx = SimpleNamespace(db=db, runtime=runtime, services={"timekeeper": tk}, config={}, approve_command=lambda *_: False)

    result = ScheduleSubagent().run(ctx, operation="add", title="Nope", prompt="do it", cron="0 8 * * *")

    assert not result.success
    assert db.conversations == {}
    assert tk.jobs == {}


def test_schedule_subagent_validation_and_duplicate_title():
    """Verify schedule subagent validation and duplicate title."""
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
    """Verify next turn after history load sends loaded history to LLM."""
    db = FakeConversationDB()
    conv_id = db.create_conversation("Old chat")
    db.save_message(conv_id, "user", "earlier")
    db.save_message(conv_id, "assistant", "previous answer")
    llm = FakeLLM([FakeResponse.text("next")])
    runtime = ConversationRuntime(db=db, services={"llm": llm}, tool_registry=FakeToolRegistry())

    runtime.load_history("chat", conv_id)
    result = runtime.handle_action("chat", "send_text", "continue")

    assert result.messages[-1] == "next"
    expected_prompt = f"\n\n## Current conversation\nNumber: {conv_id}\nCategory: Main\nTitle: Old chat"
    assert [(m["role"], m["content"]) for m in llm.seen[0][0]] == [
        ("system", expected_prompt),
        ("user", "earlier"),
        ("assistant", "previous answer"),
        ("user", "continue"),
    ]


def test_agent_tool_approval_uses_state_machine_phase_and_resumes():
    """Verify agent tool approval uses state machine phase and resumes."""
    class NeedsApproval(BaseTool):
        """Needs approval."""
        name = "needs_approval"
        description = "Needs approval"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run(self, context, **_kwargs):
            """Run needs approval."""
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


def test_plan_command_toggles_and_persists_plan_mode():
    """Verify /plan toggles and persists plan mode."""
    db = FakeConversationDB()
    runtime = ConversationRuntime(db=db)
    session = runtime.open_session("chat", title="Plan test")
    ctx = SimpleNamespace(runtime=runtime, session_key="chat")

    assert PlanCommand().run({}, ctx) == "Plan mode on."
    assert session.plan_mode is True
    assert latest_state(db.get_conversation_messages(session.conversation_id))["plan_mode"] is True

    restored = ConversationRuntime(db=db).load_conversation("chat", session.conversation_id)
    assert restored.plan_mode is True
    assert PlanCommand().run({}, SimpleNamespace(runtime=runtime, session_key="chat")) == "Plan mode off."


def test_session_prompt_includes_plan_mode_guidance_only_when_active():
    """Verify plan-mode prompt guidance is session-scoped."""
    session = RuntimeSession("chat", make_cs())
    runtime = SimpleNamespace(db=None, system_prompt=sectioned_prompt, active_session_key="chat")

    assert "Plan mode is active" not in session_system_prompt(runtime, session)()[-1]["content"]
    session.plan_mode = True

    dynamic = session_system_prompt(runtime, session)()[-1]["content"]
    assert "Plan mode is active" in dynamic
    assert "propose_plan" in dynamic


def test_plan_mode_rejects_permission_dialogs_without_request(monkeypatch):
    """Verify plan mode denies approval-gated tools before a dialog appears."""
    monkeypatch.setattr("plugins.tools.tool_run_command.subprocess.run", lambda *a, **k: pytest.fail("should not run"))
    registry = ToolRegistry(None, {"tool_timeout": 10, "skip_permissions": ["run_command"]})
    registry.register(RunCommand())
    runtime = ConversationRuntime(config=registry.config, tool_registry=registry)
    registry.runtime = runtime
    runtime.get_session("chat").plan_mode = True
    runtime.active_session_key = "chat"

    result = registry.call("run_command", command="git pull", justification="update repo", _session_key="chat")

    assert not result.success
    assert result.error == PLAN_MODE_PERMISSION_DENIED
    assert runtime._approval_requests == {}


def test_skip_permissions_auto_approves_outside_plan_mode():
    """Verify skip_permissions auto-approves approval dialogs."""
    path = Path(".codex_skip_permissions_test.txt")
    registry = ToolRegistry(None, {"tool_timeout": 10, "skip_permissions": ["edit_file"]})
    registry.register(EditFile())
    runtime = ConversationRuntime(config=registry.config, tool_registry=registry)
    registry.runtime = runtime
    runtime.get_session("chat")
    runtime.active_session_key = "chat"
    try:
        result = registry.call(
            "edit_file",
            operation="create",
            path=str(path),
            content="ok",
            justification="test skip permissions",
            _session_key="chat",
        )
        assert result.success
        assert path.read_text(encoding="utf-8") == "ok"
        assert runtime._approval_requests == {}
    finally:
        path.unlink(missing_ok=True)


def test_plan_mode_overrides_skip_permissions_for_plan_unsafe_tools():
    """Verify plan mode beats skip_permissions."""
    path = Path(".codex_skip_permissions_test.txt")
    registry = ToolRegistry(None, {"tool_timeout": 10, "skip_permissions": ["edit_file"]})
    registry.register(EditFile())
    runtime = ConversationRuntime(config=registry.config, tool_registry=registry)
    registry.runtime = runtime
    runtime.get_session("chat").plan_mode = True
    runtime.active_session_key = "chat"

    result = registry.call(
        "edit_file",
        operation="create",
        path=str(path),
        content="nope",
        justification="test plan override",
        _session_key="chat",
    )

    assert not result.success
    assert result.error == PLAN_MODE_PERMISSION_DENIED
    assert not path.exists()


def test_read_only_run_command_still_works_in_plan_mode(monkeypatch):
    """Verify read-only run_command calls are allowed in plan mode."""
    monkeypatch.setattr("plugins.tools.tool_run_command.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="Python 3.x\n", stderr="", returncode=0))
    registry = ToolRegistry(None, {"tool_timeout": 10})
    registry.register(RunCommand())
    runtime = ConversationRuntime(config=registry.config, tool_registry=registry)
    registry.runtime = runtime
    runtime.get_session("chat").plan_mode = True
    runtime.active_session_key = "chat"

    result = registry.call("run_command", command="python --version", justification="check python", _session_key="chat")

    assert result.success
    assert "Python 3.x" in result.llm_summary


def test_propose_plan_approval_turns_plan_mode_off():
    """Verify approved propose_plan exits plan mode."""
    runtime = ConversationRuntime()
    runtime.get_session("chat").plan_mode = True
    ctx = SimpleNamespace(
        request_user_input=lambda title, prompt, **kw: runtime.request_input("chat", title, prompt, **kw),
        runtime=runtime,
        session_key="chat",
    )
    seen = []
    worker = threading.Thread(target=lambda: seen.append(ProposePlan().run(ctx, title="Do it", plan="- Step")), daemon=True)

    worker.start()
    deadline = time.time() + 5
    while time.time() < deadline and runtime.sessions["chat"].cs.phase != PHASE_APPROVING_REQUEST:
        time.sleep(0.01)
    assert runtime.handle_action("chat", "answer_approval", {"value": True}).ok
    worker.join(timeout=5)

    assert seen and seen[0].success
    assert runtime.sessions["chat"].plan_mode is False


def test_tools_command_toggles_skip_permissions(monkeypatch):
    """Verify /tools can add and remove skip_permissions entries."""
    saved = []
    monkeypatch.setattr("plugins.commands.command_tools.config_manager.save", lambda config: saved.append(dict(config)))
    context = SimpleNamespace(config={"skip_permissions": []}, tool_registry=SimpleNamespace(tools={"edit_file": object()}), session_key="chat")
    cmd = ToolsCommand()

    assert cmd.run({"tool_name": "edit_file", "action": "enable_skip_permissions"}, context) == "Skip permissions enabled for edit_file."
    assert context.config["skip_permissions"] == ["edit_file"]
    assert saved[-1]["skip_permissions"] == ["edit_file"]

    assert cmd.run({"tool_name": "edit_file", "action": "disable_skip_permissions"}, context) == "Skip permissions disabled for edit_file."
    assert context.config["skip_permissions"] == []
    assert saved[-1]["skip_permissions"] == []


def test_stale_approval_request_id_does_not_answer_current_frame():
    """Verify stale approval request ID does not answer current frame."""
    runtime = ConversationRuntime()
    req = runtime.request_input("chat", "Pick", "Pick one", type="string")

    result = runtime.handle_action("chat", "answer_approval", {"request_id": "old", "value": "x"})

    assert not result.ok
    assert req.id in runtime._approval_requests
    assert runtime.sessions["chat"].cs.phase == PHASE_APPROVING_REQUEST


def test_cancel_resolves_pending_user_input_without_value():
    """Verify cancel resolves pending user input without value."""
    runtime = ConversationRuntime()
    req = runtime.request_input("chat", "Pick", "Pick one", type="string")

    result = runtime.handle_action("chat", "cancel")

    assert result.ok
    assert req.wait(timeout=0)
    assert req.value is None
    assert req.metadata["cancelled"] is True
    assert req.id not in runtime._approval_requests
    assert runtime.sessions["chat"].cs.phase == "awaiting_input"


def test_cancel_resolves_pending_user_input_while_busy():
    """Verify cancel resolves pending user input while busy."""
    runtime = ConversationRuntime()
    req = runtime.request_input("chat", "Pick", "Pick one", type="string")
    runtime.sessions["chat"].busy = True

    result = runtime.handle_action("chat", "cancel")

    assert result.ok
    assert req.wait(timeout=0)
    assert req.metadata["cancelled"] is True
    assert runtime.sessions["chat"].cs.phase == "awaiting_input"


def test_ask_user_question_returns_typed_values():
    """Verify ask user question returns typed values."""
    runtime = ConversationRuntime()

    def ask(type_, answer):
        """Handle ask."""
        ctx = SimpleNamespace(
            request_user_input=lambda title, prompt, **kw: runtime.request_input("chat", title, prompt, **kw),
            runtime=runtime,
            session_key="chat",
        )
        seen = []
        worker = threading.Thread(target=lambda: seen.append(AskUserQuestion().run(ctx, question="Value?", type=type_)), daemon=True)
        worker.start()
        deadline = time.time() + 5
        while time.time() < deadline and ("chat" not in runtime.sessions or runtime.sessions["chat"].cs.phase != PHASE_APPROVING_REQUEST):
            time.sleep(0.01)
        assert runtime.handle_action("chat", "answer_approval", {"value": answer}).ok
        worker.join(timeout=5)
        assert seen and seen[0].success
        return seen[0].data["value"]

    assert ask("integer", "7") == 7
    assert ask("number", "1.5") == 1.5
    assert ask("array", '["a", "b"]') == ["a", "b"]
    assert ask("object", '{"x": 1}') == {"x": 1}


def test_ask_user_question_fails_when_cancelled():
    """Verify ask user question fails when cancelled."""
    runtime = ConversationRuntime()
    ctx = SimpleNamespace(
        request_user_input=lambda title, prompt, **kw: runtime.request_input("chat", title, prompt, **kw),
        runtime=runtime,
        session_key="chat",
    )
    seen = []
    worker = threading.Thread(target=lambda: seen.append(AskUserQuestion().run(ctx, question="Value?")), daemon=True)

    worker.start()
    deadline = time.time() + 5
    while time.time() < deadline and ("chat" not in runtime.sessions or runtime.sessions["chat"].cs.phase != PHASE_APPROVING_REQUEST):
        time.sleep(0.01)
    assert runtime.handle_action("chat", "cancel").ok
    worker.join(timeout=5)

    assert seen and not seen[0].success
    assert "cancelled" in seen[0].error.lower()


def test_ask_user_question_uses_form_display_type_hints():
    """Verify ask user question uses form display type hints."""
    requests = []
    ctx = SimpleNamespace(
        request_user_input=lambda title, prompt, **kw: requests.append((title, prompt, kw)) or SimpleNamespace(
            id="req",
            value={"key": "value"},
            metadata={},
            wait=lambda timeout=None: True,
        )
    )

    result = AskUserQuestion().run(ctx, question="Send settings.", type="object")

    assert result.success
    assert 'Send a JSON object, for example: {"key": "value"}.' in requests[0][1]


def test_enum_user_input_rejects_invalid_then_accepts_valid():
    """Verify enum user input rejects invalid then accepts valid."""
    runtime = ConversationRuntime()
    req = runtime.request_input("chat", "Pick", "Pick one", type="string", enum=["a", "b"])

    bad = runtime.handle_action("chat", "answer_approval", {"value": "c"})
    good = runtime.handle_action("chat", "answer_approval", {"value": "b"})

    assert not bad.ok
    assert good.ok
    assert req.value == "b"


def test_inject_user_message_appends_without_driving_agent_turn():
    """Verify inject user message appends without driving agent turn."""
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
    """Verify new conversation action uses reset lifecycle."""
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
    """Verify new command starts default main conversation."""
    db = FakeConversationDB()
    runtime = ConversationRuntime(db=db)
    result = NewCommand().run({}, SimpleNamespace(db=db, runtime=runtime, session_key="chat"))

    assert result == "Started new conversation #1 under 'Main'.\nAgent: default"
    assert db.conversations[1]["category"] is None
    assert runtime.sessions["chat"].conversation_id == 1
    assert runtime.sessions["chat"].profile_override == "default"


def test_conversations_command_changes_category_after_selection():
    """Verify conversations command changes category after selection."""
    db = FakeConversationDB()
    cid = db.create_conversation("Old", category=None)
    result = ConversationsCommand().run({"category": "Main", "conversation_id": str(cid), "action": "Change category", "target_category": "Projects"}, SimpleNamespace(db=db, runtime=ConversationRuntime(db=db), session_key="chat"))

    assert result == "Conversation #1 moved to 'Projects'."
    assert db.conversations[cid]["category"] == "Projects"


def test_conversations_command_form_manages_existing_conversations_only():
    """Verify conversations command form manages existing conversations only."""
    db = FakeConversationDB()
    cid = db.create_conversation("Old", category=None)
    cmd = ConversationsCommand()

    top = cmd.form({}, SimpleNamespace(db=db))[0]
    steps = cmd.form({"category": "Main", "conversation_id": str(cid)}, SimpleNamespace(db=db))

    assert "➕ New conversation" not in top.enum
    assert "Change category" in steps[-1].enum


def test_tool_budget_gets_one_final_model_summary():
    """Verify tool budget gets one final model summary."""
    schema = {"function": {"name": "echo", "parameters": {"type": "object", "properties": {}, "required": []}}}
    registry = FakeToolRegistry([schema], {"echo": FakeToolResult(success=True, data={}, llm_summary="ok", attachment_paths=[])})
    cs = make_cs(tools={"echo": CallableSpec("echo", handler=lambda _cs, _actor, args: registry.call("echo", **args))})
    cs.set_priority("agent")
    llm = FakeLLM([FakeResponse.tool([{"id": f"tc{i}", "name": "echo", "arguments": "{}"}]) for i in range(24)] + [FakeResponse.text("here is what I found")])

    final_text, new_messages, _ = ConversationLoop(llm, registry, {}, "").drive(cs, "agent", [{"role": "user", "content": "loop"}])

    assert final_text == "here is what I found"
    assert new_messages[-1]["content"] == "here is what I found"


def test_busy_runtime_rejects_text_but_accepts_cancel_signal():
    """Verify busy runtime rejects text but accepts cancel signal."""
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
    """Verify form path collects args then runs command."""
    captured: dict = {}

    def handler(_cs, _actor, args):
        """Handle handler."""
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
    """Verify invalid integer form input keeps same field active."""
    cmd = CallableSpec("count", handler=lambda *_: "ok", form=[FormStep("n", required=True, type="integer")])
    cs = make_cs(commands={"count": cmd})

    assert create_action(cs, "call_command", {"name": "count"}, "user").enact().ok
    result = create_action(cs, "submit_form_text", "nope", "user").enact()

    assert not result.ok
    assert result.error.code == "invalid_input"
    assert cs.frame.step.name == "n"


def test_new_command_supersedes_pending_form():
    """Verify new command supersedes pending form."""
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
    """Verify tool optional args do not start forms."""
    captured: dict = {}

    def handler(_cs, _actor, args):
        """Handle handler."""
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
    """Verify optional prompted args can be skipped."""
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
    """Verify tools command prompts optional schema args before calling tool."""
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
    """Verify cancel and skip confirm when no handler message."""
    cs = make_cs(commands={
        "need": CallableSpec("need", handler=lambda *_: None, form=[FormStep("x", required=True)]),
        "maybe": CallableSpec("maybe", handler=lambda *_: None, form=[FormStep("x", required=False, default="", prompt_when_missing=True)]),
    })

    assert create_action(cs, "call_command", {"name": "need"}, "user").enact().ok
    assert create_action(cs, "cancel", None, "user").enact().message == "Cancelled."
    assert create_action(cs, "call_command", {"name": "maybe"}, "user").enact().ok
    assert create_action(cs, "skip_form", None, "user").enact().message == "Skipped."


def test_dynamic_form_steps_follow_collected_args():
    """Verify dynamic form steps follow collected args."""
    captured: dict = {}

    def form(args, _cs):
        """Handle form."""
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
    """Verify back form revisits static step and uses replacement value."""
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
    """Verify back form follows dynamic form factory."""
    def form(args, _cs):
        """Handle form."""
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
    """Verify back form can undo skipped optional before next field."""
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
    """Verify back form at first step fails without mutating prompt."""
    cs = make_cs(commands={"hello": CallableSpec("hello", handler=lambda *_: "ok", form=[FormStep("name", required=True)])})

    assert create_action(cs, "call_command", {"name": "hello", "args": {}}, "user").enact().ok
    result = create_action(cs, "back_form", None, "user").enact()

    assert not result.ok
    assert result.error.message == "Nothing to go back to."
    assert cs.frame.step.name == "name"


def test_back_form_does_not_remove_prefilled_command_args():
    """Verify back form does not remove prefilled command args."""
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
    """Verify legal actions in phase returns registered types."""
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
