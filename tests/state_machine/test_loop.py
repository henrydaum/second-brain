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
from types import SimpleNamespace

import pytest

from agent.tool_registry import ToolRegistry
from plugins.BaseTool import BaseTool, ToolResult
from state_machine.action_map import create_action
from state_machine.conversation import CallableSpec, ConversationState, FormStep, Participant
from runtime.conversation_loop import ConversationLoop
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST
from state_machine.serialization import latest_state, save_state_marker
from runtime.conversation_runtime import ConversationRuntime
from events.event_channels import COMMAND_CALL_FINISHED, SESSION_CLOSED, SESSION_CREATED, SESSION_TURN_COMPLETED, TOOL_CALL_FINISHED, TOOL_CALL_STARTED
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

    def create_conversation(self, title="New conversation", kind="user"):
        cid = self.next_id; self.next_id += 1
        self.conversations[cid] = {"id": cid, "title": title, "kind": kind}
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
    conv_id = db.create_conversation("Cron")
    db.save_message(conv_id, "user", "earlier")
    events = []
    runtime = ConversationRuntime(
        db=db,
        services={"llm": FakeLLM([FakeResponse.text("done")])},
        tool_registry=FakeToolRegistry(),
        emit_event=lambda channel, payload: events.append((channel, payload)),
    )

    runtime.load_conversation("job", conv_id)
    result = runtime.iterate_agent_turn("job", "wake up")

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
    runtime._refresh_session_specs(session)

    assert session.history == [{"role": "user", "content": "earlier"}]
    assert session.profile_override == "builder"
    assert runtime._profile_for_session(session) == "builder"
    assert result.messages == ["Loaded conversation: Builder chat\nAgent: builder\nSwitched agent: default -> builder"]
    assert [name for name, _ in events[-2:]] == [SESSION_CLOSED, SESSION_CREATED]
    assert events[-1][1]["agent_profile"] == "builder"


def test_set_conversation_notification_mode_updates_stored_marker():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    save_state_marker(db, conv_id, {"active_agent_profile": "builder"})
    runtime = ConversationRuntime(db=db)

    assert runtime.set_conversation_notification_mode(conv_id, "IMPORTANT") == "important"
    assert latest_state(db.get_conversation_messages(conv_id))["notification_mode"] == "important"
    assert runtime.load_conversation("job", conv_id).notification_mode == "important"


def test_set_conversation_notification_mode_updates_live_session():
    db = FakeConversationDB()
    conv_id = db.create_conversation("Cron")
    runtime = ConversationRuntime(db=db)
    session = runtime.load_conversation("job", conv_id)

    assert any(getattr(t, "name", None) == "notify" for t in session.extra_tool_instances)
    assert runtime.set_conversation_notification_mode(conv_id, "off") == "off"
    assert session.notification_mode == latest_state(db.get_conversation_messages(conv_id))["notification_mode"] == "off"
    assert not any(getattr(t, "name", None) == "notify" for t in session.extra_tool_instances)


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
    assert "cancel" in form
