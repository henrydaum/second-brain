"""Tests for the agent-turn driver (``runtime.conversation_loop``).

The loop is the heart of the kernel: it asks the LLM, translates the response
into typed ``send_text`` / ``call_tool`` / ``end_turn`` actions, dispatches each
through ``cs.enact()``, and records provider-shaped history. These tests drive a
real ``ConversationState`` with a fake LLM (no network) and assert the resulting
transcript and turn hand-off.
"""

from types import SimpleNamespace

# Import the state_machine package before runtime.conversation_loop to settle
# the package-init circular import (state_machine/__init__ pulls in the loop).
from state_machine.conversation import CallableSpec, ConversationState, Participant
from state_machine.conversation_phases import BASE_PHASE

from attachments.attachment import Attachment
from plugins.BaseTool import ToolResult
from runtime.conversation_loop import ConversationLoop
from runtime.hooks import HookRegistry


def _response(content="", tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        is_error=False,
        prompt_tokens=0,
    )


class _FakeLLM:
    """Returns queued responses, one per ``chat_with_tools`` call."""

    context_size = 0  # disables proactive compaction

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.attachments = []

    def chat_with_tools(self, messages, tools, attachments=None):
        self.calls.append(messages)
        self.attachments.append(attachments)
        return self._responses.pop(0)


class _FakeRegistry:
    max_tool_calls = 5
    tools = {}  # empty -> no per-tool budget enforcement in the test

    def __init__(self, schemas):
        self._schemas = schemas

    def get_all_schemas(self):
        return self._schemas


def _agent_state(tools=None, cache=None):
    base_cache = {"session_key": "chat", "agent_scoped_tool_names": list((tools or {}).keys())}
    base_cache.update(cache or {})
    return ConversationState(
        [Participant("user", "user"), Participant("agent", "agent", tools=tools or {})],
        "agent",
        BASE_PHASE,
        base_cache,
    )


def _loop(llm, registry):
    return ConversationLoop(llm, registry, {"tool_timeout": 10}, "You are a helpful agent.")


def test_text_only_turn_records_reply_and_hands_back_to_user():
    cs = _agent_state()
    llm = _FakeLLM([_response(content="Hello there!")])
    loop = _loop(llm, _FakeRegistry([]))
    history = [{"role": "user", "content": "hi"}]

    reply, new_messages, attachments = loop.drive(cs, "agent", history)

    assert reply == "Hello there!"
    assert {"role": "assistant", "content": "Hello there!"} in new_messages
    assert attachments == []
    # The turn is finished: priority is handed back to the user.
    assert cs.turn_priority == "user"


def test_tool_call_then_text_produces_full_transcript():
    captured = {}

    def echo_handler(cs, actor, args):
        captured["args"] = args
        return ToolResult(llm_summary="echoed: ping", data={"echoed": "ping"})

    tools = {"echo": CallableSpec("echo", handler=echo_handler)}
    cs = _agent_state(tools=tools)

    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}
    llm = _FakeLLM([
        _response(content="", tool_calls=[{"id": "call_1", "name": "echo", "arguments": '{"text": "ping"}'}]),
        _response(content="All done."),
    ])
    loop = _loop(llm, _FakeRegistry([schema]))
    history = [{"role": "user", "content": "please echo"}]

    reply, new_messages, _ = loop.drive(cs, "agent", history)

    assert captured["args"] == {"text": "ping"}
    assert reply == "All done."

    roles = [(m["role"], m.get("content")) for m in new_messages]
    # assistant(tool_calls) -> tool result -> assistant(final text)
    assert ("tool", "echoed: ping") in roles
    assert ("assistant", "All done.") in roles
    tool_msg = next(m for m in new_messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["name"] == "echo"
    assert cs.turn_priority == "user"


def test_tool_failure_is_surfaced_to_the_model_as_error():
    def boom_handler(cs, actor, args):
        return ToolResult(success=False, error="kaboom")

    tools = {"boom": CallableSpec("boom", handler=boom_handler)}
    cs = _agent_state(tools=tools)

    schema = {"type": "function", "function": {"name": "boom", "parameters": {}}}
    llm = _FakeLLM([
        _response(content="", tool_calls=[{"id": "c1", "name": "boom", "arguments": "{}"}]),
        _response(content="I hit an error."),
    ])
    loop = _loop(llm, _FakeRegistry([schema]))

    reply, new_messages, _ = loop.drive(cs, "agent", [{"role": "user", "content": "go"}])

    tool_msg = next(m for m in new_messages if m["role"] == "tool")
    assert "kaboom" in tool_msg["content"]
    assert reply == "I hit an error."


def test_unknown_tool_name_feeds_error_back_instead_of_ending_turn():
    """A hallucinated tool name must not end the turn: the unknown-tool error
    goes into history as the tool result and the LLM gets another chance to
    correct course (mirrors the `pip show openpyxl` incident)."""
    tools = {"echo": CallableSpec("echo", handler=lambda cs, actor, args: ToolResult(llm_summary="ok"))}
    cs = _agent_state(tools=tools)

    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}
    llm = _FakeLLM([
        _response(content="Checking the dep.",
                  tool_calls=[{"id": "c1", "name": "pip show openpyxl", "arguments": "{}"}]),
        _response(content="", tool_calls=[{"id": "c2", "name": "echo", "arguments": "{}"}]),
        _response(content="Recovered with the real tool."),
    ])
    loop = _loop(llm, _FakeRegistry([schema]))

    reply, new_messages, _ = loop.drive(cs, "agent", [{"role": "user", "content": "go"}])

    # The turn survived the bogus call and finished with real text.
    assert reply == "Recovered with the real tool."
    assert cs.turn_priority == "user"
    # The transcript stays provider-valid: the bogus call's assistant row and
    # a matching tool-result row carrying the error the LLM can read.
    error_msg = next(m for m in new_messages if m["role"] == "tool" and m["tool_call_id"] == "c1")
    assert "Unknown tool" in error_msg["content"]
    assert "pip show openpyxl" in error_msg["content"]
    # The second LLM call saw the error before answering.
    assert len(llm.calls) == 3


def test_agent_missing_required_args_fails_fast_instead_of_form():
    """An agent tool call omitting a required argument must get an immediate
    readable error — never push a form phase frame the model can't see."""
    from state_machine.forms import schema_to_form_steps

    ran = []
    steps = schema_to_form_steps({"properties": {"sql": {"type": "string"}}, "required": ["sql"]})
    tools = {"sql_query": CallableSpec("sql_query", handler=lambda cs, actor, args: ran.append(args), form=steps)}
    cs = _agent_state(tools=tools)

    schema = {"type": "function", "function": {"name": "sql_query", "parameters": {"properties": {"sql": {"type": "string"}}, "required": ["sql"]}}}
    llm = _FakeLLM([
        _response(content="", tool_calls=[{"id": "c1", "name": "sql_query", "arguments": "{}"}]),
        _response(content="I forgot the sql argument."),
    ])
    loop = _loop(llm, _FakeRegistry([schema]))

    reply, new_messages, _ = loop.drive(cs, "agent", [{"role": "user", "content": "go"}])

    assert ran == []  # the handler never ran with missing args
    assert reply == "I forgot the sql argument."
    tool_msg = next(m for m in new_messages if m["role"] == "tool")
    assert "Missing required argument" in tool_msg["content"]
    assert "sql" in tool_msg["content"]
    # No phantom form frame left behind.
    assert cs.phase == BASE_PHASE
    assert not cs.cache.get("phases")


def test_invalid_json_arguments_are_refused_without_enacting():
    """Unparseable tool-call JSON is refused by the loop itself: the handler
    never runs, and the LLM reads the parse error as the tool result."""
    ran = []
    tools = {"echo": CallableSpec("echo", handler=lambda cs, actor, args: ran.append(args))}
    cs = _agent_state(tools=tools)

    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}
    llm = _FakeLLM([
        _response(content="", tool_calls=[{"id": "c1", "name": "echo", "arguments": "{not json"}]),
        _response(content="Let me fix that JSON."),
    ])
    loop = _loop(llm, _FakeRegistry([schema]))

    reply, new_messages, _ = loop.drive(cs, "agent", [{"role": "user", "content": "go"}])

    assert ran == []
    assert reply == "Let me fix that JSON."
    tool_msg = next(m for m in new_messages if m["role"] == "tool")
    assert "Invalid JSON" in tool_msg["content"]


def test_session_message_emitted_for_every_transcript_row():
    """_record is the single SESSION_MESSAGE source: a tool-call turn feeds
    the bus the assistant tool-call row, the tool result, and the final text."""
    from events.event_bus import bus
    from events.event_channels import SESSION_MESSAGE

    seen = []
    unsub = bus.subscribe(SESSION_MESSAGE, seen.append)
    try:
        tools = {"echo": CallableSpec("echo", handler=lambda cs, actor, args: ToolResult(llm_summary="echoed"))}
        cs = _agent_state(tools=tools)
        schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}
        llm = _FakeLLM([
            _response(content="", tool_calls=[{"id": "c1", "name": "echo", "arguments": "{}"}]),
            _response(content="Done."),
        ])
        loop = ConversationLoop(llm, _FakeRegistry([schema]), {"tool_timeout": 10}, "prompt",
                                session_key="chat")

        loop.drive(cs, "agent", [{"role": "user", "content": "go"}])
    finally:
        unsub()

    assert [(e["role"], e["actor_id"]) for e in seen] == [
        ("assistant", "agent"),  # tool-call row
        ("tool", "agent"),       # tool result row
        ("assistant", "agent"),  # final text row
    ]
    tool_event = seen[1]
    assert tool_event["name"] == "echo"
    assert tool_event["tool_call_id"] == "c1"
    assert tool_event["content"] == "echoed"
    assert seen[0]["tool_calls"][0]["function"]["name"] == "echo"
    assert seen[2]["content"] == "Done."
    assert all(e["session_key"] == "chat" for e in seen)


def test_llm_call_events_bracket_each_request():
    from events.event_bus import bus
    from events.event_channels import AGENT_LLM_CALL_FINISHED, AGENT_LLM_CALL_STARTED

    events = []
    unsubs = [
        bus.subscribe(AGENT_LLM_CALL_STARTED, lambda p: events.append(("started", p))),
        bus.subscribe(AGENT_LLM_CALL_FINISHED, lambda p: events.append(("finished", p))),
    ]
    try:
        cs = _agent_state()
        llm = _FakeLLM([_response(content="Hello!")])
        loop = ConversationLoop(llm, _FakeRegistry([]), {"tool_timeout": 10}, "prompt",
                                session_key="chat")
        loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])
    finally:
        for u in unsubs:
            u()

    assert [kind for kind, _ in events] == ["started", "finished"]
    finished = events[1][1]
    assert finished["ok"] is True
    assert finished["has_tool_calls"] is False
    assert finished["session_key"] == "chat"


def test_compaction_emits_session_compacted_event():
    from events.event_bus import bus
    from events.event_channels import SESSION_COMPACTED

    class _Compactor:
        loaded = True

        def compact(self, **kwargs):
            return "Earlier summary."

    seen = []
    unsub = bus.subscribe(SESSION_COMPACTED, seen.append)
    try:
        runtime = SimpleNamespace(services={"compactor": _Compactor()}, sessions={})
        loop = ConversationLoop(_FakeLLM([]), _FakeRegistry([]), {}, "prompt",
                                runtime=runtime, session_key="chat")
        history = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        loop._compact(history)
    finally:
        unsub()

    [event] = seen
    assert event["session_key"] == "chat"
    assert event["messages_compacted"] == 3
    assert event["summary"] == "Earlier summary."


def test_empty_response_after_tool_error_is_retried_with_nudge():
    """A response that cleans to empty (e.g. a weak model emitting only a
    think block after a tool error) is not a final answer: the loop nudges
    once with an ephemeral message and takes the retry's text."""
    def failing_sql(cs, actor, args):
        return ToolResult(success=False, error="no such column: text")

    tools = {"sql_query": CallableSpec("sql_query", handler=failing_sql)}
    cs = _agent_state(tools=tools)
    schema = {"type": "function", "function": {"name": "sql_query", "parameters": {}}}
    llm = _FakeLLM([
        _response(content="", tool_calls=[{"id": "c1", "name": "sql_query", "arguments": "{}"}]),
        _response(content="<think>hmm</think>"),  # cleans to empty
        _response(content="The query failed: no such column."),
    ])
    loop = _loop(llm, _FakeRegistry([schema]))

    reply, new_messages, _ = loop.drive(cs, "agent", [{"role": "user", "content": "query it"}])

    assert reply == "The query failed: no such column."
    assert cs.turn_priority == "user"
    # The nudge was ephemeral: it reached the LLM but never entered history.
    assert any("response was empty" in (m.get("content") or "") for m in llm.calls[2])
    assert not any("response was empty" in (m.get("content") or "") for m in new_messages)


def test_persistently_empty_response_still_ends_the_turn():
    """If the nudge retry is also empty, the loop gives up cleanly — one
    retry only, empty final text, priority handed back to the user."""
    cs = _agent_state()
    llm = _FakeLLM([_response(content=""), _response(content="")])
    loop = _loop(llm, _FakeRegistry([]))

    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == ""
    assert len(llm.calls) == 2  # exactly one nudge retry, no loop
    assert cs.turn_priority == "user"


class _StreamingLLM(_FakeLLM):
    """Streams each queued response's content in 4-char fragments, then
    returns the same LLMResponse shape as the blocking call."""

    supports_streaming = True

    def chat_with_tools_streaming(self, messages, tools, attachments=None, on_delta=None):
        response = self.chat_with_tools(messages, tools, attachments)
        content = response.content or ""
        for i in range(0, len(content), 4):
            if on_delta and not on_delta(content[i:i + 4]):
                break
        return response


def test_streaming_emits_deltas_and_clean_final_text():
    events = []
    cs = _agent_state()
    # <think> tokens are filtered out of the streamed deltas (even split
    # across 4-char fragments), and the done event carries the CLEANED text —
    # the dedup key must match what the whole-message path delivers.
    llm = _StreamingLLM([_response(content="<think>hmm</think>Hello there!")])
    loop = ConversationLoop(llm, _FakeRegistry([]), {"tool_timeout": 10}, "prompt",
                            on_delta=events.append)

    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "Hello there!"
    deltas = [e for e in events if not e["done"]]
    assert "".join(e["delta"] for e in deltas) == "Hello there!"
    [done] = [e for e in events if e["done"]]
    assert done["aborted"] is False
    assert done["final_text"] == "Hello there!"
    assert done["kind"] == "final"
    assert {e["stream_id"] for e in events} == {done["stream_id"]}
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


def test_streaming_narration_done_precedes_tool_events():
    timeline = []
    tools = {"echo": CallableSpec("echo", handler=lambda cs, actor, args: ToolResult(llm_summary="ok"))}
    cs = _agent_state(tools=tools)
    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}
    llm = _StreamingLLM([
        _response(content="Let me check.", tool_calls=[{"id": "c1", "name": "echo", "arguments": "{}"}]),
        _response(content="Done."),
    ])
    loop = ConversationLoop(
        llm, _FakeRegistry([schema]), {"tool_timeout": 10}, "prompt",
        on_tool_start=lambda *a, **k: timeline.append(("tool_start",)),
        on_delta=lambda p: timeline.append(("done", p["kind"], p["final_text"]) if p["done"] else ("delta",)),
    )

    loop.drive(cs, "agent", [{"role": "user", "content": "go"}])

    dones = [t for t in timeline if t[0] == "done"]
    assert dones == [("done", "narration", "Let me check."), ("done", "final", "Done.")]
    # Narration closes before the tool call starts.
    assert timeline.index(dones[0]) < timeline.index(("tool_start",))


def test_cancel_mid_stream_stops_backend_and_skips_send_text():
    import threading

    cancel = threading.Event()
    events = []
    wrapper_returns = []

    class _CancellingLLM(_StreamingLLM):
        def chat_with_tools_streaming(self, messages, tools, attachments=None, on_delta=None):
            response = self.chat_with_tools(messages, tools, attachments)
            wrapper_returns.append(on_delta("partial "))
            cancel.set()
            wrapper_returns.append(on_delta("text"))
            return response

    cs = _agent_state()
    llm = _CancellingLLM([_response(content="partial text and more")])
    loop = ConversationLoop(llm, _FakeRegistry([]), {"tool_timeout": 10}, "prompt",
                            cancel_event=cancel, on_delta=events.append)

    _, new_messages, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert wrapper_returns == [True, False]  # abort signalled to the backend
    # The cancelled partial never entered the transcript.
    assert not any(m.get("role") == "assistant" for m in new_messages)
    assert events[-1]["done"] is True  # stream was closed


def test_stream_error_emits_aborted_done_then_non_streaming_retry():
    events = []

    class _OverflowLLM:
        context_size = 0
        supports_streaming = True

        def chat_with_tools_streaming(self, messages, tools, attachments=None, on_delta=None):
            on_delta("par")
            raise RuntimeError("prompt tokens exceed model token limit")

        def chat_with_tools(self, messages, tools, attachments=None):
            return _response(content="Recovered.")

    cs = _agent_state()
    loop = ConversationLoop(_OverflowLLM(), _FakeRegistry([]), {"tool_timeout": 10}, "prompt",
                            on_delta=events.append)
    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    reply, _, _ = loop.drive(cs, "agent", history)

    assert reply == "Recovered."
    dones = [e for e in events if e["done"]]
    # One stream: the failed call, closed aborted. The retry answer arrives
    # whole (no deltas), so no second done and no stale dedup entry.
    assert len(dones) == 1 and dones[0]["aborted"] is True


def test_no_on_delta_means_blocking_call_even_with_streaming_backend():
    class _NeverStream(_StreamingLLM):
        def chat_with_tools_streaming(self, *a, **k):
            raise AssertionError("streaming path must not be used")

    cs = _agent_state()
    llm = _NeverStream([_response(content="Plain.")])
    loop = _loop(llm, _FakeRegistry([]))

    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "Plain."


def test_queued_message_is_absorbed_mid_turn():
    """A user message queued while the turn runs is drained at the next loop
    boundary as a real user history row, and the LLM is asked again instead
    of the turn ending on the earlier final text."""
    import threading

    session = SimpleNamespace(key="chat", lock=threading.RLock(), pending_user_messages=[])
    runtime = SimpleNamespace(sessions={"chat": session})

    class _QueueingLLM(_FakeLLM):
        def chat_with_tools(self, messages, tools, attachments=None):
            if not self.calls:  # first call: simulate a mid-turn user message
                session.pending_user_messages.append("wait, also do X")
            return super().chat_with_tools(messages, tools, attachments)

    cs = _agent_state()
    llm = _QueueingLLM([_response(content="First answer."), _response(content="Second answer.")])
    loop = ConversationLoop(llm, _FakeRegistry([]), {"tool_timeout": 10}, "prompt",
                            runtime=runtime, session_key="chat")
    history = [{"role": "user", "content": "hi"}]

    reply, new_messages, _ = loop.drive(cs, "agent", history)

    assert reply == "Second answer."
    roles = [(m["role"], m.get("content")) for m in new_messages]
    assert roles.index(("assistant", "First answer.")) \
        < roles.index(("user", "wait, also do X")) \
        < roles.index(("assistant", "Second answer."))
    assert session.pending_user_messages == []
    assert cs.turn_priority == "user"
    # The second LLM call saw the queued message in its transcript.
    assert any(m.get("content") == "wait, also do X" for m in llm.calls[1])


def test_tool_can_stage_attachment_for_followup_model_call():
    runtime = SimpleNamespace(sessions={}, hooks=HookRegistry())
    runtime.sessions["chat"] = SimpleNamespace(key="chat")

    def inspect_handler(cs, actor, args):
        runtime.hooks.stage_attachment(
            runtime.sessions["chat"],
            Attachment("C:/tmp/photo.png", ".png", "photo.png", "image"),
        )
        return ToolResult(llm_summary="Attached photo.png for inspection.")

    def noop_handler(cs, actor, args):
        return ToolResult(llm_summary="No-op done.")

    tools = {
        "inspect": CallableSpec("inspect", handler=inspect_handler),
        "noop": CallableSpec("noop", handler=noop_handler),
    }
    cs = _agent_state(tools=tools)
    llm = _FakeLLM([
        _response(tool_calls=[
            {"id": "c1", "name": "inspect", "arguments": "{}"},
            {"id": "c2", "name": "noop", "arguments": "{}"},
        ]),
        _response(content="I can see it now."),
    ])
    loop = ConversationLoop(
        llm,
        _FakeRegistry([
            {"type": "function", "function": {"name": "inspect", "parameters": {}}},
            {"type": "function", "function": {"name": "noop", "parameters": {}}},
        ]),
        {"tool_timeout": 10},
        "prompt",
        runtime=runtime,
        session_key="chat",
    )

    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "inspect this"}])

    assert reply == "I can see it now."
    assert not llm.attachments[0]
    assert [a.file_name for a in llm.attachments[1]] == ["photo.png"]


def test_compaction_uses_compactor_service_directly():
    class _Compactor:
        loaded = True

        def __init__(self):
            self.calls = []

        def compact(self, **kwargs):
            self.calls.append(kwargs)
            return "Earlier summary."

    notices = []
    compactor = _Compactor()
    runtime = SimpleNamespace(services={"compactor": compactor}, sessions={})
    loop = ConversationLoop(
        _FakeLLM([]),
        _FakeRegistry([]),
        {},
        "prompt",
        on_notice=notices.append,
        runtime=runtime,
        session_key="chat",
    )
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    loop._compact(history)

    assert compactor.calls[0]["runtime"] is runtime
    assert compactor.calls[0]["session_key"] == "chat"
    assert history[0]["content"].startswith("[Conversation summary from earlier]")
    assert history[0]["content"].endswith("Earlier summary.")
    # The synthesized turn carries the compaction ground rules: the full
    # transcript survives in the DB, and unremembered turns must not be denied.
    assert "conversation_messages" in history[0]["content"]
    assert "never deny" in history[0]["content"]
    assert history[1]["content"] == "Understood - I have the earlier context."
    assert notices == ["Compacting conversation...", "Compacted 3 messages."]
