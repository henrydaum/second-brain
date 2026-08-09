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
from state_machine.conversation import CallableSpec
from state_machine.conversation_phases import BASE_PHASE

from attachments.attachment import Attachment
from plugins.native.tool import ToolResult
from runtime.conversation_loop import ConversationLoop
from runtime.hooks import HookRegistry


from tests.support import FakeLLM as _FakeLLM
from tests.support import FakeRegistry as _FakeRegistry
from tests.support import agent_state as _agent_state
from tests.support import response as _response


def _loop(llm, registry):
    return ConversationLoop(llm, registry, {}, "You are a helpful agent.")


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
        loop = ConversationLoop(llm, _FakeRegistry([schema]), {}, "prompt",
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
        loop = ConversationLoop(llm, _FakeRegistry([]), {}, "prompt",
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
    returns the same LLMResponse shape as the blocking call.

    One method for both paths, like a real backend: ``request.stream`` says
    whether the caller wants deltas. There were two (``chat_with_tools`` and
    ``chat_with_tools_streaming``) while a backend could be an in-process
    service, and the adapter chose between them.
    """

    supports_streaming = True

    def chat(self, request, on_delta=None):
        response = super().chat(request)
        if not (request.stream and on_delta):
            return response
        content = response.content or ""
        for i in range(0, len(content), 4):
            if not on_delta(content[i:i + 4]):
                break
        return response


def test_streaming_emits_deltas_and_clean_final_text():
    events = []
    cs = _agent_state()
    # <think> tokens are filtered out of the streamed deltas (even split
    # across 4-char fragments), and the done event carries the CLEANED text —
    # the dedup key must match what the whole-message path delivers.
    llm = _StreamingLLM([_response(content="<think>hmm</think>Hello there!")])
    loop = ConversationLoop(llm, _FakeRegistry([]), {}, "prompt",
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
        llm, _FakeRegistry([schema]), {}, "prompt",
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
        def chat(self, request, on_delta=None):
            response = _FakeLLM.chat(self, request)
            wrapper_returns.append(on_delta("partial "))
            cancel.set()
            wrapper_returns.append(on_delta("text"))
            return response

    cs = _agent_state()
    llm = _CancellingLLM([_response(content="partial text and more")])
    loop = ConversationLoop(llm, _FakeRegistry([]), {}, "prompt",
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
        loaded = True
        name = "overflow"

        def chat(self, request, on_delta=None):
            if request.stream and on_delta:
                on_delta("par")
                raise RuntimeError("prompt tokens exceed model token limit")
            return _response(content="Recovered.")

    cs = _agent_state()
    loop = ConversationLoop(_OverflowLLM(), _FakeRegistry([]), {}, "prompt",
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
        def chat(self, request, on_delta=None):
            if request.stream and on_delta:
                raise AssertionError("streaming path must not be used")
            return _FakeLLM.chat(self, request)

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
        def chat(self, request, on_delta=None):
            if not self.calls:  # first call: simulate a mid-turn user message
                session.pending_user_messages.append("wait, also do X")
            return super().chat(request, on_delta)

    cs = _agent_state()
    llm = _QueueingLLM([_response(content="First answer."), _response(content="Second answer.")])
    loop = ConversationLoop(llm, _FakeRegistry([]), {}, "prompt",
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


def test_tool_can_stage_attachment_for_followup_llm_call():
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
        {},
        "prompt",
        runtime=runtime,
        session_key="chat",
    )

    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "inspect this"}])

    assert reply == "I can see it now."
    assert not llm.attachments[0]
    # Plain ``{path, modality, file_name}`` dicts, which is what an
    # ``LLMRequest`` carries. They used to arrive as ``Attachment`` objects,
    # because the adapter rebuilt an ``AttachmentBundle`` for the old
    # in-process contract; nothing rebuilds them and no backend wants them to.
    assert [a["file_name"] for a in llm.attachments[1]] == ["photo.png"]


def test_the_add_attachment_request_stages_onto_the_live_session(tmp_path):
    """The Request half of the test above, which had no way to be written.

    Staging was reachable only from in-process native code: the runtime method
    was deleted for having no callers, and no Request replaced it. A sandboxed
    tool asking to show the model a file goes handler -> runtime -> hooks ->
    session, and this drives that whole chain.
    """
    from runtime.conversation_runtime import ConversationRuntime
    from sandbox.handlers.kernel import _session_add_attachment

    photo = tmp_path / "photo.png"
    photo.write_bytes(b"\x89PNG\r\n\x1a\n")

    runtime = SimpleNamespace(sessions={"chat": SimpleNamespace(key="chat")},
                              hooks=HookRegistry())
    runtime.add_turn_attachment = (
        lambda key, att: ConversationRuntime.add_turn_attachment(runtime, key, att))
    ctx = SimpleNamespace(runtime=runtime, session_key="chat", config={})

    assert _session_add_attachment(ctx, {"path": str(photo)}).ok

    staged = runtime.sessions["chat"].staged_attachments
    assert [a.file_name for a in staged] == ["photo.png"]
    # Resolved with no parser installed, off the registry's native defaults.
    assert staged[0].modality == "image"


def test_staging_into_a_dead_session_fails_rather_than_answering_false(tmp_path):
    """A caller ignoring the return value must not silently lose the file.

    ``False`` reads to the agent as a model that looked and saw nothing.
    """
    from runtime.conversation_runtime import ConversationRuntime
    from sandbox.handlers.kernel import _session_add_attachment

    photo = tmp_path / "photo.png"
    photo.write_bytes(b"\x89PNG\r\n\x1a\n")

    runtime = SimpleNamespace(sessions={}, hooks=HookRegistry())
    runtime.add_turn_attachment = (
        lambda key, att: ConversationRuntime.add_turn_attachment(runtime, key, att))
    ctx = SimpleNamespace(runtime=runtime, session_key="gone", config={})

    result = _session_add_attachment(ctx, {"path": str(photo)})

    assert not result.ok and not result.denied


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

    assert compactor.calls[0] == {
        "session_key": "chat",
        "transcript": "USER: one\nASSISTANT: two\nUSER: three",
    }
    assert history[0]["content"].startswith("[Conversation summary from earlier]")
    assert history[0]["content"].endswith("Earlier summary.")
    # The synthesized turn carries the compaction ground rules: the full
    # transcript survives in the DB, and unremembered turns must not be denied.
    assert "conversation_messages" in history[0]["content"]
    assert "never deny" in history[0]["content"]
    assert history[1]["content"] == "Understood - I have the earlier context."
    assert notices == ["Compacting conversation...", "Compacted 3 messages."]


# ──────────────────────────────────────────────────────────────────────
# The compaction layer (the kernel's context-safety escort)
# ──────────────────────────────────────────────────────────────────────

_OVERFLOW_ERROR = "This model's maximum context length is 8192 tokens."


class _OverflowLLM(_FakeLLM):
    """Raises a context-limit error for the first ``overflows`` calls."""

    def __init__(self, responses, overflows=1):
        super().__init__(responses)
        self.overflows = overflows

    def chat(self, request, on_delta=None):
        if self.overflows > 0:
            self.overflows -= 1
            self.calls.append(list(request.messages))
            raise RuntimeError(_OVERFLOW_ERROR)
        return super().chat(request, on_delta)


class _CountingCompactor:
    loaded = True

    def __init__(self):
        self.calls = 0

    def compact(self, **kwargs):
        self.calls += 1
        return "Earlier summary."


def _overflow_rig(llm, compactor=None):
    from runtime.hooks import HookRegistry as _HR
    session = SimpleNamespace(restart_turn=False)
    runtime = SimpleNamespace(
        sessions={"chat": session},
        hooks=_HR(),
        services={"compactor": compactor} if compactor else {},
    )
    loop = ConversationLoop(llm, _FakeRegistry([]), {}, "prompt",
                            runtime=runtime, session_key="chat")
    return loop, runtime


def test_context_overflow_compacts_and_retries():
    """A context-limit failure is caught by the kernel's compaction layer:
    history is summarized in place and the call retried with the rebuilt,
    smaller prompt."""
    compactor = _CountingCompactor()
    llm = _OverflowLLM([_response(content="Recovered.")], overflows=1)
    loop, _ = _overflow_rig(llm, compactor)
    cs = _agent_state()
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "hi"},
    ]

    reply, _, _ = loop.drive(cs, "agent", history)

    assert reply == "Recovered."
    assert compactor.calls == 1
    # The retry's prompt was rebuilt from the compacted history.
    retry_messages = llm.calls[-1]
    assert any("[Conversation summary from earlier]" in (m.get("content") or "")
               for m in retry_messages)
    assert cs.turn_priority == "user"


def test_double_overflow_falls_back_to_emergency_truncation():
    """When the post-compact retry still overflows, the layer truncates
    history to an emergency stub and tries once more."""
    llm = _OverflowLLM([_response(content="Barely made it.")], overflows=2)
    loop, _ = _overflow_rig(llm, _CountingCompactor())
    cs = _agent_state()
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "hi"},
    ]

    reply, _, _ = loop.drive(cs, "agent", history)

    assert reply == "Barely made it."
    assert history[0]["content"].startswith("[Earlier conversation dropped")


def test_unrecoverable_overflow_surfaces_the_start_fresh_error():
    llm = _OverflowLLM([], overflows=3)
    loop, _ = _overflow_rig(llm, _CountingCompactor())
    cs = _agent_state()
    history = [{"role": "user", "content": "hi"}]

    import pytest
    with pytest.raises(RuntimeError, match="Use /new"):
        loop.drive(cs, "agent", history)


def test_overflow_retry_stays_on_the_escort_swapped_brain():
    """The compaction layer sits inside registered escorts, so its retries
    keep the post-escort brain instead of falling back to the loop default."""
    default_llm = _FakeLLM([])  # must never be called
    swapped = _OverflowLLM([_response(content="From the swapped brain.")], overflows=1)
    loop, runtime = _overflow_rig(default_llm, _CountingCompactor())

    def escort(ctx, request, proceed):
        request.llm = swapped
        return proceed(request)

    runtime.hooks.add("llm_call", escort)
    cs = _agent_state()
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "hi"},
    ]

    reply, _, _ = loop.drive(cs, "agent", history)

    assert reply == "From the swapped brain."
    assert default_llm.calls == []  # both the call and its retry used request.llm


def test_proactive_compaction_measures_the_brain_that_took_the_call():
    """Proactive compaction reads context_size off the post-escort brain."""
    compactor = _CountingCompactor()

    class _SmallBrain(_FakeLLM):
        context_size = 100  # 90/100 prompt tokens -> compaction triggers

    small = _SmallBrain([SimpleNamespace(
        content="Done.", tool_calls=[], has_tool_calls=False,
        is_error=False, prompt_tokens=90,
    )])
    loop, runtime = _overflow_rig(_FakeLLM([]), compactor)  # default has context_size=0

    def escort(ctx, request, proceed):
        request.llm = small
        return proceed(request)

    runtime.hooks.add("llm_call", escort)
    cs = _agent_state()
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "hi"},
    ]

    reply, _, _ = loop.drive(cs, "agent", history)

    assert reply == "Done."
    assert compactor.calls == 1


# ────────────────────────────────────────────────────────────────────
# Queued user messages (was test_message_queue.py)
# ────────────────────────────────────────────────────────────────────

import state_machine  # noqa: F401
from pipeline.database import Database
from tests.support import plain_runtime


def _db(tmp_path):
    return Database(str(tmp_path / "queue.db"))


def _busy_runtime(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    rt = plain_runtime(db)
    session = rt.load_conversation("s", cid)
    session.busy = True
    return rt, session


def test_busy_send_text_is_queued_not_rejected(tmp_path):
    rt, session = _busy_runtime(tmp_path)

    out = rt.handle_action("s", "send_text", "hello mid-turn")

    assert out.ok
    assert out.data.get("queued") is True
    assert session.pending_user_messages == ["hello mid-turn"]

    rt.handle_action("s", "send_text", "and another")
    assert session.pending_user_messages == ["hello mid-turn", "and another"]


def test_busy_empty_text_is_still_rejected(tmp_path):
    rt, session = _busy_runtime(tmp_path)

    out = rt.handle_action("s", "send_text", "")

    assert not out.ok
    assert out.error["code"] == "empty_input"
    assert session.pending_user_messages == []


def test_cancel_clears_the_queue(tmp_path):
    """And leaves it empty. Anything put here would be *driven*, not merely
    read — the closing-race check in ``handle_action`` pops this list and
    dispatches it as a fresh ``send_text``. The "you were cancelled" notice
    the model needs is recorded into history by the loop instead
    (``ConversationLoop._record_cancellation``)."""
    rt, session = _busy_runtime(tmp_path)
    rt.handle_action("s", "send_text", "queued one")

    out = rt.handle_action("s", "cancel", None)

    assert "Cancelled." in out.messages
    assert session.pending_user_messages == []
    assert session.cancel_event.is_set()


def test_non_send_text_actions_still_get_busy_error(tmp_path):
    rt, _ = _busy_runtime(tmp_path)

    out = rt.handle_action("s", "call_command", {"name": "anything", "args": {}})

    assert not out.ok
    assert out.error["code"] == "busy"


def test_end_of_turn_leftover_starts_a_fresh_turn(tmp_path):
    """A message queued in the closing race window (after the loop's final
    drain, before busy=False) is dispatched as a real user send_text once the
    turn ends, driving a follow-up turn."""
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    rt = plain_runtime(db)
    session = rt.load_conversation("s", cid)

    turns = []

    def fake_drive(sess, out, allow_restart=True):
        turns.append(list(m["content"] for m in sess.history if m["role"] == "user"))
        if len(turns) == 1:
            # Simulate the race: a message lands after the loop's final drain.
            sess.pending_user_messages.append("leftover")
        out.messages.append(f"reply {len(turns)}")
        # Mimic the real driver's finally-block hand-back.
        sess.cs.set_priority("user")
        sess.busy = False
        return out

    rt._drive_agent_turn = fake_drive

    out = rt.handle_action("s", "send_text", "first message")

    assert len(turns) == 2
    # The follow-up turn saw the leftover as a real user history row.
    assert turns[1][-1] == "leftover"
    assert session.pending_user_messages == []
    assert "reply 1" in out.messages and "reply 2" in out.messages
    assert session.cs.turn_priority == "user"


# ────────────────────────────────────────────────────────────────────
# The compactor as a sandboxed service (was test_service_compactor_sandbox.py)
# ────────────────────────────────────────────────────────────────────

from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.guest.requests import AGENT_COMPLETE
from sandbox.handlers import HANDLERS
from tests.support import call_handler


def test_agent_complete_selects_the_session_llm(monkeypatch):
    """A resident service still compacts with the session's active profile."""
    selected = object()
    fallback = object()
    session = object()
    runtime = SimpleNamespace(
        sessions={"chat": session},
        services={"llm": fallback},
    )
    ctx = SimpleNamespace(runtime=runtime, services=runtime.services)
    seen = []

    class Brain:
        # The backend contract, in full. A double taking ``messages`` alone
        # passed only because the handler used to call it directly — which
        # meant the handler spoke a language no real brain answered to.
        def chat(self, request, on_delta=None):
            seen.append(list(request.messages))
            return SimpleNamespace(
                is_error=False,
                content="summary",
                tool_calls=[],
                error=None,
                error_code=None,
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

    brain = Brain()
    # ``monkeypatch.setattr`` on a dotted string imports the module first, and
    # ``runtime.runtime_config`` cannot *be* imported first: it and
    # ``runtime.persistence`` import each other, so whichever is asked for
    # initially fails on a partially initialized partner. ``runtime.bootstrap``
    # is an entry point that resolves the cycle. This test passed only when
    # some earlier test in the session had already pulled runtime in — a pass
    # by luck, and it went green or red depending on which files pytest
    # collected alongside it.
    # ``from … import`` and not ``import runtime.bootstrap``: the latter binds
    # the name ``runtime`` in this scope, shadowing the fake runtime above.
    from runtime import bootstrap  # noqa: F401

    monkeypatch.setattr(
        "runtime.runtime_config.active_llm",
        lambda actual_runtime, actual_session: (
            brain
            if actual_runtime is runtime and actual_session is session
            else selected
        ),
    )

    result = call_handler(AGENT_COMPLETE, ctx,
        {
            "session_key": "chat",
            "messages": [{"role": "user", "content": "history"}],
        },
    )

    assert result.ok
    assert result.data["content"] == "summary"
    assert seen == [[{"role": "user", "content": "history"}]]


def test_compactor_runs_through_the_sandbox(monkeypatch):
    """The real service exports one serializable compaction call."""
    sandbox = Sandbox()
    configure(sandbox)
    module = adapt("bundled/services/service_compactor.py")
    service = module.build_services({})["compactor"]
    seen = []

    def complete(_ctx, args):
        seen.append(args)
        from sandbox.guest.requests import Result

        return Result(data={"content": "  compacted  ", "tool_calls": []})

    monkeypatch.setitem(HANDLERS, AGENT_COMPLETE, complete)
    try:
        assert service.load()
        assert service.compact(
            session_key="chat",
            transcript="USER: hello",
        ) == "compacted"
        assert seen[0]["session_key"] == "chat"
        assert seen[0]["messages"][1]["content"] == "USER: hello"
    finally:
        service.unload()
        configure(None)
        sandbox.shutdown()


# ── naming a model, rather than holding one ───────────────────────────
#
# ``ModelRequest.llm`` is a name the kernel resolves, and ``agent.complete``
# works the same way for the same reason: a box cannot hold a live model, so a
# background chore that wants a cheap one has to be able to say which.

def test_agent_complete_resolves_a_named_profile(monkeypatch):
    """An explicit profile wins over whatever the session drives with."""
    from sandbox.guest.llm import LLMResponse

    placed = []

    class Cheap:
        name = "cheap"

        def chat(self, request, on_delta=None):
            placed.append(request)
            return LLMResponse(content="six words or fewer")

    monkeypatch.setattr("llm.registry.usable_brain",
                        lambda name: Cheap() if name == "cheap" else None)

    result = call_handler(AGENT_COMPLETE, SimpleNamespace(config={}),
        {"profile": "cheap", "messages": [{"role": "user", "content": "hi"}]})

    assert result.ok
    assert result.data["content"] == "six words or fewer"
    assert result.data["llm"] == "cheap"
    assert placed[0].messages == [{"role": "user", "content": "hi"}]


def test_a_named_profile_that_does_not_exist_says_so(monkeypatch):
    """Falling back to the default would title conversations with the
    expensive model and never mention it."""
    monkeypatch.setattr("llm.registry.usable_brain", lambda name: None)

    result = call_handler(AGENT_COMPLETE, SimpleNamespace(config={}),
                          {"profile": "gone", "prompt": "hi"})

    assert not result.ok
    assert "gone" in result.error


def test_no_profile_and_no_session_uses_the_default_brain(monkeypatch):
    """The fallback used to be ``services["llm"]`` — a service that stopped
    existing when the LLM moved kernel-side, so this path was simply dead."""
    from sandbox.guest.llm import LLMResponse

    class Default:
        name = "default"

        def chat(self, request, on_delta=None):
            return LLMResponse(content="from the default")

    monkeypatch.setattr("llm.default_brain", lambda config: Default())

    result = call_handler(AGENT_COMPLETE, SimpleNamespace(config={}),
                                      {"prompt": "hi"})

    assert result.ok
    assert result.data["content"] == "from the default"


def test_a_prompt_becomes_one_user_message(monkeypatch):
    """``prompt`` is the convenience shape; ``messages`` is the real one."""
    from sandbox.guest.llm import LLMResponse

    seen = []

    class Brain:
        name = "b"

        def chat(self, request, on_delta=None):
            seen.append(request.messages)
            return LLMResponse(content="ok")

    monkeypatch.setattr("llm.default_brain", lambda config: Brain())
    call_handler(
        AGENT_COMPLETE, SimpleNamespace(config={}), {"prompt": "hello"})

    assert seen == [[{"role": "user", "content": "hello"}]]


# ──────────────────────────────────────────────────────────────────────
# The reserved ``narration`` parameter
# ──────────────────────────────────────────────────────────────────────

def test_a_declared_narration_never_reaches_the_tool():
    """The kernel owns the rendering, so ``run`` must not receive the argument.

    Tool signatures are explicit by house style, so an unstripped kwarg is a
    ``TypeError`` — which ``ToolRegistry.call`` catches and reports as
    ``ToolResult.failed``. That reads exactly like a bug in the tool, which is
    why this is pinned rather than trusted.
    """
    from agent.tool_registry import ToolRegistry
    from plugins.native.tool import BaseTool

    seen = {}

    class _Narrating(BaseTool):
        """A tool that declares the reserved name."""

        name = "narrating"
        description = "Runs a command."
        parameters = {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "narration": {"type": "string"},
            },
            "required": ["command"],
        }

        def run(self, context, command):
            """Note the signature: no ``narration``."""
            seen["command"] = command
            return ToolResult(llm_summary="ok")

    registry = ToolRegistry(None, {})
    registry.register(_Narrating())

    result = registry.call("narrating", command="git status",
                           narration="checking what changed")

    assert result.success, result.error
    assert seen == {"command": "git status"}


def test_the_narration_reaches_both_status_events():
    """Started *and* finished, because a frontend may overwrite in place.

    The REPL redraws its status line with ``\r``, so a narration that only
    rode the started event would vanish the moment the tool returned — and the
    readable scrollback is the entire point of the declaration. That failure is
    invisible (the started line renders correctly), so it is asserted here.
    """
    started, finished = [], []
    tools = {"echo": CallableSpec("echo", handler=lambda cs, actor, args: ToolResult(llm_summary="ok"))}
    cs = _agent_state(tools=tools)
    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}
    llm = _FakeLLM([
        _response(content="", tool_calls=[{
            "id": "c1", "name": "echo",
            "arguments": '{"text": "ping", "narration": "saying hello"}',
        }]),
        _response(content="Done."),
    ])
    loop = ConversationLoop(
        llm, _FakeRegistry([schema]), {}, "prompt",
        on_tool_start=lambda name, call_id, args: started.append(args),
        on_tool_result=lambda name, call_id, result, error, narration: finished.append(narration),
    )

    loop.drive(cs, "agent", [{"role": "user", "content": "go"}])

    assert started == [{"text": "ping", "narration": "saying hello"}]
    assert finished == ["saying hello"]


def test_a_tool_that_declares_no_narration_carries_none():
    """The common case stays exactly as it was: no key, no blurb, no cost."""
    finished = []
    tools = {"echo": CallableSpec("echo", handler=lambda cs, actor, args: ToolResult(llm_summary="ok"))}
    cs = _agent_state(tools=tools)
    schema = {"type": "function", "function": {"name": "echo", "parameters": {}}}
    llm = _FakeLLM([
        _response(content="", tool_calls=[{"id": "c1", "name": "echo", "arguments": '{"text": "ping"}'}]),
        _response(content="Done."),
    ])
    loop = ConversationLoop(
        llm, _FakeRegistry([schema]), {}, "prompt",
        on_tool_result=lambda name, call_id, result, error, narration: finished.append(narration),
    )

    loop.drive(cs, "agent", [{"role": "user", "content": "go"}])

    assert finished == [None]


def test_both_status_events_carry_the_narration_at_the_top_level():
    """The bug this exists for: Telegram rendered no blurb at all.

    Each frontend has its own status renderer, so anything a renderer has to
    *derive* gets derived differently in each one — or, as happened here, not
    at all. The narration is lifted out of ``args`` and normalized once, so a
    renderer reads one key on both events and only decides styling.
    """
    from runtime.runtime_config import tool_blurb, tool_callbacks

    emitted = []
    runtime = SimpleNamespace(
        on_tool_start=None, on_tool_result=None,
        emit_event=lambda channel, payload: emitted.append((channel, payload)))
    started, finished = tool_callbacks(runtime, "repl:1")

    started("run_command", "c1", {"command": "git status",
                                  "narration": "checking  what\nchanged"})
    finished("run_command", "c1", narration="checking  what\nchanged")

    keys = [payload["narration"] for _, payload in emitted]
    assert keys == ["checking what changed", "checking what changed"]
    # ...and the verbatim call is still intact for anyone who wants it.
    assert emitted[0][1]["args"]["narration"] == "checking  what\nchanged"


def test_an_overlong_narration_is_capped_before_any_frontend_sees_it():
    """Capping is policy, so it happens once rather than in each renderer."""
    from runtime.runtime_config import tool_blurb

    assert tool_blurb("  spaced   out \n words ") == "spaced out words"
    assert tool_blurb(None) == ""
    assert tool_blurb("") == ""
    assert len(tool_blurb("x" * 500)) == 80
    assert tool_blurb("x" * 500).endswith("...")


def _finished_payload(tool_result, ok=True, **kwargs):
    """Emit one TOOL_CALL_FINISHED for ``tool_result`` and hand back its payload."""
    from runtime.runtime_config import tool_callbacks

    emitted = []
    runtime = SimpleNamespace(
        on_tool_start=None, on_tool_result=None,
        emit_event=lambda channel, payload: emitted.append(payload))
    _, finished = tool_callbacks(runtime, "http:1")
    result = SimpleNamespace(ok=ok, error=None, data={"result": tool_result})
    finished("search", "c1", result=result, **kwargs)
    return emitted[0]


def test_the_finished_event_carries_what_the_tool_actually_returned():
    """The bug this exists for: a frontend could only see a tool *fail*.

    ``ok`` and ``error`` were on the event and the result itself was not, so
    every client rendered a successful call as a checkmark with nothing behind
    it — the outcome was legible only when there was an error to print.
    """
    payload = _finished_payload(ToolResult(llm_summary="Found 3 files."))
    assert payload["summary"] == "Found 3 files."
    assert payload["ok"] is True


def test_a_tool_that_filled_in_only_the_structured_half_still_says_something():
    """``data`` is documented as being for frontends. Let it reach one."""
    payload = _finished_payload(ToolResult(data={"files": 3}))
    assert payload["summary"] == '{"files": 3}'


def test_nothing_to_report_is_empty_rather_than_the_word_null():
    """``json.dumps(None)`` is "null", which no person should ever be shown."""
    assert _finished_payload(ToolResult())["summary"] == ""


def test_a_failed_call_leaves_the_summary_to_the_error():
    """Two fields saying the same thing is how they come to disagree."""
    payload = _finished_payload(
        ToolResult(success=False, error="No such path."), ok=False)
    assert payload["summary"] == ""
    assert payload["error"] == "No such path."


def test_an_unserializable_result_costs_the_summary_and_not_the_event():
    """The call succeeded. Losing its ✓ over an unprintable blob would lie."""
    class Unprintable:
        def __repr__(self):
            raise ValueError("no")

    payload = _finished_payload(ToolResult(data={"x": {Unprintable()}}))
    assert payload["summary"] == ""
    assert payload["ok"] is True


def test_the_wire_and_the_transcript_cap_a_long_result_identically():
    """A frontend showing one live and the other on reload must not appear to
    change its mind about what happened."""
    from runtime.conversation_loop import ConversationLoop

    tool_result = ToolResult(llm_summary="x" * 20000)
    loop = ConversationLoop(_FakeLLM([]), _FakeRegistry([]), {}, "prompt")
    stored, _ = loop._format_tool_result(
        "search", SimpleNamespace(ok=True, error=None,
                                  data={"result": tool_result}), {})

    assert _finished_payload(tool_result)["summary"] == stored
    assert len(stored) < 20000
