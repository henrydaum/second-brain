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

from plugins.BaseTool import ToolResult
from runtime.conversation_loop import ConversationLoop


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

    def chat_with_tools(self, messages, tools, attachments=None):
        self.calls.append(messages)
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
    assert history[0]["content"] == "[Conversation summary from earlier]\nEarlier summary."
    assert history[1]["content"] == "Understood - I have the earlier context."
    assert notices == ["Compacting conversation...", "Compacted 3 messages."]
