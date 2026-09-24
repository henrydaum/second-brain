"""The prompt-cache handle an ``LLMRequest`` carries.

Providers that route cache lookups by key (OpenAI's ``prompt_cache_key``)
scatter a conversation across machines without one, so only the shared
system-prompt head ever hits. The kernel names the conversation; the backend
decides whether its provider cares.
"""

from state_machine.conversation_phases import BASE_PHASE  # noqa: F401 (import order)

from llm import LLMRequest
from runtime.conversation_loop import ConversationLoop
from tests.support import FakeLLM, FakeRegistry, agent_state, response


def test_the_key_survives_the_wire():
    request = LLMRequest(model_name="m", cache_key="sb-abc")
    assert LLMRequest.from_dict(request.to_dict()).cache_key == "sb-abc"
    assert LLMRequest.from_dict({"model_name": "m"}).cache_key == ""


def _drive(loop):
    loop.drive(agent_state(), "agent", [{"role": "user", "content": "hi"}])


def test_one_conversation_sends_one_stable_opaque_key():
    llm = FakeLLM([response(content="a"), response(content="b")])
    loop = ConversationLoop(llm, FakeRegistry([]), {}, "sys", session_key="repl:1")
    loop._active_conversation_id = 7
    first = loop._cache_key()
    assert first.startswith("sb-") and "repl" not in first
    assert loop._cache_key() == first
    loop._active_conversation_id = 8
    assert loop._cache_key() != first


def test_a_drive_puts_the_key_on_the_request():
    llm = FakeLLM([response(content="hello")])
    loop = ConversationLoop(llm, FakeRegistry([]), {}, "sys", session_key="repl:1")
    _drive(loop)
    assert llm.records[0]["cache_key"].startswith("sb-")


def test_no_identity_means_no_key():
    loop = ConversationLoop(FakeLLM(), FakeRegistry([]), {}, "sys")
    assert loop._cache_key() == ""
