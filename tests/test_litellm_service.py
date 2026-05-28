"""Tests for the LiteLLMService unified backend."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from plugins.services.service_llm import (
    LLMProviderError,
    LiteLLMService,
    _build_llm_from_profile,
)


def _install_fake_litellm(monkeypatch, completion):
    """Replace the ``litellm`` module with a stub exposing ``completion``."""
    fake = ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def _make_llm(monkeypatch, completion, **kwargs):
    _install_fake_litellm(monkeypatch, completion)
    llm = LiteLLMService("anthropic/claude-sonnet-4-6", **kwargs)
    llm.loaded = True
    return llm


def _response(content="ok", tool_calls=None, prompt_tokens=10, cached_tokens=None):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens else None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls or []))],
        usage=usage,
    )


def test_build_from_profile_picks_litellm_service():
    llm = _build_llm_from_profile("anthropic/claude-sonnet-4-6", {
        "llm_service_class": "LiteLLMService",
        "llm_api_key": "ANTHROPIC_API_KEY",
        "llm_context_size": 200000,
    })
    assert isinstance(llm, LiteLLMService)
    assert llm.context_size == 200000


def test_invoke_forwards_model_messages_and_credentials(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _response()

    llm = _make_llm(monkeypatch, completion, api_key="sk-test", base_url="https://example.test")
    result = llm.invoke([{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert calls[0]["model"] == "anthropic/claude-sonnet-4-6"
    assert calls[0]["api_key"] == "sk-test"
    assert calls[0]["base_url"] == "https://example.test"


def test_tool_calls_round_trip(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="search", arguments='{"q": "x"}'),
    )

    def completion(**kwargs):
        return _response(content="", tool_calls=[tool_call])

    llm = _make_llm(monkeypatch, completion)
    result = llm.chat_with_tools(
        [{"role": "user", "content": "find x"}],
        tools=[{"type": "function", "function": {"name": "search"}}],
    )

    assert result.has_tool_calls
    assert result.tool_calls[0] == {"id": "call_1", "name": "search", "arguments": '{"q": "x"}'}


def test_rate_limit_not_misclassified_as_context_limit(monkeypatch):
    """The is_context_limit_error heuristic matches 'tokens' + 'limit', so a
    rate-limit message containing both would otherwise trigger the
    compact-and-retry path. Class-name check must short-circuit first."""
    class RateLimitError(Exception):
        pass

    def completion(**kwargs):
        raise RateLimitError("Rate limit exceeded. Quota request exceeds the tokens limit.")

    llm = _make_llm(monkeypatch, completion)
    result = llm.invoke([{"role": "user", "content": "hi"}])

    assert result.is_error
    assert result.error_code == "provider_error"


def test_context_limit_raises_provider_error(monkeypatch):
    def completion(**kwargs):
        raise RuntimeError("prompt is too long for context window")

    llm = _make_llm(monkeypatch, completion)
    with pytest.raises(LLMProviderError) as exc_info:
        llm.invoke([{"role": "user", "content": "hi"}])
    assert exc_info.value.code == "context_limit"


def test_invoke_not_loaded_returns_error():
    llm = LiteLLMService("anthropic/claude-sonnet-4-6")
    result = llm.invoke([{"role": "user", "content": "hi"}])
    assert result.error_code == "not_loaded"


def test_stream_yields_chunks(monkeypatch):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
    ]

    def completion(**kwargs):
        assert kwargs.get("stream") is True
        return iter(chunks)

    llm = _make_llm(monkeypatch, completion)
    out = "".join(llm.stream([{"role": "user", "content": "hi"}]))
    assert out == "Hello"


def test_image_capability_inferred_from_model_name():
    assert LiteLLMService("anthropic/claude-sonnet-4-6").capabilities["image"] is True
    assert LiteLLMService("openai/gpt-4o").capabilities["image"] is True
    assert LiteLLMService("vertex_ai/gemini-2.5-flash").capabilities["image"] is True
    assert LiteLLMService("mistral/mistral-large-latest").capabilities["image"] is None
