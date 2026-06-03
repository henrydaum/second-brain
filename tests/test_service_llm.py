"""Tests for the LLM service (``service_llm``).

Covers the unified LiteLLM backend — invoke/stream/tool-calls, credential
forwarding, error classification, and capability inference — using a fake
``litellm`` module so no network or API key is required.
"""
import sys
import logging
from types import ModuleType, SimpleNamespace

import pytest

from plugins.services.service_llm import (
    BaseLLM,
    LLMProviderError,
    LLMRouter,
    _build_llm_from_profile,
    is_context_limit_error,
    refresh_llm_profile_services,
)
try:
    from plugins.services.service_litellm import LiteLLMService
except ModuleNotFoundError:
    LiteLLMService = None


def _require_litellm_plugin():
    if LiteLLMService is None:
        pytest.skip("LiteLLMService is an optional store plugin in lite.")


def _install_fake_litellm(monkeypatch, completion):
    """Replace the ``litellm`` module with a stub exposing ``completion``."""
    fake = ModuleType("litellm")
    fake.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake)


def _make_llm(monkeypatch, completion, **kwargs):
    _require_litellm_plugin()
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
    _require_litellm_plugin()
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
    # LiteLLM's canonical connection param is api_base, not base_url.
    assert calls[0]["api_base"] == "https://example.test"


def test_custom_base_url_routes_unknown_model_as_openai_compatible(monkeypatch):
    _require_litellm_plugin()
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return _response()

    llm = LiteLLMService("deepseek-ai/deepseek-v4-pro", api_key="sk-test", base_url="https://api.atlascloud.ai/v1")
    _install_fake_litellm(monkeypatch, completion)
    llm.loaded = True

    assert llm.invoke([{"role": "user", "content": "hi"}]).content == "ok"
    assert calls[0]["model"] == "openai/deepseek-ai/deepseek-v4-pro"
    assert calls[0]["api_base"] == "https://api.atlascloud.ai/v1"


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
    _require_litellm_plugin()
    llm = LiteLLMService("anthropic/claude-sonnet-4-6")
    result = llm.invoke([{"role": "user", "content": "hi"}])
    assert result.error_code == "not_loaded"


def test_load_suppresses_litellm_logging(monkeypatch):
    _require_litellm_plugin()
    _install_fake_litellm(monkeypatch, lambda **kwargs: _response())
    llm_logger = logging.getLogger("LiteLLM")
    llm_logger.addHandler(logging.StreamHandler())
    llm_logger.propagate = True
    llm_logger.setLevel(logging.INFO)

    llm = LiteLLMService("minimax/MiniMax-M2.7")

    assert llm.load() is True
    fake = sys.modules["litellm"]
    assert fake.telemetry is False
    assert fake.set_verbose is False
    assert fake.suppress_debug_info is True
    assert fake.logging is False
    assert llm_logger.handlers == []
    assert llm_logger.propagate is False
    assert llm_logger.level == logging.ERROR


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


def test_capabilities_come_from_profile_not_model_name():
    _require_litellm_plugin()
    assert LiteLLMService("openai/gpt-4o").capabilities["image"] is None
    llm = _build_llm_from_profile("openai/gpt-4o", {
        "llm_service_class": "LiteLLMService",
        "llm_capabilities": {"image": True, "audio": False},
    })
    assert llm.capabilities["image"] is True
    assert llm.capabilities["audio"] is False


def test_refresh_llm_profile_services_adds_and_removes_backend(monkeypatch):
    class StoreBackend(BaseLLM):
        is_llm_backend = True
        def __init__(self, model_name, api_key=None, base_url=None):
            super().__init__()
            self.model_name = model_name
        def _load(self): self.loaded = True; return True
        def unload(self): self.loaded = False
        def invoke(self, messages, attachments=None, **kwargs): return _response()
        def stream(self, messages, attachments=None, **kwargs): return iter(())
        def chat_with_tools(self, messages, tools=None, **kwargs): return _response()

    config = {"llm_profiles": {"model-x": {"llm_service_class": "StoreBackend"}}, "default_llm_profile": "model-x"}
    services = {}
    services["llm"] = LLMRouter(config, services)
    monkeypatch.setattr("plugins.services.service_llm._llm_backend_classes", lambda: {"StoreBackend": StoreBackend})

    assert refresh_llm_profile_services(services, config)
    assert isinstance(services["model-x"], StoreBackend)
    assert services["model-x"].loaded

    monkeypatch.setattr("plugins.services.service_llm._llm_backend_classes", lambda: {})
    assert refresh_llm_profile_services(services, config)
    assert "model-x" not in services
    assert services["llm"].active is None


# ── Error classification heuristics ──────────────────────────────────

def test_model_plan_error_is_not_context_limit():
    err = "your current token plan not support model, MiniMax-M2.7 (2061)"
    assert not is_context_limit_error(err)


def test_prompt_token_limit_is_context_limit():
    err = "prompt tokens exceed model token limit"
    assert is_context_limit_error(err)
