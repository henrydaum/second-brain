"""Contract tests for the store LiteLLM backend's streaming path.

The backend lives on the ``store`` branch; the module is materialized from
the local store ref via ``git show`` (mirroring what ``/packages install``
copies) and driven with a fake ``litellm`` module injected into
``sys.modules`` — no network, no litellm dependency.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.services.service_llm import LLMProviderError

_REPO = Path(__file__).resolve().parents[1]
_STORE_REL = "services/service_litellm.py"


def _store_module_source() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_STORE_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture()
def litellm_service(tmp_path, monkeypatch):
    """Yield (service, fake_litellm_module); the fake's .completion is set per test."""
    source = _store_module_source()
    if source is None:
        pytest.skip(f"{_STORE_REL} not present on a local store ref")
    fake = types.ModuleType("litellm")
    fake.completion = None  # tests assign
    monkeypatch.setitem(sys.modules, "litellm", fake)
    path = tmp_path / "service_litellm.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("store_service_litellm_streaming", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    svc = module.LiteLLMService("openai/test-model")
    assert svc._load()
    assert svc.supports_streaming is True
    return svc, fake


def _content_chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))], usage=None)


def _tool_chunk(index, call_id, name, arguments):
    tc = SimpleNamespace(index=index, id=call_id,
                         function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))], usage=None)


def _usage_chunk(prompt_tokens, cached=None):
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt_tokens, prompt_tokens_details=details))


def test_streaming_accumulates_content_and_usage(litellm_service):
    svc, fake = litellm_service
    seen_kwargs = {}

    def completion(**kwargs):
        seen_kwargs.update(kwargs)
        return iter([_content_chunk("Hel"), _content_chunk("lo!"), _usage_chunk(42, cached=7)])

    fake.completion = completion
    deltas = []

    resp = svc.chat_with_tools_streaming([{"role": "user", "content": "hi"}],
                                         on_delta=lambda d: deltas.append(d) or True)

    assert seen_kwargs["stream"] is True
    assert seen_kwargs["stream_options"] == {"include_usage": True}
    assert deltas == ["Hel", "lo!"]
    assert resp.content == "Hello!"
    assert resp.tool_calls == []
    assert resp.prompt_tokens == 42
    assert resp.cached_prompt_tokens == 7


def test_streaming_accumulates_tool_call_deltas_by_index(litellm_service):
    svc, fake = litellm_service
    fake.completion = lambda **kw: iter([
        _content_chunk("Let me check."),
        _tool_chunk(0, "call_1", "echo", '{"a"'),
        _tool_chunk(0, None, None, ": 1}"),
        _tool_chunk(1, "call_2", "noop", "{}"),
    ])

    resp = svc.chat_with_tools_streaming([], tools=[{"type": "function"}], on_delta=lambda d: True)

    assert resp.content == "Let me check."
    assert resp.tool_calls == [
        {"id": "call_1", "name": "echo", "arguments": '{"a": 1}'},
        {"id": "call_2", "name": "noop", "arguments": "{}"},
    ]


def test_streaming_abort_returns_partial_and_stops_consuming(litellm_service):
    svc, fake = litellm_service
    consumed = []

    def chunks():
        for text in ("one ", "two ", "three"):
            consumed.append(text)
            yield _content_chunk(text)

    fake.completion = lambda **kw: chunks()

    resp = svc.chat_with_tools_streaming([], on_delta=lambda d: d != "two ")

    assert resp.content == "one two "  # partial accumulation returned
    assert consumed == ["one ", "two "]  # generator abandoned after the abort


def test_streaming_context_limit_error_still_raises(litellm_service):
    svc, fake = litellm_service

    def completion(**kwargs):
        raise RuntimeError("prompt tokens exceed model token limit")

    fake.completion = completion

    with pytest.raises(LLMProviderError):
        svc.chat_with_tools_streaming([], on_delta=lambda d: True)


def test_streaming_provider_error_returns_error_response(litellm_service):
    svc, fake = litellm_service

    def completion(**kwargs):
        raise RuntimeError("boom: provider fell over")

    fake.completion = completion

    resp = svc.chat_with_tools_streaming([], on_delta=lambda d: True)

    assert resp.is_error
    assert resp.error_code == "provider_error"
