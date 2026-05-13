from types import SimpleNamespace

from plugins.services.service_llm import OpenAILLM


def test_openai_llm_forwards_prompt_cache_options_and_usage():
    """Verify OpenAI prompt-cache options and cached-token telemetry."""
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=[]))],
                usage=SimpleNamespace(prompt_tokens=1500, prompt_tokens_details=SimpleNamespace(cached_tokens=1024)),
            )

    llm = OpenAILLM("gpt-5", prompt_cache_key="second-brain", prompt_cache_retention="24h")
    llm.loaded = True
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    response = llm.invoke([{"role": "user", "content": "hi"}])

    assert calls[0]["prompt_cache_key"] == "second-brain"
    assert calls[0]["prompt_cache_retention"] == "24h"
    assert response.prompt_tokens == 1500
    assert response.cached_prompt_tokens == 1024
