"""Tests for LiteLLM service plugin."""

import unittest
from unittest.mock import MagicMock, patch

from plugins.services.service_llm import LiteLLM, LLMResponse, LLMProviderError, _build_llm_from_profile


def _make_response(content="Hello!", tool_calls=None, prompt_tokens=10, completion_tokens=5):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = tool_calls
    resp.choices[0].finish_reason = "stop"
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    resp.usage.prompt_tokens_details = None
    return resp


class TestLiteLLM(unittest.TestCase):
    @patch("litellm.completion", return_value=_make_response())
    def test_invoke_dispatches_to_litellm(self, mock_completion):
        llm = LiteLLM("anthropic/claude-sonnet-4-6")
        llm.loaded = True
        result = llm.invoke([{"role": "user", "content": "Hi"}])
        mock_completion.assert_called_once()
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["model"], "anthropic/claude-sonnet-4-6")
        self.assertTrue(kw["drop_params"])
        self.assertEqual(result.content, "Hello!")
        self.assertIsNone(result.error)

    @patch("litellm.completion", return_value=_make_response())
    def test_invoke_forwards_api_key(self, mock_completion):
        llm = LiteLLM("openai/gpt-4o", api_key="sk-test")
        llm.loaded = True
        llm.invoke([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["api_key"], "sk-test")

    @patch("litellm.completion", return_value=_make_response())
    def test_invoke_forwards_base_url(self, mock_completion):
        llm = LiteLLM("openai/gpt-4o", base_url="http://localhost:4000")
        llm.loaded = True
        llm.invoke([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["api_base"], "http://localhost:4000")

    @patch("litellm.completion", return_value=_make_response())
    def test_invoke_returns_prompt_tokens(self, mock_completion):
        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        result = llm.invoke([{"role": "user", "content": "test"}])
        self.assertEqual(result.prompt_tokens, 10)

    @patch("litellm.completion")
    def test_invoke_tool_calls(self, mock_completion):
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "London"}'
        mock_completion.return_value = _make_response(content="", tool_calls=[tc])
        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        result = llm.invoke([{"role": "user", "content": "weather?"}], tools=[{"type": "function"}])
        self.assertTrue(result.has_tool_calls)
        self.assertEqual(result.tool_calls[0]["name"], "get_weather")

    @patch("litellm.completion", return_value=_make_response())
    def test_stream_yields_chunks(self, mock_completion):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        mock_completion.return_value = iter([chunk1, chunk2])

        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        chunks = list(llm.stream([{"role": "user", "content": "test"}]))
        self.assertEqual(chunks, ["Hello", " world"])
        kw = mock_completion.call_args.kwargs
        self.assertTrue(kw["stream"])

    def test_invoke_not_loaded_returns_error(self):
        llm = LiteLLM("openai/gpt-4o")
        result = llm.invoke([{"role": "user", "content": "test"}])
        self.assertTrue(result.is_error)
        self.assertEqual(result.error_code, "not_loaded")

    def test_build_llm_from_profile_litellm(self):
        profile = {"llm_service_class": "LiteLLM", "llm_api_key": "sk-test", "llm_context_size": 128000}
        llm = _build_llm_from_profile("anthropic/claude-sonnet-4-6", profile)
        self.assertIsInstance(llm, LiteLLM)
        self.assertEqual(llm.model_name, "anthropic/claude-sonnet-4-6")
        self.assertEqual(llm.context_size, 128000)

    @patch("litellm.completion", return_value=_make_response())
    def test_chat_with_tools_delegates_to_invoke(self, mock_completion):
        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        tools = [{"type": "function", "function": {"name": "test"}}]
        llm.chat_with_tools([{"role": "user", "content": "test"}], tools=tools)
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["tools"], tools)

    @patch("litellm.completion", return_value=_make_response(content=None))
    def test_null_content_returns_empty(self, mock_completion):
        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        result = llm.invoke([{"role": "user", "content": "test"}])
        self.assertEqual(result.content, "")

    @patch("litellm.completion")
    def test_rate_limit_not_misclassified_as_context_limit(self, mock_completion):
        import litellm as _litellm
        mock_completion.side_effect = _litellm.RateLimitError(
            message="Rate limit exceeded. Quota request exceeds the tokens limit.",
            llm_provider="anthropic",
            model="anthropic/claude-sonnet-4-6",
        )
        llm = LiteLLM("anthropic/claude-sonnet-4-6")
        llm.loaded = True
        result = llm.invoke([{"role": "user", "content": "test"}])
        self.assertTrue(result.is_error)
        self.assertEqual(result.error_code, "provider_error")
        self.assertNotEqual(result.error_code, "context_limit")

    @patch("litellm.completion")
    def test_context_limit_raises_provider_error(self, mock_completion):
        import litellm as _litellm
        mock_completion.side_effect = _litellm.ContextWindowExceededError(
            message="This model's maximum context length is 8192 tokens",
            llm_provider="openai",
            model="openai/gpt-4o",
        )
        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        with self.assertRaises(LLMProviderError) as ctx:
            llm.invoke([{"role": "user", "content": "test"}])
        self.assertEqual(ctx.exception.code, "context_limit")

    @patch("litellm.completion", return_value=_make_response())
    def test_response_format_passthrough(self, mock_completion):
        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        llm.invoke(
            [{"role": "user", "content": "test"}],
            response_format={"type": "json_object"},
        )
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["response_format"], {"type": "json_object"})

    @patch("litellm.completion", return_value=_make_response())
    def test_chat_with_tools_passes_attachments(self, mock_completion):
        llm = LiteLLM("openai/gpt-4o")
        llm.loaded = True
        result = llm.chat_with_tools(
            [{"role": "user", "content": "test"}],
            tools=None,
            attachments=None,
        )
        self.assertEqual(result.content, "Hello!")

    def test_stream_not_loaded_returns_nothing(self):
        llm = LiteLLM("openai/gpt-4o")
        chunks = list(llm.stream([{"role": "user", "content": "test"}]))
        self.assertEqual(chunks, [])

    def test_image_capability_inferred(self):
        llm = LiteLLM("anthropic/claude-sonnet-4-6")
        self.assertFalse(llm.has_capability("image"))
        llm2 = LiteLLM("openai/gpt-4o")
        self.assertTrue(llm2.has_capability("image"))
        llm3 = LiteLLM("anthropic/claude-3-5-sonnet")
        self.assertTrue(llm3.has_capability("image"))


if __name__ == "__main__":
    unittest.main()
