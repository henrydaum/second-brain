"""Tests for LiteLLMLLM service plugin."""

import unittest
from unittest.mock import MagicMock, patch

from plugins.services.service_llm import LiteLLMLLM, LLMResponse, _build_llm_from_profile


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


class TestLiteLLMLLM(unittest.TestCase):
    @patch("litellm.completion", return_value=_make_response())
    def test_invoke_dispatches_to_litellm(self, mock_completion):
        llm = LiteLLMLLM("anthropic/claude-sonnet-4-6")
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
        llm = LiteLLMLLM("openai/gpt-4o", api_key="sk-test")
        llm.loaded = True
        llm.invoke([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["api_key"], "sk-test")

    @patch("litellm.completion", return_value=_make_response())
    def test_invoke_forwards_base_url(self, mock_completion):
        llm = LiteLLMLLM("openai/gpt-4o", base_url="http://localhost:4000")
        llm.loaded = True
        llm.invoke([{"role": "user", "content": "test"}])
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["api_base"], "http://localhost:4000")

    @patch("litellm.completion", return_value=_make_response())
    def test_invoke_returns_prompt_tokens(self, mock_completion):
        llm = LiteLLMLLM("openai/gpt-4o")
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
        llm = LiteLLMLLM("openai/gpt-4o")
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

        llm = LiteLLMLLM("openai/gpt-4o")
        llm.loaded = True
        chunks = list(llm.stream([{"role": "user", "content": "test"}]))
        self.assertEqual(chunks, ["Hello", " world"])
        kw = mock_completion.call_args.kwargs
        self.assertTrue(kw["stream"])

    def test_invoke_not_loaded_returns_error(self):
        llm = LiteLLMLLM("openai/gpt-4o")
        result = llm.invoke([{"role": "user", "content": "test"}])
        self.assertTrue(result.is_error)
        self.assertEqual(result.error_code, "not_loaded")

    def test_build_llm_from_profile_litellm(self):
        profile = {"llm_service_class": "LiteLLMLLM", "llm_api_key": "sk-test", "llm_context_size": 128000}
        llm = _build_llm_from_profile("anthropic/claude-sonnet-4-6", profile)
        self.assertIsInstance(llm, LiteLLMLLM)
        self.assertEqual(llm.model_name, "anthropic/claude-sonnet-4-6")
        self.assertEqual(llm.context_size, 128000)

    @patch("litellm.completion", return_value=_make_response())
    def test_chat_with_tools_delegates_to_invoke(self, mock_completion):
        llm = LiteLLMLLM("openai/gpt-4o")
        llm.loaded = True
        tools = [{"type": "function", "function": {"name": "test"}}]
        llm.chat_with_tools([{"role": "user", "content": "test"}], tools=tools)
        kw = mock_completion.call_args.kwargs
        self.assertEqual(kw["tools"], tools)

    @patch("litellm.completion", return_value=_make_response(content=None))
    def test_null_content_returns_empty(self, mock_completion):
        llm = LiteLLMLLM("openai/gpt-4o")
        llm.loaded = True
        result = llm.invoke([{"role": "user", "content": "test"}])
        self.assertEqual(result.content, "")


if __name__ == "__main__":
    unittest.main()
