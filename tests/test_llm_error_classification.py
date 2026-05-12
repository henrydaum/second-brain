"""Regression tests for LLM error classification."""

from plugins.services.llmService import is_context_limit_error


def test_model_plan_error_is_not_context_limit():
    """Verify model plan error is not context limit."""
    err = "your current token plan not support model, MiniMax-M2.7 (2061)"

    assert not is_context_limit_error(err)


def test_prompt_token_limit_is_context_limit():
    """Verify prompt token limit is context limit."""
    err = "prompt tokens exceed model token limit"

    assert is_context_limit_error(err)
