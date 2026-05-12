"""Regression tests for tool result formatters."""

from types import SimpleNamespace

from plugins.frontends.helpers.formatters import format_tool_result


def test_tool_result_prefers_success_summary_over_json():
    """Verify tool result prefers success summary over JSON."""
    result = SimpleNamespace(
        success=True,
        error="",
        llm_summary="Scheduled subagent job 'nightly_wisdom' on subagent.spawn: Nightly Wisdom.",
        data={"job_name": "nightly_wisdom", "scheduled": True},
    )

    text = format_tool_result(result)

    assert text.startswith("Done: Scheduled subagent job 'nightly_wisdom'")
    assert '"conversation_id"' not in text


def test_tool_result_failure_uses_clear_failure_message():
    """Verify tool result failure uses clear failure message."""
    text = format_tool_result(SimpleNamespace(success=False, error="Schedule denied.", llm_summary="", data=None))

    assert text == "Failed: Schedule denied."
