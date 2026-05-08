from types import SimpleNamespace

from plugins.frontends.helpers.formatters import format_tool_result


def test_tool_result_prefers_success_summary_over_json():
    result = SimpleNamespace(
        success=True,
        error="",
        llm_summary="Created default-agent subagent conversation #3: Nightly Wisdom. Ran one subagent turn immediately.",
        data={"conversation_id": 3, "final_text": "2 + 2 = 4"},
    )

    text = format_tool_result(result)

    assert text.startswith("Done: Created default-agent subagent conversation #3")
    assert "2 + 2 = 4" in text
    assert '"conversation_id"' not in text


def test_tool_result_failure_uses_clear_failure_message():
    text = format_tool_result(SimpleNamespace(success=False, error="Schedule denied.", llm_summary="", data=None))

    assert text == "Failed: Schedule denied."
