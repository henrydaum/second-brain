"""Regression tests for SQL query summary."""

from plugins.tools.tool_sql_query import _sql_summary


def test_sql_summary_truncates_large_cells():
    """Verify SQL summary truncates large cells."""
    text = _sql_summary("SELECT content FROM conversation_messages", ["content"], [("x" * 2000,)], 1, False)
    assert len(text) < 800
    assert "truncated 1500 chars" in text
