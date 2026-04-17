"""
SQL Query tool.

Gives both humans (via REPL) and the LLM (via function calling) read-only
access to the entire database. The LLM can explore the schema, inspect
the task queue, read extracted text, check file metadata — anything that
a SELECT can reach.

The tool validates that only SELECT/PRAGMA statements are executed
(enforced by Database.query()). No writes, no drops, no tricks.
"""

import logging

from Stage_3.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("SQLQuery")


class SQLQuery(BaseTool):
    name = "sql_query"
    config_settings = [
        ("Max Query Rows", "max_query_rows",
         "Maximum rows returned from database queries.",
         25,
         {"type": "slider", "range": (5, 100, 19), "is_float": False}),
    ]
    description = (
        "Execute a read-only SQL query against the local file database. "
        "Use this to inspect schema, file metadata, pipeline state, extracted text, "
        "OCR results, and stored conversations. Only SELECT and PRAGMA statements "
        "are allowed. Results are capped at 100 rows.\n\n"
        "Useful queries:\n"
        "- SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\n"
        "- PRAGMA table_info(table_name)\n"
        "- SELECT path, status FROM task_queue WHERE task_name='extract_text' AND status='FAILED'\n"
        "- SELECT path, char_count FROM extracted_text ORDER BY char_count DESC LIMIT 10\n"
        "- SELECT modality, COUNT(*) FROM files GROUP BY modality\n"
        "- SELECT c.id, c.title, c.updated_at FROM conversations c ORDER BY c.updated_at DESC LIMIT 10\n"
        "- SELECT role, content FROM conversation_messages WHERE conversation_id = N ORDER BY timestamp"
    )
    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A read-only SQL query. Only SELECT and PRAGMA statements are allowed.",
            }
        },
        "required": ["sql"],
    }
    requires_services = []
    max_calls = 6  # Failed queries are common, so allow a few extra calls.

    def run(self, context, **kwargs):
        sql = kwargs.get("sql", "").strip()
        if not sql:
            return ToolResult.failed("No SQL provided.")

        try:
            result = context.db.query(sql)
        except ValueError as e:
            logger.warning(f"Query rejected: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ToolResult.failed(str(e))

        columns = result["columns"]
        rows = result["rows"]
        row_count = len(rows)
        truncated = result["truncated"]

        return ToolResult(
            data={
                "columns": columns,
                "rows": rows,
                "row_count": row_count,
                "truncated": truncated,
            },
            llm_summary=_sql_summary(sql, columns, rows, row_count, truncated),
        )


def _sql_summary(sql: str, columns: list, rows: list, row_count: int, truncated: bool) -> str:
    """Format a SQL result as a readable summary for the LLM."""
    trunc_note = " (truncated)" if truncated else ""
    header = f"SQL: {sql}\n\nReturned {row_count} row(s){trunc_note}."
    if not rows:
        return header
    col_str = " | ".join(str(c) for c in columns)
    sep = " | ".join("-" * max(len(str(c)), 3) for c in columns)
    row_lines = [" | ".join(str(v) for v in row) for row in rows]
    table = "\n".join([col_str, sep] + row_lines)
    return f"{header}\n\n{table}"
