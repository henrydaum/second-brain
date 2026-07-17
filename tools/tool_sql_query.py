"""
SQL Query tool.

Gives both humans (via REPL) and the LLM (via function calling) access to the
local database.

Read-only statements (SELECT / PRAGMA / EXPLAIN) run immediately — the agent can
freely explore the schema, the task queue, extracted text, file metadata, and
stored conversations. Any *mutating* statement (INSERT / UPDATE / DELETE / DDL,
etc.) is allowed too, but only after the user approves the exact SQL through the
active approval dialog. Reads go through ``Database.query()`` (read-only guarded);
approved writes go through ``Database.execute_write()``.
"""

dependencies_files = []
dependencies_pip = []

import logging
import re
from difflib import get_close_matches

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("SQLQuery")

# Statements that never mutate state — these run without approval. Everything
# else is treated as a write and must be approved before it executes.
_READ_ONLY_PREFIXES = ("select", "pragma", "explain")


def _is_read_only(sql: str) -> bool:
    """True if ``sql`` is a read-only statement (SELECT / PRAGMA / EXPLAIN).

    Uses the same normalization as ``Database.query()`` so classification and
    execution agree. Anything else — including CTEs that wrap a write
    (``WITH ... INSERT``) — is conservatively treated as a mutation, so the
    worst case is an extra approval prompt, never an unapproved write.
    """
    normalized = " ".join(sql.strip().split()).lower()
    return normalized.startswith(_READ_ONLY_PREFIXES)


def _schema_hint(db, error_msg: str) -> str:
    """Build a helpful hint listing tables (and a guessed table's columns) for the agent."""
    try:
        with db.lock:
            tables = [r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()]
    except Exception:
        return ""

    if not tables:
        return ""

    lines = [f"\n\nAvailable tables: {', '.join(tables)}"]

    # If error references a missing table, suggest closest match
    m = re.search(r"no such table:\s*(\S+)", error_msg)
    if m:
        bad = m.group(1)
        guesses = get_close_matches(bad, tables, n=2, cutoff=0.5)
        if guesses:
            lines.append(f"Did you mean: {', '.join(guesses)}?")
        return "\n".join(lines)

    # If error references a missing column, try to find candidate tables and list their columns
    m = re.search(r"no such column:\s*(\S+)", error_msg)
    if m:
        bad_col = m.group(1).split(".")[-1]
        suggestions = []
        for t in tables:
            try:
                with db.lock:
                    cols = [r[1] for r in db.conn.execute(f"PRAGMA table_info({t})").fetchall()]
            except Exception:
                continue
            if bad_col in cols:
                suggestions.append(f"{t}: {', '.join(cols)}")
            else:
                close = get_close_matches(bad_col, cols, n=1, cutoff=0.6)
                if close:
                    suggestions.append(f"{t} has '{close[0]}' (cols: {', '.join(cols)})")
        if suggestions:
            lines.append("Column hints:")
            lines.extend("  " + s for s in suggestions[:5])

    return "\n".join(lines)


class SQLQuery(BaseTool):
    """Sqlquery."""
    name = "sql_query"
    config_settings = [
        ("Max Query Rows", "max_query_rows",
         "Maximum rows returned from database queries.",
         25,
         {"type": "slider", "range": (5, 100, 19), "is_float": False}),
    ]
    description = (
        "Run SQL against the local file database. SELECT / PRAGMA / EXPLAIN "
        "statements run immediately and are capped at 100 rows; use them to inspect "
        "schema, file metadata, pipeline state, extracted text, OCR results, and "
        "stored conversations. Mutating statements (INSERT / UPDATE / DELETE / DDL) "
        "are also allowed, but each one pauses for explicit user approval of the exact "
        "SQL before it runs — so provide a clear `justification` when you write. "
        "Do NOT issue mutating statements unless the user has specifically asked you to "
        "change the database; default to read-only.\n\n"
        "Useful queries:\n"
        "- SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\n"
        "- PRAGMA table_info(table_name)\n"
        "- SELECT path, status FROM task_queue WHERE task_name='extract_text' AND status='FAILED'\n"
        "- SELECT path, char_count FROM extracted_text ORDER BY char_count DESC LIMIT 10\n"
        "- SELECT modality, COUNT(*) FROM files GROUP BY modality\n"
        "- UPDATE task_queue SET status='PENDING' WHERE status='FAILED'   (requires approval)"
    )
    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "A single SQL statement. SELECT/PRAGMA/EXPLAIN run immediately; "
                    "any mutating statement requires user approval first."
                ),
            },
            "justification": {
                "type": "string",
                "description": (
                    "Short plain-English reason for a mutating statement. Shown to the "
                    "user in the approval dialog. Ignored for read-only queries."
                ),
            },
        },
        "required": ["sql"],
    }
    requires_services = []
    max_calls = 6  # Failed queries are common, so allow a few extra calls.

    def agent_prompt_for(self, ctx) -> str:
        """Point the agent at the attachment cache and the live table list."""
        from paths import ATTACHMENT_CACHE
        try:
            names = [row[0] for row in ctx.db.query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )["rows"]] if ctx.db else []
        except Exception:
            names = []
        tables = ", ".join(names) if names else "No tables yet."
        return (
            f"""## Attachments
Files sent through frontends are saved to the attachment cache and indexed by the normal task pipeline. If they can be parsed into text, they will be added to the user message directly using a separate attachment parser system. You can extend this system by following the structure within attachments/parsers/.

To find recent attachments with sql_query:
SELECT path, file_name, mtime FROM files WHERE path LIKE '{ATTACHMENT_CACHE}%' ORDER BY mtime DESC LIMIT 10

## Writing to the database
sql_query can also modify data, but writing is an exception, not a routine action. Only issue an INSERT/UPDATE/DELETE/DDL statement when the user has specifically asked you to change the database — otherwise stay read-only. When you do write: SELECT/PRAGMA/EXPLAIN run automatically, but any mutating statement pauses for the user to approve the exact SQL, so include a clear `justification` and write one statement per call.

## Conversation history
Past conversations live in the 'conversations' and 'conversation_messages' tables; the current conversation's number is in the runtime context. Compacted history is no longer in your context but remains queryable there.

## Database tables (inspect with sql_query)
{tables}"""
        )

    def run(self, context, **kwargs):
        """Run sqlquery."""
        sql = kwargs.get("sql", "").strip()
        if not sql:
            return ToolResult.failed("No SQL provided.")

        if _is_read_only(sql):
            return self._run_read(context, sql)
        return self._run_write(context, sql, (kwargs.get("justification") or "").strip())

    # ── Read path (no approval) ──────────────────────────────────────
    def _run_read(self, context, sql: str):
        """Execute a read-only query through the read-only-guarded ``query()``."""
        try:
            result = context.db.query(sql)
        except ValueError as e:
            logger.warning(f"Query rejected: {e}")
            return ToolResult.failed(str(e))
        except Exception as e:
            logger.error(f"Query failed: {e}")
            msg = str(e)
            return ToolResult.failed(msg + _schema_hint(context.db, msg))

        columns = result["columns"]
        rows = result["rows"]
        return ToolResult(
            data={
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "truncated": result["truncated"],
                "wrote": False,
            },
            llm_summary=_sql_summary(sql, columns, rows, len(rows), result["truncated"]),
        )

    # ── Write path (approval required) ───────────────────────────────
    def _run_write(self, context, sql: str, justification: str):
        """Approve, then execute a mutating statement through ``execute_write()``."""
        # Don't try to get approval from an unattended/background session — there
        # is no human to answer, so fail fast instead of blocking on the dialog.
        runtime = getattr(context, "runtime", None)
        session_key = getattr(context, "session_key", None)
        if runtime is not None and session_key and hasattr(runtime, "is_attended"):
            try:
                attended = runtime.is_attended(session_key)
            except Exception:
                attended = True
            if not attended:
                return ToolResult.failed(
                    "This SQL modifies the database and needs user approval, but no "
                    "user is attending this session. Writes are not available to "
                    "background/unattended runs."
                )

        approve_fn = getattr(context, "approve_command", None)
        if approve_fn is None:
            return ToolResult.failed(
                "This SQL modifies the database and needs user approval, but no "
                "approval handler is available in this context."
            )

        reason = justification or "Modify the local database."
        try:
            approved = approve_fn(sql, f"{reason}\n\n(This statement writes to the database.)")
        except Exception as e:
            logger.error(f"Approval callback failed: {e}")
            return ToolResult.failed(f"Approval dialog error: {e}")

        if not approved:
            return ToolResult.failed(
                getattr(context, "approval_denial_reason", "")
                or "Write denied by user. STOP — do not retry this statement. "
                "Ask the user what they would like you to do instead."
            )

        try:
            result = context.db.execute_write(sql)
        except Exception as e:
            logger.error(f"Write failed: {e}")
            msg = str(e)
            return ToolResult.failed(msg + _schema_hint(context.db, msg))

        return ToolResult(
            data={
                "columns": result["columns"],
                "rows": result["rows"],
                "rowcount": result["rowcount"],
                "lastrowid": result["lastrowid"],
                "wrote": True,
            },
            llm_summary=_write_summary(sql, result),
        )


def _write_summary(sql: str, result: dict) -> str:
    """Format the outcome of an approved write for the LLM."""
    rowcount = result.get("rowcount", -1)
    affected = "unknown number of" if rowcount is None or rowcount < 0 else str(rowcount)
    parts = [f"SQL: {sql}", "", f"Statement executed. Rows affected: {affected}."]
    if result.get("lastrowid"):
        parts.append(f"Last inserted rowid: {result['lastrowid']}.")
    columns, rows = result.get("columns") or [], result.get("rows") or []
    if rows:
        parts.append("")
        parts.append(_render_table(columns, rows))
    return "\n".join(parts)


def _render_table(columns: list, rows: list) -> str:
    """Render columns/rows as a simple text table (shared by read and RETURNING output)."""
    def cell(v, limit=500):
        text = str(v)
        return text if len(text) <= limit else f"{text[:limit]}...[truncated {len(text) - limit} chars]"

    col_str = " | ".join(cell(c, 80) for c in columns)
    sep = " | ".join("-" * max(len(str(c)), 3) for c in columns)
    row_lines = [" | ".join(cell(v) for v in row) for row in rows]
    return "\n".join([col_str, sep] + row_lines)


def _sql_summary(sql: str, columns: list, rows: list, row_count: int, truncated: bool) -> str:
    """Format a SQL result as a readable summary for the LLM."""
    trunc_note = " (truncated)" if truncated else ""
    header = f"SQL: {sql}\n\nReturned {row_count} row(s){trunc_note}."
    if not rows:
        return header
    return f"{header}\n\n{_render_table(columns, rows)}"
