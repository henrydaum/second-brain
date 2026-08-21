"""
SQL Query tool.

Gives both humans (via REPL) and the LLM (via function calling) access to the
local database.

Sandboxed, and the approval machinery went away with the migration — not
because writing got safer, but because the kernel now answers the question
this tool used to ask. ``db.write`` and ``db.define`` refuse every table the
kernel maintains outright (``users``, ``conversations``, ``action_ledger``,
``files``, the task tables), naming the Request that does the job properly, so
the statements that needed a dialog can no longer be issued at all. What
remains reaches only plugin-owned tables, which is what ``db.write`` is for
and which nobody needs to approve.

Reads are broad on purpose: a plugin that reads everything still cannot send
anything anywhere, because egress is gated. What *is* narrowed is whose rows —
user-owned tables are read through their ``my_`` name, which the kernel expands
to the current user, and reading the base table is refused.
"""

dependencies_files = []
dependencies_pip = []
requests = ["db.query", "db.write", "db.define"]

from difflib import get_close_matches

from guest.bases import BaseTool

# Statements that never mutate state. Anything else is routed to db.write or
# db.define, which carry their own refusals — so misclassifying here costs a
# clearer error message, never an unguarded write.
READ_ONLY_PREFIXES = ("select", "pragma", "explain", "with")
DDL_PREFIXES = ("create", "alter", "drop")

MAX_CELL = 500
MAX_ROWS = 500  # The kernel's own cap on what a db.query answer may carry.


class SQLQuery(BaseTool):
    """Sqlquery."""
    name = "sql_query"
    description = (
        "Run SQL against the local database. SELECT / PRAGMA / EXPLAIN statements "
        "read; use them to inspect schema, file metadata, pipeline state, extracted "
        "text and stored conversations.\n\n"
        "Two rules the kernel enforces, so write the SQL that follows them:\n"
        "- User-owned tables are read through their 'my_' name — 'my_conversations', "
        "'my_action_ledger'. Reading the base table is refused, because it holds "
        "other people's rows.\n"
        "- Kernel tables (users, conversations, conversation_messages, action_ledger, "
        "files, registered_tasks, task_queue, task_runs) cannot be written through "
        "this tool at all. Use the dedicated tool or command for that change. "
        "INSERT / UPDATE / DELETE / DDL against a table your own plugin created "
        "works normally.\n\n"
        "Useful queries:\n"
        "- SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\n"
        "- PRAGMA table_info(table_name)\n"
        "- SELECT title, updated_at FROM my_conversations ORDER BY updated_at DESC LIMIT 10\n"
        "- SELECT modality, COUNT(*) FROM files GROUP BY modality"
    )
    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "A single SQL statement. Reads run immediately; writes reach "
                    "plugin-owned tables only."
                ),
            },
            "narration": {
                "type": "string",
                "description": (
                    "A few words on what you are looking up and why, shown to "
                    "the user beside the call. E.g. 'counting the notes written "
                    "since April'."
                ),
            },
        },
        "required": ["sql"],
    }
    requires_services = []

    # No cue declared, for the same reason as run_script: the table list is
    # live, and a CREATE from anywhere has to appear. The default rung is the
    # one that notices.

    def agent_prompt(self, sdk) -> str:
        """Point the agent at the live table list and the two scoping rules."""
        return (
            f"""## Querying the database
sql_query reads the local SQLite database. Two rules are enforced by the kernel rather than by convention, so writing SQL that ignores them produces a refusal, not a result:

1. **Rows you own are read through a 'my_' name.** `SELECT * FROM my_conversations` returns the current user's conversations; `SELECT * FROM conversations` is refused, because that table holds every user's rows. The same applies to `my_action_ledger`.
2. **Kernel tables are not writable through SQL.** users, conversations, conversation_messages, action_ledger, files, registered_tasks, task_queue and task_runs each have a dedicated tool or command that carries the right access checks. A table your own plugin created with a CREATE statement stays freely writable.

## Conversation history
Past conversations live in `my_conversations` and `conversation_messages`. Compacted history is no longer in your context but remains queryable there.

`role` does not tell you who wrote a row. The kernel writes `role='user'` rows the person never typed — a cancel notice, a doorman's note, the summary bridge left by compaction, a note that a slash command ran — and every one of them carries a non-empty `author`. When you want what the *user* actually said, add `AND COALESCE(author, '') = ''`; without it you will read the kernel's own bookkeeping back as their words. (`role='system'` is separate again: those are state markers, not messages.)

## Database tables (inspect with sql_query)
{_table_list(sdk)}"""
        )

    def run(self, sdk, **kwargs):
        """Run sqlquery."""
        sql = (kwargs.get("sql") or "").strip()
        if not sql:
            return sdk.fail("No SQL provided.")

        if _is_read_only(sql):
            return self._read(sdk, sql)
        return self._write(sdk, sql)

    def _read(self, sdk, sql: str):
        """Execute a read through the row-scoped ``db.query``."""
        try:
            rows = sdk.db.query(sql)
        except sdk.Denied as refused:
            # A scoping refusal already names the right table, so passing it
            # through unchanged is more useful than any wrapper.
            return sdk.fail(str(refused))
        except sdk.Failed as failed:
            return sdk.fail(failed.error + _schema_hint(sdk, failed.error))

        header = f"SQL: {sql}\n\nReturned {len(rows)} row(s)."
        if len(rows) >= MAX_ROWS:
            # The kernel caps what may cross; saying so is what stops the
            # model reading a capped answer as the whole table.
            header += (f" That is the cap — there may be more. "
                       f"Add LIMIT/OFFSET or an aggregate to see the rest.")
        summary = header if not rows else f"{header}\n\n{_render_table(rows)}"
        return sdk.ok({"rows": rows, "row_count": len(rows), "wrote": False},
                      llm_summary=summary)

    def _write(self, sdk, sql: str):
        """Execute a mutation, which reaches plugin-owned tables only."""
        define = sql.lower().lstrip().startswith(DDL_PREFIXES)
        try:
            if define:
                sdk.db.define(sql)
            else:
                sdk.db.write(sql)
        except sdk.Denied as refused:
            # The kernel's message names the Request that does this properly.
            # Surfacing it verbatim is what stops the model retrying the same
            # statement in a slightly different shape.
            return sdk.fail(str(refused))
        except sdk.Failed as failed:
            return sdk.fail(failed.error + _schema_hint(sdk, failed.error))

        return sdk.ok({"wrote": True, "ddl": define},
                      llm_summary=f"SQL: {sql}\n\nStatement executed.")


def _is_read_only(sql: str) -> bool:
    """Whether ``sql`` only reads.

    A CTE (``WITH ...``) counts as a read: it may wrap a write, and if it does
    the kernel's own table check refuses it. The worst case of guessing wrong
    here is a less helpful error message.
    """
    normalized = " ".join(sql.strip().split()).lower()
    return normalized.startswith(READ_ONLY_PREFIXES)


def _table_list(sdk) -> str:
    """Every table name, for the agent prompt."""
    try:
        rows = sdk.db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    except Exception:
        return "Could not read the table list."
    names = [str(row.get("name")) for row in rows if row.get("name")]
    return ", ".join(names) if names else "No tables yet."


def _schema_hint(sdk, error_msg: str) -> str:
    """Name the tables, and guess at what the failing statement meant.

    A model that misremembers a table or column name will otherwise retry by
    guessing again. Listing what exists turns the next attempt into a lookup.
    """
    try:
        rows = sdk.db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    except Exception:
        return ""
    tables = [str(row.get("name")) for row in rows if row.get("name")]
    if not tables:
        return ""

    lines = [f"\n\nAvailable tables: {', '.join(tables)}"]

    bad_table = _after(error_msg, "no such table:")
    if bad_table:
        guesses = get_close_matches(bad_table, tables, n=2, cutoff=0.5)
        if guesses:
            lines.append(f"Did you mean: {', '.join(guesses)}?")
        return "\n".join(lines)

    bad_column = _after(error_msg, "no such column:")
    if bad_column:
        bad_column = bad_column.split(".")[-1]
        suggestions = []
        for table in tables:
            columns = _columns(sdk, table)
            if not columns:
                continue
            if bad_column in columns:
                suggestions.append(f"{table}: {', '.join(columns)}")
            else:
                close = get_close_matches(bad_column, columns, n=1, cutoff=0.6)
                if close:
                    suggestions.append(
                        f"{table} has '{close[0]}' (cols: {', '.join(columns)})")
        if suggestions:
            lines.append("Column hints:")
            lines.extend("  " + s for s in suggestions[:5])

    return "\n".join(lines)


def _columns(sdk, table: str) -> list:
    """Column names for one table, or [] if it cannot be inspected."""
    try:
        return [str(row.get("name")) for row in
                sdk.db.query(f"PRAGMA table_info({table})") if row.get("name")]
    except Exception:
        return []


def _after(text: str, marker: str) -> str:
    """The first whitespace-delimited token following ``marker``.

    A small parser rather than a regex, because the only thing being read is
    SQLite's own fixed error prefix.
    """
    lowered = (text or "").lower()
    at = lowered.find(marker)
    if at == -1:
        return ""
    return (text[at + len(marker):].strip().split() or [""])[0]


def _render_table(rows: list) -> str:
    """Rows of dicts as a simple aligned text table.

    Columns come from the first row: ``db.query`` answers with plain dicts,
    having dropped the cursor description on the way across the wire.
    """
    columns = list(rows[0].keys())

    def cell(value, limit=MAX_CELL):
        text = str(value)
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...[truncated {len(text) - limit} chars]"

    head = " | ".join(cell(c, 80) for c in columns)
    rule = " | ".join("-" * max(len(str(c)), 3) for c in columns)
    body = [" | ".join(cell(row.get(c)) for c in columns) for row in rows]
    return "\n".join([head, rule] + body)
