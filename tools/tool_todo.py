"""Per-conversation todo checklist — the agent's working plan for multi-step tasks.

Todos live in a ``todos`` table keyed by conversation_id with ON DELETE
CASCADE, so they are cleaned up with their conversation (explicit delete or
retention prune) — no separate retention knob.
"""

dependencies_files = []
dependencies_pip = []

import time

from plugins.BaseTool import BaseTool, ToolResult

MAX_TODOS = 50

_DDL = """
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'completed')),
    position INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_todos_conversation ON todos(conversation_id);
"""

_STATUSES = {"pending", "in_progress", "completed"}


def _conversation_id(context) -> int | None:
    """Current conversation id via the session, or None."""
    runtime, key = getattr(context, "runtime", None), getattr(context, "session_key", None)
    session = getattr(runtime, "sessions", {}).get(key) if runtime and key else None
    return getattr(session, "conversation_id", None)


class Todo(BaseTool):
    """Todo."""
    name = "todo"
    description = (
        "Manage this conversation's todo checklist. Use it as your working plan on "
        "multi-step tasks: add the steps up front, mark exactly one in_progress at a "
        "time, and complete each item as soon as it is done. Every call returns the "
        "full current checklist."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["add", "update", "complete", "remove", "list"], "description": "Operation to perform."},
            "content": {"type": "string", "description": "Todo text. Required for add (unless items is given); optional rewording for update."},
            "items": {"type": "array", "items": {"type": "string"}, "description": "Bulk add: several todos at once (add only)."},
            "todo_id": {"type": "integer", "description": "Target todo id. Required for update, complete, and remove."},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"], "description": "New status (update only)."},
        },
        "required": ["operation"],
    }
    requires_services = []
    max_calls = 20
    background_safe = True
    agent_prompt = (
        "## Todos\n"
        "For any task with 3+ distinct steps, plan with the todo tool: add the steps, "
        "keep exactly one in_progress, and mark items completed immediately when done."
    )

    def run(self, context, **kwargs) -> ToolResult:
        """Run todo."""
        op = (kwargs.get("operation") or "").strip().lower()
        if op not in {"add", "update", "complete", "remove", "list"}:
            return ToolResult.failed(f"Unknown operation: {op}")
        db = getattr(context, "db", None)
        if db is None:
            return ToolResult.failed("No database available.")
        cid = _conversation_id(context)
        if cid is None:
            return ToolResult.failed("Todos require a persisted conversation; none is active in this session.")

        db.ensure_output_table("todo", _DDL)
        now = time.time()

        if op == "add":
            texts = [t.strip() for t in (kwargs.get("items") or []) if (t or "").strip()]
            single = (kwargs.get("content") or "").strip()
            if single:
                texts.append(single)
            if not texts:
                return ToolResult.failed("add requires 'content' or 'items'.")
            with db.lock:
                count = db.conn.execute(
                    "SELECT COUNT(*) FROM todos WHERE conversation_id = ?", (cid,)).fetchone()[0]
                if count + len(texts) > MAX_TODOS:
                    return ToolResult.failed(f"Todo cap reached ({MAX_TODOS} per conversation). Remove or complete items first.")
                pos = db.conn.execute(
                    "SELECT COALESCE(MAX(position), 0) FROM todos WHERE conversation_id = ?", (cid,)).fetchone()[0]
                for text in texts:
                    pos += 1
                    db.conn.execute(
                        "INSERT INTO todos (conversation_id, content, status, position, created_at, updated_at) "
                        "VALUES (?, ?, 'pending', ?, ?, ?)", (cid, text, pos, now, now))
                db.conn.commit()

        elif op in {"update", "complete", "remove"}:
            todo_id = kwargs.get("todo_id")
            if not isinstance(todo_id, int):
                return ToolResult.failed(f"{op} requires an integer 'todo_id'.")
            with db.lock:
                row = db.conn.execute(
                    "SELECT id FROM todos WHERE id = ? AND conversation_id = ?", (todo_id, cid)).fetchone()
                if row is None:
                    return ToolResult.failed(f"No todo #{todo_id} in this conversation.")
                if op == "remove":
                    db.conn.execute("DELETE FROM todos WHERE id = ? AND conversation_id = ?", (todo_id, cid))
                else:
                    status = "completed" if op == "complete" else (kwargs.get("status") or "").strip()
                    content = (kwargs.get("content") or "").strip()
                    if op == "update" and not status and not content:
                        return ToolResult.failed("update requires 'status' and/or 'content'.")
                    if status and status not in _STATUSES:
                        return ToolResult.failed(f"Unknown status: {status}")
                    if status:
                        db.conn.execute(
                            "UPDATE todos SET status = ?, updated_at = ? WHERE id = ? AND conversation_id = ?",
                            (status, now, todo_id, cid))
                    if content:
                        db.conn.execute(
                            "UPDATE todos SET content = ?, updated_at = ? WHERE id = ? AND conversation_id = ?",
                            (content, now, todo_id, cid))
                db.conn.commit()

        return self._checklist(db, cid)

    @staticmethod
    def _checklist(db, cid) -> ToolResult:
        """Render the conversation's full checklist."""
        with db.lock:
            rows = db.conn.execute(
                "SELECT id, content, status FROM todos WHERE conversation_id = ? ORDER BY position, id",
                (cid,)).fetchall()
        items = [{"id": r[0], "content": r[1], "status": r[2]} for r in rows]
        open_n = sum(1 for i in items if i["status"] != "completed")
        done_n = len(items) - open_n
        lines = [f"### Todos ({open_n} open, {done_n} done)"]
        if not items:
            lines.append("(empty)")
        for i in items:
            if i["status"] == "completed":
                lines.append(f"- [x] #{i['id']} {i['content']}")
            elif i["status"] == "in_progress":
                lines.append(f"- [ ] #{i['id']} **{i['content']}** (in progress)")
            else:
                lines.append(f"- [ ] #{i['id']} {i['content']}")
        return ToolResult(True, data={"conversation_id": cid, "todos": items},
                          llm_summary="\n".join(lines))
