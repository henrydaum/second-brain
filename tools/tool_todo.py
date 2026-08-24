"""Per-conversation todo checklist backed by persisted plugin state."""

dependencies_files = []
dependencies_pip = []
requests = [
    "session.get", "session.push", "session.state_get", "session.state_set",
]

from guest.bases import BaseTool


MAX_TODOS = 50
_STATUSES = {"pending", "completed"}
_NAMESPACE = "todo"


class Todo(BaseTool):
    """Manage the active conversation's working checklist."""

    name = "todo"
    description = (
        "Manage this conversation's todo checklist. Use it as your working plan "
        "on any task with three or more distinct steps: add the steps up front "
        "and complete each item as soon as it is done. 'clear' drops the whole "
        "checklist when the plan is finished or abandoned. Every call returns "
        "the full current checklist."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "update", "complete", "remove", "clear", "list"],
                "description": "Operation to perform.",
            },
            "content": {
                "type": "string",
                "description": "Todo text. Required for add unless items is given; optional rewording for update.",
            },
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Bulk add: several todos at once (add only).",
            },
            "todo_id": {
                "type": "integer",
                "description": "Target todo id. Required for update, complete, and remove.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "completed"],
                "description": "New status (update only).",
            },
        },
        "required": ["operation"],
    }
    requires_services = []
    def run(self, sdk, **kwargs):
        op = str(kwargs.get("operation") or "").strip().lower()
        if op not in {"add", "update", "complete", "remove", "clear", "list"}:
            return sdk.fail(f"Unknown operation: {op}")

        try:
            session = sdk.session.get() or {}
            conversation_id = session.get("conversation_id")
            if conversation_id is None:
                return sdk.fail(
                    "Todos require a persisted conversation; none is active in this session."
                )
            state = sdk.session.state_get(namespace=_NAMESPACE) or {}
            items = [dict(item) for item in state.get("items", [])]
            next_id = int(state.get("next_id") or 1)

            if op == "add":
                texts = [
                    str(text).strip() for text in (kwargs.get("items") or [])
                    if str(text or "").strip()
                ]
                single = str(kwargs.get("content") or "").strip()
                if single:
                    texts.append(single)
                if not texts:
                    return sdk.fail("add requires 'content' or 'items'.")
                if len(items) + len(texts) > MAX_TODOS:
                    return sdk.fail(
                        f"Todo cap reached ({MAX_TODOS} per conversation). Remove items first."
                    )
                for text in texts:
                    items.append({"id": next_id, "content": text, "status": "pending"})
                    next_id += 1

            elif op == "clear":
                items = []
                next_id = 1

            elif op in {"update", "complete", "remove"}:
                todo_id = kwargs.get("todo_id")
                if not isinstance(todo_id, int) or isinstance(todo_id, bool):
                    return sdk.fail(f"{op} requires an integer 'todo_id'.")
                item = next((entry for entry in items if entry.get("id") == todo_id), None)
                if item is None:
                    return sdk.fail(f"No todo #{todo_id} in this conversation.")
                if op == "remove":
                    items.remove(item)
                else:
                    status = (
                        "completed" if op == "complete"
                        else str(kwargs.get("status") or "").strip()
                    )
                    content = str(kwargs.get("content") or "").strip()
                    if op == "update" and not status and not content:
                        return sdk.fail("update requires 'status' and/or 'content'.")
                    if status and status not in _STATUSES:
                        return sdk.fail(f"Unknown status: {status}")
                    if status:
                        item["status"] = status
                    if content:
                        item["content"] = content

            if op != "list":
                sdk.session.state_set(
                    {"items": items, "next_id": next_id}, namespace=_NAMESPACE
                )
            return self._checklist(sdk, conversation_id, items)
        except sdk.Denied as error:
            return sdk.fail(f"Todo access was denied: {error}")
        except sdk.Failed as error:
            return sdk.fail(f"Todo operation failed: {error}")

    @staticmethod
    def _checklist(sdk, conversation_id, items):
        open_count = sum(1 for item in items if item.get("status") != "completed")
        done_count = len(items) - open_count
        lines = [f"### Todos ({open_count} open, {done_count} done)"]
        if not items:
            lines.append("(empty)")
        for item in items:
            if item.get("status") == "completed":
                lines.append(f"- [x] #{item['id']} {item['content']}")
            else:
                lines.append(f"- [ ] #{item['id']} {item['content']}")
        summary = "\n".join(lines)
        sdk.session.push(summary)
        return sdk.ok(
            {"conversation_id": conversation_id, "todos": items},
            llm_summary=summary,
        )
