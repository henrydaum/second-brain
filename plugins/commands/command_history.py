"""/history — switch into a previous conversation.

Two-step picker:
    1. Pick an origin tag. The "main" tag covers untagged user
       conversations; cron-driven sessions appear as ``cron:<job>`` etc.
    2. Pick from the 15 most-recent conversations under that tag.

``/history <id>`` (handled in BaseFrontend) loads a conversation directly,
bypassing the picker. For older conversations not in the recent 15,
query the ``conversations`` table with the SQL tool.
"""

from datetime import datetime, timezone

from plugins.BaseCommand import BaseCommand
from state_machine.conversationClass import FormStep


_LIMIT = 15
_MAIN = "main"


class HistoryCommand(BaseCommand):
    name = "history"
    description = "Switch into a recent conversation"
    category = "Conversation"

    def form(self, args, context):
        if args.get("conversation_id"):
            return []
        db = getattr(context, "db", None)
        origins = db.list_conversation_origins() if db else []
        if not origins:
            return []
        tag_enum = [_tag_label(o) for o in origins]
        steps = [FormStep("origin", "Tag", True, enum=tag_enum, columns=1)]

        picked = args.get("origin")
        if not picked:
            return steps

        rows, _ = db.list_conversations_page(offset=0, limit=_LIMIT, origin=_origin_from_label(picked))
        enum = [_encode(r) for r in rows] or ["(no conversations)"]
        steps.append(FormStep("conversation_id", f"Recent under '{picked}' ({len(rows)})", True, enum=enum, columns=1))
        return steps

    def run(self, args, context):
        cid = _decode(args.get("conversation_id"))
        if cid is None:
            return "No conversation selected."

        runtime = getattr(context, "runtime", None)
        session_key = getattr(context, "session_key", None)
        if not runtime or not session_key:
            return "Cannot switch conversations from this context."
        runtime.handle_action(session_key, "load_history", {"conversation_id": cid})
        return None


def _tag_label(origin) -> str:
    return _MAIN if origin in (None, "") else origin


def _origin_from_label(label: str):
    return "" if label == _MAIN else label


def _encode(row: dict) -> str:
    cid = row.get("id")
    title = (row.get("title") or "").strip() or "(untitled)"
    when = _ago(row.get("updated_at"))
    suffix = f" ({when})" if when else ""
    return f"#{cid} {title}{suffix}"


def _decode(value) -> int | None:
    if value in (None, "", "(no conversations)"):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith("#"):
        text = text[1:]
    head = text.split(" ", 1)[0].strip()
    try:
        return int(head)
    except (TypeError, ValueError):
        return None


def _ago(ts) -> str:
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return ""
    now = datetime.now(timezone.utc).timestamp()
    delta = max(0, now - ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)}d ago"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
