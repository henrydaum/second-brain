"""/history — switch into a previous conversation.

Two-step picker:
    1. Pick an origin tag. The "main" tag covers untagged user
       conversations; cron-driven sessions appear as ``cron:<job>`` etc.
    2. Pick from the 15 most-recent conversations under that tag.

``/history <id>`` (handled in BaseFrontend) loads a conversation directly,
bypassing the picker. For older conversations not in the recent 15,
query the ``conversations`` table with the SQL tool.
"""

from plugins.BaseCommand import BaseCommand
from state_machine.conversationClass import FormStep
from state_machine.persistence import latest_state


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
        _add_agent_labels(db, rows)
        if rows:
            enum = [str(r.get("id")) for r in rows]
            labels = [_encode(r) for r in rows]
        else:
            enum = ["(no conversations)"]
            labels = ["(no conversations)"]
        steps.append(FormStep("conversation_id", f"Recent under '{picked}' ({len(rows)})", True, enum=enum, enum_labels=labels, columns=1))
        return steps

    def run(self, args, context):
        cid = _decode(args.get("conversation_id"))
        return "History selection should be loaded by the runtime." if cid is not None else "No conversation selected."


def _tag_label(origin) -> str:
    return _MAIN if origin in (None, "") else origin


def _origin_from_label(label: str):
    return "" if label == _MAIN else label


def _encode(row: dict) -> str:
    cid = row.get("id")
    title = (row.get("title") or "").strip() or "(untitled)"
    agent = (row.get("agent_profile") or "").strip()
    return f"#{cid} {title}{f' [agent: {agent}]' if agent else ''}"


def _add_agent_labels(db, rows: list[dict]) -> None:
    for row in rows:
        marker = latest_state(db.get_conversation_messages(row.get("id"))) or {}
        row["agent_profile"] = marker.get("profile_override") or marker.get("active_agent_profile") or ""


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
