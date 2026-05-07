import shlex

NOTIFICATION_MODES = ("all", "important", "off")
DEFAULT_NOTIFICATION_MODE = "all"


def notification_mode(value, default=DEFAULT_NOTIFICATION_MODE) -> str:
    mode = str(value or default).strip().lower()
    return mode if mode in NOTIFICATION_MODES else default


def load_conversation_suffix(db, conversation_id: int | None) -> str:
    """Render the trailing line that lets the user jump to the
    conversation a notification came from. Shared by ``NotifyTool``
    (when the agent calls notify_user) and the run_subagent fallback
    push (when the agent did not call notify but the mode demanded a
    surface)."""
    if conversation_id is None or db is None:
        return ""
    try:
        conv = db.get_conversation(conversation_id)
    except Exception:
        return ""
    if not conv:
        return ""
    category = (conv.get("category") or "").strip()
    if not category:
        return ""
    cmd = f"/conversations {shlex.quote(category)} {conversation_id} 'Load conversation'"
    return f"\n\nLoad this conversation: `{cmd}`"
