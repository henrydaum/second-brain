from plugins.BaseCommand import BaseCommand


class ClearCommand(BaseCommand):
    name = "clear"
    description = "Clear all messages in the current conversation"
    category = "Conversation"

    def run(self, _args, context):
        runtime = getattr(context, "runtime", None)
        session_key = getattr(context, "session_key", None)
        db = getattr(context, "db", None)
        if runtime is None or not session_key or db is None:
            return "No active session."
        session = runtime.sessions.get(session_key)
        conv_id = session.conversation_id if session else None
        if conv_id is None:
            return "No conversation loaded."
        db.clear_conversation_messages(conv_id)
        conv = db.get_conversation(conv_id) or {}
        title = (conv.get("title") or "").strip()
        if title and not title.endswith(" (cleared)"):
            db.update_conversation_title(conv_id, f"{title} (cleared)")
        runtime.close_session(session_key)
        runtime.load_conversation(session_key, conv_id)
        return "Conversation cleared."
