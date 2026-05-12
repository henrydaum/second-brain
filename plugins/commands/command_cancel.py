"""Slash command plugin for `/cancel`."""

from plugins.BaseCommand import BaseCommand


class CancelCommand(BaseCommand):
    """Slash-command handler for `/cancel`."""
    name = "cancel"
    description = "Cancel the current interaction"
    category = "Conversation"

    def run(self, _args, context):
        """Execute `/cancel` for the active session."""
        runtime = getattr(context, "runtime", None)
        session_key = getattr(context, "session_key", None)
        if runtime is None or not session_key:
            return "No active session to cancel."
        result = runtime.handle_action(session_key, "cancel")
        if not getattr(result, "ok", True):
            return "Cancelled."
        if getattr(result, "messages", None):
            return "\n".join(result.messages)
        if getattr(result, "error", None):
            return result.error.get("message") if isinstance(result.error, dict) else str(result.error)
        return "Cancelled."
