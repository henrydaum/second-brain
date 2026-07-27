"""Slash command plugin for `/clear`."""

from guest.bases import BaseCommand


class ClearCommand(BaseCommand):
    """Slash-command handler for `/clear`."""
    name = "clear"
    description = "Clear all messages in the current conversation"
    category = "Conversation"
    requests = ["session.get", "conv.clear"]

    def run(self, sdk, args):
        """Execute `/clear` for the active session."""
        try:
            session = sdk.session.get()
        except sdk.Failed:
            return "No active session."
        if session is None:
            return "No active session."
        conv_id = session.get("conversation_id")
        if conv_id is None:
            return "No conversation loaded."
        try:
            sdk.conv.clear(conv_id)
        except sdk.Failed as exc:
            if "database" in exc.error and "not available" in exc.error:
                return "No active session."
            raise
        return "Conversation cleared."
