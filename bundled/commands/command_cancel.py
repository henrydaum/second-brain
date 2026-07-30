"""Slash command plugin for `/cancel`."""

from guest.bases import BaseCommand


class CancelCommand(BaseCommand):
    """Slash-command handler for `/cancel`."""
    name = "cancel"
    description = "Cancel the current interaction"
    category = "Conversation"
    requests = ["session.get", "session.cancel"]

    def run(self, sdk, args):
        """Execute `/cancel` for the active session."""
        if sdk.session.get() is None:
            return "No active session to cancel."
        result = sdk.session.cancel()
        if result.get("messages"):
            return "\n".join(result["messages"])
        if result.get("error"):
            error = result["error"]
            return error.get("message") if isinstance(error, dict) else str(error)
        if not result.get("ok", True):
            return "Nothing to cancel."
        return "Cancelled."
