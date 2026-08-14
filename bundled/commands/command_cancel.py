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
        # ``callable_output`` first: the kernel's answer is what this command
        # returns, so it comes back on the channel a command answers on.
        # ``messages`` stays a fallback rather than being dropped — a future
        # cancel path that genuinely speaks in the conversation would put its
        # line there, and losing it would be silent.
        said = result.get("callable_output") or result.get("messages")
        if said:
            return "\n".join(said)
        if result.get("error"):
            error = result["error"]
            return error.get("message") if isinstance(error, dict) else str(error)
        if not result.get("ok", True):
            return "Nothing to cancel."
        return "Cancelled."
