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
        if result.get("error"):
            error = result["error"]
            return error.get("message") if isinstance(error, dict) else str(error)
        # Two shapes, because there are two things "cancel" can mean and the
        # kernel answers them differently. Dismissing a *form* comes back as
        # text on ``callable_output`` — that is the form reporting on its own
        # navigation, and it is this command's output too.
        said = result.get("callable_output") or result.get("messages")
        if said:
            return "\n".join(said)
        # Stopping a *turn* comes back as ``data``, because the kernel raises
        # that acknowledgement as a notification: the usual way to reach it is
        # a Cancel button, which invokes no callable and so has no output. Word
        # it here anyway — this command was invoked by name, and one that
        # answered with nothing would read as having silently failed.
        outcome = result.get("data") or {}
        if not outcome.get("cancelled"):
            return "Nothing to cancel."
        return ("Cancelled. Subagents stopped." if outcome.get("subagents_stopped")
                else "Cancelled.")
