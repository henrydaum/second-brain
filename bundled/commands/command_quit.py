"""Slash command plugin for `/quit`."""

from guest.bases import BaseCommand


class QuitCommand(BaseCommand):
    """Shut Second Brain down."""

    name = "quit"
    description = "Shutdown"
    category = "Conversation"
    requests = ["app.stop"]
    # Answered by the state machine before the body runs, which is the right
    # doorway for a command with one consequential act: the grant is stated and
    # given up front rather than interrupting a half-finished shutdown.
    require_approval = True
    approval_actor_id = "user"

    def run(self, sdk, args):
        """Execute `/quit` for the active session."""
        try:
            return sdk.app.stop()
        except sdk.Failed as exc:
            return f"Could not shut down: {exc.error}"
