"""Slash command plugin for `/restart`."""

from guest.bases import BaseCommand


class RestartCommand(BaseCommand):
    """Restart Second Brain in place."""

    name = "restart"
    description = "Restart the app"
    category = "System"
    requests = ["app.stop"]
    # See command_quit: declared up front so the state machine asks before the
    # body runs, rather than the body asking mid-shutdown.
    require_approval = True
    approval_actor_id = "user"

    def run(self, sdk, args):
        """Execute `/restart` for the active session."""
        try:
            return sdk.app.stop(restart=True)
        except sdk.Failed as exc:
            return f"Could not restart: {exc.error}"
