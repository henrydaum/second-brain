"""Slash command plugin for `/new`."""

from guest.bases import BaseCommand


class NewCommand(BaseCommand):
    """Start a current-user conversation with default settings."""

    name = "new"
    description = "Start a conversation with default settings"
    category = "Conversation"
    requests = ["conv.new", "conv.list", "config.read", "session.get"]

    def run(self, sdk, args):
        """Start a fresh conversation.

        Nothing is written here. The conversation is created by the first
        message sent into it, so there is no number to report yet and running
        this twice over leaves nothing behind — which is exactly what stopped
        blank conversations accumulating.
        """
        if not _available(sdk):
            return "Conversations are not available in this context."
        if not sdk.config.read("llm_profiles", present=True):
            return (
                "No LLM is configured yet. Run /setup to add one before "
                "starting a conversation."
            )
        sdk.conv.new()
        return (
            "Started a new conversation under 'Main'.\n"
            f"Permission mode: {_mode(sdk)}"
        )


DEFAULT_MODE = "ask"


def _mode(sdk) -> str:
    """How the session answers approval dialogs, or the default if unreadable."""
    try:
        return (sdk.session.get() or {}).get("mode") or DEFAULT_MODE
    except sdk.Failed:
        return DEFAULT_MODE


def _available(sdk):
    try:
        session = sdk.session.get()
        sdk.conv.list(limit=1)
        return bool(session)
    except sdk.Failed:
        return False
