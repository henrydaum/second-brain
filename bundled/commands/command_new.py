"""Slash command plugin for `/new`."""

from guest.bases import BaseCommand


class NewCommand(BaseCommand):
    """Start a current-user conversation with default settings."""

    name = "new"
    description = "Start a conversation with default settings"
    category = "Conversation"
    requests = ["conv.create", "conv.list", "config.read", "session.get"]

    def run(self, sdk, args):
        """Create and switch to a new Main conversation."""
        if not _available(sdk):
            return "Conversations are not available in this context."
        if not sdk.config.read("llm_profiles", present=True):
            return (
                "No LLM is configured yet. Run /setup to add one before "
                "starting a conversation."
            )
        before = _mode(sdk)
        created = sdk.conv.create(
            "New conversation (Main)",
            category=None,
            activate=True,
        )
        if not created:
            return "Failed to create conversation."
        return (
            f"Started new conversation #{created['id']} under 'Main'.\n"
            f"Agent: {created.get('profile') or 'default'}\n"
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
