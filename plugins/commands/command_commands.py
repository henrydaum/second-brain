"""Slash command plugin for `/commands`."""

from plugins.BaseCommand import BaseCommand


class CommandsCommand(BaseCommand):
    """Slash-command handler for `/commands`."""
    name = "commands"
    description = "List available commands"
    category = "Conversation"

    def run(self, _args, context):
        """Execute `/commands` for the active session."""
        registry = getattr(context, "command_registry", None)
        return registry.help_text() if registry else "No command registry is available."
