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
        if not registry:
            return "No command registry is available."
        from plugins.frontends.helpers.command_registry import frontend_command_filter
        return registry.help_text(frontend_command_filter(context.config, _frontend_of(context)))


def _frontend_of(context):
    """Resolve which frontend the calling session belongs to, if known."""
    runtime = getattr(context, "runtime", None)
    session = getattr(runtime, "sessions", {}).get(getattr(context, "session_key", None)) if runtime else None
    return getattr(session, "frontend_name", None)
