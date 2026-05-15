"""Slash command plugin for `/plan`."""

from plugins.BaseCommand import BaseCommand


class PlanCommand(BaseCommand):
    """Slash-command handler for `/plan`."""
    name = "plan"
    description = "Toggle plan mode for this conversation"
    category = "Conversation"

    def run(self, _args, context):
        """Toggle plan mode for the active session."""
        runtime = getattr(context, "runtime", None)
        session_key = getattr(context, "session_key", None)
        if runtime is None or not session_key:
            return "No active session."
        session = runtime.sessions.get(session_key)
        enabled = not bool(getattr(session, "plan_mode", False))
        runtime.set_plan_mode(session_key, enabled)
        return f"Plan mode {'on' if enabled else 'off'}."
