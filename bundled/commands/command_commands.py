"""Slash command plugin for `/commands`."""

from guest.bases import BaseCommand


_HELP_SECTIONS = [
    "Conversation",
    "System",
    "Services & Tools",
    "Tasks",
    "Config & System",
    "Other",
]


class CommandsCommand(BaseCommand):
    """Slash-command handler for `/commands`."""
    name = "commands"
    description = "List available commands"
    category = "Conversation"
    requests = ["command.list"]

    def run(self, sdk, args):
        """Execute `/commands` for the active session."""
        try:
            commands = sdk.commands.list(details=True, visible=True)
        except sdk.Failed:
            return "No command registry is available."

        by_category = {}
        for command in commands:
            by_category.setdefault(command["category"], []).append(command)
        ordered = [
            category for category in _HELP_SECTIONS if category in by_category
        ] + [
            category for category in by_category
            if category not in _HELP_SECTIONS
        ]

        lines = ["Commands:"]
        for category in ordered:
            rows = []
            for command in by_category[category]:
                hint = _arg_hint(command.get("form") or [])
                call = "/" + command["name"] + (f" {hint}" if hint else "")
                rows.append((call, command["description"]))
            table = sdk.md.table(["Command", "Description"], rows).lstrip("\n")
            lines += ["", f"**{category}**", "", table]
        return "\n".join(lines)


def _arg_hint(form):
    """Render required and optional form fields like the native registry."""
    return " ".join(
        f"<{step['name']}>" if step.get("required") else f"[{step['name']}]"
        for step in form
    )
