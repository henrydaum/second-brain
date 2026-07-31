"""Slash command plugin for `/commands`."""

from guest.bases import BaseCommand


# Four sections, in the order a person meets them: what you are doing now,
# what the app can do, what it does unattended, and the machinery.
#
# There were six, and two of them were never used by anything ("Services &
# Tools", "Other") while two more overlapped so plainly that /config and
# /update sat in "Config & System" with /setup one section away in "System".
# Nine of twenty commands were "System", which is another way of saying the
# taxonomy had stopped sorting anything. /quit and /restart were filed under
# "Conversation", where they read as ways to end a *chat*.
#
# ``category`` is a per-command attribute, so this list only orders the
# sections and names the ones a bundled command actually uses; an unknown
# category from an installed command still renders, after these.
_HELP_SECTIONS = [
    "Conversation",
    "Capabilities",
    "Automation",
    "System",
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
