"""Slash command plugin for `/frontends`."""

from plugins.BaseCommand import BaseCommand
from state_machine.conversation import FormStep


ACTIONS = ["enable", "disable"]
SUPPORTED = ["repl", "telegram"]


class FrontendsCommand(BaseCommand):
    """Slash-command handler for `/frontends`."""
    name = "frontends"
    description = "Select a frontend, then enable or disable it"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        steps = [FormStep("frontend_name", "Select a frontend to enable or disable.", True, enum=SUPPORTED, columns=2)]
        if args.get("frontend_name"):
            steps.append(FormStep("action", f"What do you want to do with this frontend?\n\n{_describe(context.config, args['frontend_name'])}", True, enum=ACTIONS, enum_labels=["Enable it", "Disable it"]))
        return steps

    def run(self, args, context):
        """Execute `/frontends` for the active session."""
        action, name = args.get("action"), args.get("frontend_name")
        if not name:
            return _show(context.config)
        names = set((context.config or {}).get("enabled_frontends", SUPPORTED))
        if name not in SUPPORTED:
            return "Unknown frontend."
        if action == "enable":
            names.add(name)
        elif action == "disable":
            if name in names and len(names) == 1:
                return "Cannot disable the last enabled frontend."
            names.discard(name)
        else:
            return f"Unknown action: {action}"
        context.config["enabled_frontends"] = sorted(names)
        from config import config_manager
        config_manager.save(context.config)
        return f"{'Enabled' if action == 'enable' else 'Disabled'} frontend: {name}. Restart required."


def _show(config, frontends=SUPPORTED):
    """Internal helper to handle show."""
    enabled = set((config or {}).get("enabled_frontends", SUPPORTED))
    lines = ["Frontends:"]
    lines += [f"  {name:<10} {'Enabled' if name in enabled else 'Disabled'}" for name in frontends]
    return "\n".join(lines)


def _describe(config, name):
    """Internal helper to handle describe."""
    enabled = set((config or {}).get("enabled_frontends", SUPPORTED))
    return f"{name}\nStatus: {'Enabled' if name in enabled else 'Disabled'}"
