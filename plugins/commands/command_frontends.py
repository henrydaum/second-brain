from plugins.BaseCommand import BaseCommand
from state_machine.conversationClass import FormStep


ACTIONS = ["enable", "disable"]
SUPPORTED = ["repl", "telegram"]


class FrontendsCommand(BaseCommand):
    name = "frontends"
    description = "Select a frontend, then enable or disable it"
    category = "System"

    def form(self, args, context):
        steps = [FormStep("frontend_name", "Frontend", True, enum=SUPPORTED)]
        if args.get("frontend_name"):
            steps.append(FormStep("action", _describe(context.config, args["frontend_name"]), True, enum=ACTIONS))
        return steps

    def run(self, args, context):
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
    enabled = set((config or {}).get("enabled_frontends", SUPPORTED))
    lines = ["Frontends:"]
    lines += [f"  {name:<10} {'Enabled' if name in enabled else 'Disabled'}" for name in frontends]
    return "\n".join(lines)


def _describe(config, name):
    enabled = set((config or {}).get("enabled_frontends", SUPPORTED))
    return f"{name}\nStatus: {'Enabled' if name in enabled else 'Disabled'}"
