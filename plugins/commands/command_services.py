"""Slash command plugin for `/services`."""

from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_services
from state_machine.conversation import FormStep


ACTIONS = ["load", "unload"]


class ServicesCommand(BaseCommand):
    """Slash-command handler for `/services`."""
    name = "services"
    description = "Select a service, then load or unload it"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        services = context.services or {}
        steps = [FormStep("service_name", "Select a service to load or unload.", True, enum=sorted((context.services or {}).keys()), columns=2)]
        if args.get("service_name"):
            steps.append(FormStep("action", f"What do you want to do with this service?\n\n{_describe(services, args['service_name'])}", True, enum=ACTIONS, enum_labels=["Load it", "Unload it"]))
        return steps

    def run(self, args, context):
        """Execute `/services` for the active session."""
        services = context.services or {}
        action, name = args.get("action"), args.get("service_name")
        if not name:
            return _show(services)
        svc = services.get(name)
        if svc is None:
            return "Unknown service."
        if action == "load":
            if svc.load() is False:
                return f"Failed to load service: {name}"
            _clear_tasks(context)
            return f"Loaded service: {name}"
        if action == "unload":
            svc.unload()
            _clear_tasks(context)
            return f"Unloaded service: {name}"
        return f"Unknown action: {action}"


def _show(services):
    """Internal helper to handle show."""
    return format_services([
        {"name": name, "loaded": getattr(svc, "loaded", False), "model_name": getattr(svc, "model_name", "")}
        for name, svc in sorted(services.items())
    ])


def _describe(services, name):
    """Internal helper to handle describe."""
    svc = services.get(name)
    if svc is None:
        return "Action"
    return f"{name}\nStatus: {'Loaded' if getattr(svc, 'loaded', False) else 'Unloaded'}\nModel: {getattr(svc, 'model_name', '') or '-'}"


def _clear_tasks(context):
    """Internal helper to clear tasks."""
    orch = getattr(context, "orchestrator", None)
    if orch and hasattr(orch, "clear_skip_cache"):
        orch.clear_skip_cache()
