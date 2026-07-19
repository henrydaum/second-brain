"""Slash command plugin for `/services`."""

from plugins.BaseCommand import BaseCommand
from plugins.BaseService import is_extension_service, is_user_managed_service, service_lifecycle
from plugins.commands.helpers.setting_links import quicklink_run, quicklink_value_steps, quicklinks, setting_rows
from plugins.frontends.helpers.formatters import detail_card, format_services
from state_machine.conversation import FormStep


class ServicesCommand(BaseCommand):
    """Slash-command handler for `/services`."""
    name = "services"
    description = "Inspect services and load or unload managed ones"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        services = context.services or {}
        steps = [FormStep("service_name", "Select a service.", True, enum=sorted((context.services or {}).keys()), columns=2)]
        name = args.get("service_name")
        svc = services.get(name) if name else None
        if svc is None:
            return steps
        actions, labels = _actions_for(context, name, svc)
        if actions:
            steps.append(FormStep("action", f"What do you want to do with this service?\n\n{_describe(services, name, context)}", True, enum=actions, enum_labels=labels))
        steps += quicklink_value_steps(args.get("action"), context)
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
        if not action:
            return _describe(services, name, context)
        handled = quicklink_run(action, args, context)
        if handled is not None:
            return handled
        if not is_user_managed_service(svc):
            return f"{name} is an installed extension and is loaded automatically."
        if action in ("toggle_loaded", "load", "unload"):
            load = not getattr(svc, "loaded", False) if action == "toggle_loaded" else action == "load"
            if load:
                if svc.load() is False:
                    return f"Failed to load service: {name}"
                _clear_tasks(context)
                return f"Loaded service: {name}"
            svc.unload()
            _clear_tasks(context)
            return f"Unloaded service: {name}"
        if action == "toggle_autoload":
            return _toggle_autoload(context, name)
        return f"Unknown action: {action}"


def _actions_for(context, name, svc):
    """Action enum + labels for one service: lifecycle toggles for managed
    services, plus Edit-setting quicklinks for anything declaring settings."""
    actions, labels = [], []
    if is_user_managed_service(svc):
        loaded = getattr(svc, "loaded", False)
        autoloaded = name in ((getattr(context, "config", None) or {}).get("autoload_services") or [])
        actions += ["toggle_loaded", "toggle_autoload"]
        labels += ["Unload it" if loaded else "Load it",
                   "Don't autoload on startup" if autoloaded else "Autoload on startup"]
    links, link_labels = quicklinks(svc)
    return actions + links, labels + link_labels


def _toggle_autoload(context, name):
    """Add or remove a service from config's autoload_services."""
    from config import config_manager
    config = context.config
    names = [str(n) for n in (config.get("autoload_services") or [])]
    enabled = name not in names
    names = sorted(set(names) | {name}) if enabled else [n for n in names if n != name]
    config["autoload_services"] = names
    config_manager.save(config)
    runtime = getattr(context, "runtime", None)
    if runtime is not None and getattr(runtime, "config", None) is not None:
        runtime.config["autoload_services"] = names
    return f"{name} will {'now' if enabled else 'no longer'} load automatically on startup."


def _show(services):
    """Internal helper to handle show."""
    return format_services([
        {"name": name, "loaded": getattr(svc, "loaded", False), "model_name": getattr(svc, "model_name", ""), "lifecycle": service_lifecycle(svc)}
        for name, svc in sorted(services.items())
    ])


def _describe(services, name, context=None):
    """Internal helper to handle describe."""
    svc = services.get(name)
    if svc is None:
        return "Action"
    status = "Extension" if is_extension_service(svc) else ("Loaded" if getattr(svc, 'loaded', False) else "Unloaded")
    pairs = [
        ("Status", status),
        ("Model", getattr(svc, "model_name", "") or "-"),
    ]
    return detail_card(name, pairs + setting_rows(svc, context))


def _clear_tasks(context):
    """Internal helper to clear tasks."""
    orch = getattr(context, "orchestrator", None)
    if orch and hasattr(orch, "clear_skip_cache"):
        orch.clear_skip_cache()
