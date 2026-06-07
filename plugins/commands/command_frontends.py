"""Slash command plugin for `/frontends`."""

import json

from config import config_manager
from plugins.BaseCommand import BaseCommand
from state_machine.conversation import FormStep


ACTIONS = ["configure", "enable", "disable"]
ACTION_LABELS = ["Edit", "Enable", "Disable"]

# Editable fields of a frontend profile. Each profile pins the agent profile
# used by sessions on that frontend and narrows which slash commands the user
# may run there.
FIELDS = ["agent_profile", "whitelist_or_blacklist_commands", "commands_list"]
FIELD_LABELS = ["Agent profile", "Command mode", "Command list"]
DEFAULT_PROFILE = {
    "agent_profile": "default",
    "whitelist_or_blacklist_commands": "blacklist",
    "commands_list": [],
}


class FrontendsCommand(BaseCommand):
    """Slash-command handler for `/frontends`."""
    name = "frontends"
    description = "Enable/disable a frontend or configure its access profile"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        names = _frontends(context)
        steps = [FormStep("frontend_name", "Select a frontend.", True, enum=names, columns=2)]
        name = args.get("frontend_name")
        if name:
            steps.append(FormStep("action", f"What do you want to do with this frontend?\n\n{_describe(context.config, name)}", True, enum=ACTIONS, enum_labels=ACTION_LABELS))
        if args.get("action") == "configure":
            steps.append(FormStep("field", "Choose which part of the frontend profile to edit.", True, enum=FIELDS, enum_labels=FIELD_LABELS))
            field = args.get("field")
            if field:
                steps.append(_value_step(field, context))
        return steps

    def run(self, args, context):
        """Execute `/frontends` for the active session."""
        action, name = args.get("action"), args.get("frontend_name")
        if not name:
            return _show(context)
        if action in ("enable", "disable"):
            return _toggle(context, name, action)
        if action == "configure":
            return _configure(context, name, args.get("field"), args.get("value"))
        return f"Unknown action: {action}"


def _write_through(context, key, value):
    """Persist a config change and update the canonical runtime config in place.

    context.config is a per-call copy used only to build the save payload; the
    canonical dict on the runtime is what later /frontends calls copy from, so
    update it too or the change won't show until restart.
    """
    config_manager.save(context.config)
    runtime = getattr(context, "runtime", None)
    if runtime is not None and getattr(runtime, "config", None) is not None:
        runtime.config[key] = value


def _toggle(context, name, action):
    """Enable or disable a frontend (the frontend itself starts on restart)."""
    config = context.config
    names = set((config or {}).get("enabled_frontends", []))
    if action == "enable":
        names.add(name)
    else:
        if name in names and len(names) == 1:
            return "Cannot disable the last enabled frontend."
        names.discard(name)
    config["enabled_frontends"] = sorted(names)
    _write_through(context, "enabled_frontends", config["enabled_frontends"])
    return f"{'Enabled' if action == 'enable' else 'Disabled'} frontend: {name}. Restart required."


def _configure(context, name, field, value):
    """Write one frontend-profile field. Takes effect on the next turn."""
    config = context.config
    if field not in FIELDS:
        return f"Unknown field: {field}"
    profiles = config.setdefault("frontend_profiles", {})
    profile = profiles.setdefault(name, dict(DEFAULT_PROFILE))
    if field == "whitelist_or_blacklist_commands" and value not in ("whitelist", "blacklist"):
        return "Command mode must be 'whitelist' or 'blacklist'."
    profile[field] = _coerce(field, value)
    _write_through(context, "frontend_profiles", config["frontend_profiles"])
    note = ""
    if field == "whitelist_or_blacklist_commands" and value == "whitelist" and not profile.get("commands_list"):
        note = "\nNote: whitelist is empty — every command is now blocked on this frontend."
    return f"Updated {name} profile: {FIELD_LABELS[FIELDS.index(field)]} → {_render_value(field, profile[field])}{note}"


def _coerce(field, value):
    """Internal helper to handle coerce."""
    if field == "commands_list":
        return value if isinstance(value, list) else json.loads(value or "[]")
    return "" if value is None else str(value)


def _value_step(field, context):
    """Build the value form step for the chosen profile field."""
    if field == "agent_profile":
        profiles = ["default", *sorted((context.config.get("agent_profiles", {}) or {}).keys())]
        # "default" already present from the sentinel; de-dupe while ordering.
        profiles = ["default", *[p for p in profiles if p != "default"]]
        return FormStep("value", "Choose the agent profile sessions on this frontend should use. 'default' follows the global active profile.", True, enum=profiles, default="default")
    if field == "whitelist_or_blacklist_commands":
        return FormStep("value", "Blacklist blocks the listed commands; whitelist allows only the listed commands.", True, enum=["blacklist", "whitelist"], enum_labels=["Blacklist commands", "Whitelist commands"], default="blacklist")
    cmds = sorted(c.name for c in getattr(context, "command_registry", None).all_commands()) if getattr(context, "command_registry", None) else []
    return FormStep("value", f"Command names for the list. Available: {', '.join(cmds) or '(none)'}", False, "array", default=[], prompt_when_missing=True)


def _frontends(context):
    """All frontends worth showing: discovered (installed) + enabled + profiled.

    Discovery reflects what is actually present on disk, so the kernel only
    lists ``repl`` until a frontend plugin is installed. Enabled/profiled names
    are unioned in so an entry stays editable even if its plugin was removed.
    """
    config = getattr(context, "config", {}) or {}
    names = set()
    manager = getattr(getattr(context, "runtime", None), "frontend_manager", None)
    names.update(getattr(manager, "available_frontends", ()) or ())
    names.update(getattr(manager, "adapters", {}) or {})
    names.update(config.get("enabled_frontends", []) or [])
    names.update(config.get("frontend_profiles", {}) or {})
    return sorted(names)


def _show(context):
    """Internal helper to handle show."""
    config = getattr(context, "config", {}) or {}
    enabled = set(config.get("enabled_frontends", []))
    profiles = config.get("frontend_profiles", {}) or {}
    lines = ["Frontends:"]
    for name in _frontends(context):
        status = "Enabled" if name in enabled else "Disabled"
        scope = _profile_summary(profiles.get(name))
        lines.append(f"  {name:<10} {status:<9} {scope}")
    return "\n".join(lines)


def _describe(config, name):
    """Internal helper to handle describe."""
    enabled = set((config or {}).get("enabled_frontends", []))
    profile = ((config or {}).get("frontend_profiles", {}) or {}).get(name)
    status = "Enabled" if name in enabled else "Disabled"
    return f"{name}\nStatus: {status}\nProfile: {_profile_summary(profile)}"


def _profile_summary(profile):
    """One-line description of a frontend profile (or the unrestricted default)."""
    if not profile:
        return "agent default, all commands"
    agent = profile.get("agent_profile") or "default"
    mode = profile.get("whitelist_or_blacklist_commands", "blacklist")
    listed = profile.get("commands_list") or []
    cmds = f"{mode} {', '.join(listed)}" if listed else ("whitelist (none → all blocked)" if mode == "whitelist" else "all commands")
    return f"agent {agent}, {cmds}"


def _render_value(field, value):
    """Internal helper to format a saved value for the confirmation message."""
    if field == "commands_list":
        return ", ".join(value) or "(none)"
    return str(value)
