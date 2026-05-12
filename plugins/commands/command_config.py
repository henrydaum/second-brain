"""Slash command plugin for `/config`."""

import json

from config.config_data import SETTINGS_DATA
from config import config_manager
from plugins.BaseCommand import BaseCommand
from plugins.plugin_discovery import get_plugin_setting_type, get_plugin_settings
from state_machine.conversation import FormStep


def _hidden(info):
    """Internal helper to handle hidden."""
    return isinstance(info, dict) and info.get("hidden") is True


CORE = {name: (title, desc) for title, name, desc, _, info in SETTINGS_DATA if not _hidden(info)}
ACTIONS = ["edit"]


class ConfigCommand(BaseCommand):
    """Slash-command handler for `/config`."""
    name = "config"
    description = "Select a config setting, then edit it"
    category = "Config & System"

    def form(self, args, context):
        """Handle form."""
        steps = [FormStep("setting_name", "Select a setting to inspect or edit.", True, enum=sorted(_settings()), columns=2)]
        if args.get("setting_name"):
            steps.append(FormStep("action", f"What do you want to do with this setting?\n\n{_describe(context, args['setting_name'])}", True, enum=ACTIONS, enum_labels=["Edit setting"]))
        if args.get("action") == "edit":
            steps.append(FormStep("value", _value_prompt(args.get("setting_name")), True, _value_type(args.get("setting_name"))))
        return steps

    def run(self, args, context):
        """Execute `/config` for the active session."""
        config = context.config if context.config is not None else {}
        key = args.get("setting_name")
        if not key:
            return _list(context)
        if key not in _settings():
            return f"Unknown setting: {key}"
        if args.get("action") != "edit":
            return _describe(context, key)
        value = _parse(args.get("value"), key)
        old = config.get(key)
        config[key] = value
        config_manager.save(config)
        if key in _plugin_keys():
            saved = config_manager.load_plugin_config()
            saved[key] = value
            config_manager.save_plugin_config(saved)
        runtime = getattr(context, "runtime", None)
        if runtime and value != old and hasattr(runtime, "refresh_session_specs"):
            runtime.refresh_session_specs()
        if value != old and get_plugin_setting_type(key) == "frontend":
            return f"Set {key} = {value}. Restart required."
        return f"Set {key} = {value}"


def _settings():
    """Internal helper to handle settings."""
    return CORE | {name: (title, desc) for title, name, desc, _, info in get_plugin_settings() if not _hidden(info)}


def _plugin_keys():
    """Internal helper to handle plugin keys."""
    return {entry[1] for entry in get_plugin_settings()}


def _setting_data(key):
    """Internal helper to handle setting data."""
    return next((entry for entry in [*SETTINGS_DATA, *get_plugin_settings()] if entry[1] == key), None)


def _value_type(key):
    """Internal helper to handle value type."""
    entry = _setting_data(key) or (None, None, None, None, {})
    default, info = entry[3], entry[4] if isinstance(entry[4], dict) else {}
    type_ = info.get("type")
    if type_ == "json_list":
        return "array"
    if type_ == "json_dict":
        return "object"
    if type_ in {"bool", "boolean"}:
        return "boolean"
    if type_ == "slider":
        return "number" if info.get("is_float") else "integer"
    return "array" if isinstance(default, list) else "object" if isinstance(default, dict) else "string"


def _value_prompt(key):
    """Internal helper to handle value prompt."""
    return "Enter a list of items, one on each line, like so:\n\nitem 1\nitem 2" if _value_type(key) == "array" else "Enter the new value."


def _describe(context, key):
    """Internal helper to handle describe."""
    title, desc = _settings().get(key, (key, ""))
    return f"{title}\n{key} = {(context.config or {}).get(key)}\n{desc}"


def _list(context):
    """Internal helper to list config."""
    return "Settings:\n" + "\n".join(f"  {k} = {(context.config or {}).get(k)}" for k in sorted(_settings()))


def _parse(value, key=None):
    """Internal helper to parse config."""
    if key:
        return FormStep("value", type=_value_type(key)).coerce(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
