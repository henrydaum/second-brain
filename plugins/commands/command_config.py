import json

from config.config_data import SETTINGS_DATA
from config import config_manager
from plugins.BaseCommand import BaseCommand
from plugins.plugin_discovery import get_plugin_settings
from state_machine.conversationClass import FormStep


def _hidden(info):
    return isinstance(info, dict) and info.get("hidden") is True


CORE = {name: (title, desc) for title, name, desc, _, info in SETTINGS_DATA if not _hidden(info)}
ACTIONS = ["edit"]


class ConfigCommand(BaseCommand):
    name = "config"
    description = "Select a config setting, then edit it"
    category = "Config & System"

    def form(self, args, context):
        steps = [FormStep("setting_name", "Setting", True, enum=sorted(_settings()), columns=2)]
        if args.get("setting_name"):
            steps.append(FormStep("action", _describe(context, args["setting_name"]), True, enum=ACTIONS))
        if args.get("action") == "edit":
            steps.append(FormStep("value", "New value", True))
        return steps

    def run(self, args, context):
        config = context.config if context.config is not None else {}
        key = args.get("setting_name")
        if not key:
            return _list(context)
        if key not in _settings():
            return f"Unknown setting: {key}"
        if args.get("action") != "edit":
            return _describe(context, key)
        value = _parse(args.get("value"))
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
        return f"Set {key} = {value}"


def _settings():
    return CORE | {name: (title, desc) for title, name, desc, _, info in get_plugin_settings() if not _hidden(info)}


def _plugin_keys():
    return {entry[1] for entry in get_plugin_settings()}


def _describe(context, key):
    title, desc = _settings().get(key, (key, ""))
    return f"{title}\n{key} = {(context.config or {}).get(key)}\n{desc}"


def _list(context):
    return "Settings:\n" + "\n".join(f"  {k} = {(context.config or {}).get(k)}" for k in sorted(_settings()))


def _parse(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
