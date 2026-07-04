"""Slash command plugin for `/config`."""

import json

from config.config_data import SETTINGS_DATA
from config import config_manager
from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import detail_card, md_table, quote_block
from plugins.plugin_discovery import get_plugin_setting_scope, get_plugin_setting_type, get_plugin_settings, get_setting_plugin_names
from state_machine.conversation import FormStep


def _hidden(info):
    """Internal helper to handle hidden."""
    return isinstance(info, dict) and info.get("hidden") is True


CORE = {name: (title, desc) for title, name, desc, _, info in SETTINGS_DATA if not _hidden(info)}
ACTIONS = ["edit"]

# Browse gate: settings are shown one category at a time — there are too many
# to pick from a single flat list. Values are stable tokens (usable one-shot:
# `/config plugin`); labels explain where each category is stored.
CATEGORIES = ["kernel", "plugin", "user"]
_CATEGORY_LABELS = {
    "kernel": "Kernel settings (config.json)",
    "plugin": "Plugin settings (plugin_config.json)",
    "user": "User settings (per-user)",
}

# Global settings the filesystem watcher reads. Changing any of these triggers a
# live watcher rescan so the sync starts immediately rather than after a restart.
_WATCHER_KEYS = {"sync_directories", "ignored_extensions", "ignored_folders", "skip_hidden_folders"}


def _rescan_watcher(context):
    """Trigger a live watcher rescan if the watcher is reachable from context."""
    orchestrator = getattr(context, "orchestrator", None)
    watcher = getattr(orchestrator, "watcher", None)
    if watcher is not None and hasattr(watcher, "rescan"):
        watcher.rescan()


class ConfigCommand(BaseCommand):
    """Slash-command handler for `/config`."""
    name = "config"
    description = "Select a config setting, then edit it"
    category = "Config & System"

    def form(self, args, context):
        """Handle form."""
        steps = []
        if not args.get("setting_name"):
            # Optional enum: interactively it gates the setting list by
            # category; one-shot, a non-category first token falls through to
            # setting_name (see parse_command_line's optional-enum skip), so
            # `/config stream_responses` still works.
            counts = _category_counts()
            steps.append(FormStep(
                "category", "Which settings do you want to browse?", False,
                enum=CATEGORIES,
                enum_labels=[f"{_CATEGORY_LABELS[c]} — {counts[c]}" for c in CATEGORIES]))
        steps.append(FormStep("setting_name", "Select a setting to inspect or edit.", True,
                              enum=sorted(_settings_for(args.get("category"))), columns=2))
        if args.get("setting_name"):
            steps.append(FormStep("action", f"What do you want to do with this setting?\n\n{_describe(context, args['setting_name'])}", True, enum=ACTIONS, enum_labels=["Edit setting"]))
        if args.get("action") == "edit":
            steps.append(value_step_for(args.get("setting_name"), context))
        return steps

    def run(self, args, context):
        """Execute `/config` for the active session."""
        key = args.get("setting_name")
        if not key:
            return _list(context, args.get("category"))
        if key not in _settings():
            return f"Unknown setting: {key}"
        if args.get("action") != "edit":
            return _describe(context, key)
        return apply_edit(context, key, args.get("value"))


def value_step_for(key, _context=None) -> FormStep:
    """The value-entry form step for one setting (shared with quicklinks)."""
    return FormStep("value", _value_prompt(key), True, _value_type(key))


def apply_edit(context, key, raw_value) -> str:
    """Write one setting through the scope-aware persistence path.

    Shared by /config and the Edit-setting quicklinks on /tools, /tasks,
    /services, and /frontends.
    """
    config = context.config if context.config is not None else {}
    if key not in _settings():
        return f"Unknown setting: {key}"
    value = _parse(raw_value, key)

    # User-scoped settings write only to the current user's config blob —
    # never to global config.json / plugin_config.json.
    if _scope(key) == "user":
        db = getattr(context, "db", None)
        if db is None:
            return "User settings are not available in this context."
        uid = getattr(context, "user_id", None)
        user_cfg = db.get_user_config(uid)
        user_cfg[key] = value
        db.set_user_config(uid, user_cfg)
        context.config[key] = value
        runtime = getattr(context, "runtime", None)
        if key == "active_agent_profile" and runtime and hasattr(runtime, "refresh_session_specs"):
            runtime.refresh_session_specs()
        return f"Set {key} = {_format_value(value)}"

    old = config.get(key)
    config[key] = value
    config_manager.save(config)
    if key in _plugin_keys():
        saved = config_manager.load_plugin_config()
        saved[key] = value
        config_manager.save_plugin_config(saved)
    runtime = getattr(context, "runtime", None)
    # context.config is a per-call copy; write through to the canonical
    # runtime config so the next /config (a fresh copy) shows the new value.
    if runtime is not None and getattr(runtime, "config", None) is not None:
        runtime.config[key] = value
    if runtime and value != old and hasattr(runtime, "refresh_session_specs"):
        runtime.refresh_session_specs()
    # Watch-affecting keys take effect live: re-read directories and run a
    # fresh scan so the database starts syncing without a restart.
    if value != old and key in _WATCHER_KEYS:
        _rescan_watcher(context)
    if value != old and get_plugin_setting_type(key) == "frontend":
        return f"Set {key} = {_format_value(value)}. Restart required."
    return f"Set {key} = {_format_value(value)}"


def _settings():
    """Internal helper to handle settings."""
    return CORE | {name: (title, desc) for title, name, desc, _, info in get_plugin_settings() if not _hidden(info)}


def _category_of(key) -> str:
    """Which /config browse category a setting belongs to."""
    if _scope(key) == "user":
        return "user"
    return "plugin" if key in _plugin_keys() else "kernel"


def _settings_for(category=None):
    """The settings dict, filtered to one browse category when given."""
    all_settings = _settings()
    if category not in CATEGORIES:
        return all_settings
    return {k: v for k, v in all_settings.items() if _category_of(k) == category}


def _category_counts() -> dict:
    """Setting count per browse category, for the gate labels."""
    counts = dict.fromkeys(CATEGORIES, 0)
    for key in _settings():
        counts[_category_of(key)] += 1
    return counts


def _scope(key) -> str:
    """"user" (stored in the current user's config) or "global". Core settings are
    global unless their type_info opts into user scope; plugins use the same flag."""
    entry = _setting_data(key)
    info = entry[4] if entry and isinstance(entry[4], dict) else {}
    return "user" if info.get("scope") == "user" else (get_plugin_setting_scope(key) if key in _plugin_keys() else "global")


def _default_for(key):
    """Declared default value for a setting key."""
    entry = _setting_data(key)
    return entry[3] if entry else None


def _current_value(context, key):
    """The value to display/edit: per-user for user-scoped keys (defaulting to the
    declared default), else the global config value."""
    if _scope(key) == "user":
        db = getattr(context, "db", None)
        if db is None:
            return _default_for(key)
        uid = getattr(context, "user_id", None)
        return db.get_user_config(uid).get(key, (context.config or {}).get(key, _default_for(key)))
    return (context.config or {}).get(key)


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
    if type_ in {"path", "path_list"}:
        return type_
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
    vtype = _value_type(key)
    if vtype == "path_list":
        return ("Enter one folder path per line. / and \\ are both accepted; each folder must already exist. Example:\n\nC:\\Users\\you\\Notes\nD:\\Archive")
    if vtype == "path":
        return "Enter a path. / and \\ are both accepted; the parent folder must exist."
    if vtype == "array":
        return "Enter a list of items, one on each line, like so:\n\nitem 1\nitem 2"
    return "Enter the new value."


def _format_value(value):
    """Render a setting value for display without Python repr artifacts —
    list brackets/quotes and the doubled backslashes that str(list) produces
    on Windows paths. Each list item is shown via str(), so a stored
    'C:\\Users\\me' displays with single separators."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "(none)" if not value else ", ".join(str(item) for item in value)
    return str(value)


def _describe(context, key):
    """Internal helper to handle describe."""
    title, desc = _settings().get(key, (key, ""))
    tag = " (per-user)" if _scope(key) == "user" else ""
    users = get_setting_plugin_names(key)
    card = detail_card(f"{title}{tag}", [
        (key, _format_value(_current_value(context, key))),
        ("Used by", ", ".join(users) if users else "kernel"),
    ])
    return card + (f"\n\n{quote_block(desc)}" if desc.strip() else "")


def _list(context, category=None):
    """Internal helper to list config, grouped by browse category."""
    sections = []
    wanted = [category] if category in CATEGORIES else CATEGORIES
    for cat in wanted:
        keys = sorted(_settings_for(cat))
        if not keys:
            continue
        rows = [(k, _format_value(_current_value(context, k))) for k in keys]
        sections.append(f"{_CATEGORY_LABELS[cat]}:\n\n" + md_table(["Setting", "Value"], rows))
    return "\n\n".join(sections) or "No settings found."


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
