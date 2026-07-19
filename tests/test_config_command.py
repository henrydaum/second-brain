"""Tests for the /config drill-down gate: settings browse by category (kernel /
plugin / user / all), and plugin settings drill down a second level by owning
plugin. One-shot ``/config <setting>`` keeps working because the category (and
plugin_name) steps are optional enums the parser can skip.
"""

from types import SimpleNamespace

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)

from plugins.commands import command_config as cc
from plugins.frontends.helpers.command_registry import parse_command_line

_PLUGIN_SETTINGS = [
    ("Brave Search API Key", "brave_search_api_key", "API key.", "", {"type": "text"}),
    ("Title Delay (minutes)", "title_delay_minutes", "Delay.", 10,
     {"type": "slider", "range": (0, 60, 60), "is_float": False}),
]

# Owning-plugin map for the plugin settings above; title_delay is intentionally
# shared by two plugins to exercise the double-listing + shared-note paths.
_OWNERS = {
    "brave_search_api_key": ["tool_web_search"],
    "title_delay_minutes": ["service_llm", "service_titler"],
}


def _patch_plugins(monkeypatch):
    monkeypatch.setattr(cc, "get_plugin_settings", lambda: _PLUGIN_SETTINGS)
    monkeypatch.setattr(cc, "get_plugin_setting_scope", lambda key: "global")
    monkeypatch.setattr(cc, "get_setting_plugin_names", lambda key: _OWNERS.get(key, []))


def _ctx():
    return SimpleNamespace(config={}, db=None, user_id=None)


def _form(args, monkeypatch):
    _patch_plugins(monkeypatch)
    return cc.ConfigCommand().form(args, _ctx())


def test_categories_partition_all_settings(monkeypatch):
    _patch_plugins(monkeypatch)

    assert cc._category_of("stream_responses") == "kernel"
    assert cc._category_of("brave_search_api_key") == "plugin"
    assert cc._category_of("skip_permissions") == "user"  # user-scoped core setting

    counts = cc._category_counts()
    assert counts["plugin"] == 2
    assert counts["all"] == len(cc._settings())
    # The three real categories partition every setting; "all" is the total.
    assert sum(counts[c] for c in cc._REAL_CATEGORIES) == len(cc._settings())


def test_plugin_groups_group_by_owner(monkeypatch):
    _patch_plugins(monkeypatch)
    groups = cc._plugin_groups()
    assert groups["tool_web_search"] == ["brave_search_api_key"]
    # Shared setting is listed under each owning plugin.
    assert "title_delay_minutes" in groups["service_llm"]
    assert "title_delay_minutes" in groups["service_titler"]


def test_form_gates_settings_by_category(monkeypatch):
    steps = _form({}, monkeypatch)
    assert steps[0].name == "category"
    assert steps[0].required is False
    # Optional but prompt_when_missing, so it's the interactive default gate
    # (the four buttons) while still skippable in one-shot parsing.
    assert steps[0].prompt_when_missing is True
    assert steps[0].enum == ["kernel", "plugin", "user", "all"]
    assert steps[1].name == "setting_name"
    assert set(steps[1].enum) == set(cc._settings())  # unfiltered until chosen

    steps = _form({"category": "user"}, monkeypatch)
    assert steps[1].name == "setting_name"
    assert "skip_permissions" in steps[1].enum
    assert "stream_responses" not in steps[1].enum

    steps = _form({"category": "all"}, monkeypatch)
    assert set(steps[-1].enum) == set(cc._settings())


def test_form_plugin_category_drills_into_plugin_level(monkeypatch):
    # Choosing plugin inserts an optional plugin_name enum before setting_name.
    steps = _form({"category": "plugin"}, monkeypatch)
    assert [s.name for s in steps][:2] == ["category", "plugin_name"]
    assert steps[1].required is False
    assert set(steps[1].enum) == {"tool_web_search", "service_llm", "service_titler"}

    # With a plugin chosen, setting_name is filtered to that plugin's settings.
    steps = _form({"category": "plugin", "plugin_name": "service_llm"}, monkeypatch)
    name_step = next(s for s in steps if s.name == "setting_name")
    assert name_step.enum == ["title_delay_minutes"]


def test_quicklink_args_skip_the_category_gate(monkeypatch):
    steps = _form({"setting_name": "stream_responses"}, monkeypatch)
    assert [s.name for s in steps][:2] == ["setting_name", "action"]


def test_one_shot_setting_name_still_parses(monkeypatch):
    _patch_plugins(monkeypatch)
    cmd = cc.ConfigCommand()

    args = parse_command_line("stream_responses", lambda a, c: cmd.form(a, _ctx()))

    assert args["setting_name"] == "stream_responses"
    assert args.get("category") is None


def test_one_shot_category_setting_skips_plugin_level(monkeypatch):
    # `/config plugin <setting>` (no plugin name) still resolves: plugin_name is
    # an optional enum the parser skips when the token isn't a known plugin.
    _patch_plugins(monkeypatch)
    cmd = cc.ConfigCommand()

    args = parse_command_line("plugin title_delay_minutes", lambda a, c: cmd.form(a, _ctx()))

    assert args["category"] == "plugin"
    assert args.get("plugin_name") is None
    assert args["setting_name"] == "title_delay_minutes"


def test_one_shot_plugin_drilldown_parses(monkeypatch):
    # The new canonical plugin path: `/config plugin <plugin> <setting> edit <val>`.
    _patch_plugins(monkeypatch)
    cmd = cc.ConfigCommand()

    args = parse_command_line("plugin service_llm title_delay_minutes edit 25",
                              lambda a, c: cmd.form(a, _ctx()))

    assert args["category"] == "plugin"
    assert args["plugin_name"] == "service_llm"
    assert args["setting_name"] == "title_delay_minutes"
    assert args["action"] == "edit"
    assert args["value"] == 25


def test_list_groups_by_category(monkeypatch):
    _patch_plugins(monkeypatch)
    context = SimpleNamespace(config={}, db=None, user_id=None)

    out = cc._list(context)
    assert "Kernel Settings (config.json):" in out
    assert "Plugin Settings (plugin_config.json):" in out
    assert "User Settings (per-user):" in out

    # Plugin category with no plugin chosen groups by owning plugin.
    out = cc._list(context, "plugin")
    assert "Kernel Settings" not in out
    assert "tool_web_search:" in out
    assert "service_llm:" in out
    assert "brave_search_api_key" in out

    # Drilled into one plugin.
    out = cc._list(context, "plugin", "tool_web_search")
    assert "brave_search_api_key" in out
    assert "title_delay_minutes" not in out


def test_describe_notes_shared_settings(monkeypatch):
    _patch_plugins(monkeypatch)
    context = SimpleNamespace(config={}, db=None, user_id=None)

    out = cc._describe(context, "title_delay_minutes")
    assert "Shared setting" in out
    assert "service_titler" in out

    out = cc._describe(context, "brave_search_api_key")
    assert "Shared setting" not in out
