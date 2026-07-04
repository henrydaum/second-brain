"""Tests for the /config category gate: settings browse by category (kernel /
plugin / user), while one-shot ``/config <setting>`` keeps working because the
category step is an optional enum the parser can skip.
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


def _patch_plugins(monkeypatch):
    monkeypatch.setattr(cc, "get_plugin_settings", lambda: _PLUGIN_SETTINGS)
    monkeypatch.setattr(cc, "get_plugin_setting_scope", lambda key: "global")


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
    assert sum(counts.values()) == len(cc._settings())


def test_form_gates_settings_by_category(monkeypatch):
    steps = _form({}, monkeypatch)
    assert steps[0].name == "category"
    assert steps[0].required is False
    assert steps[0].enum == ["kernel", "plugin", "user"]
    assert steps[1].name == "setting_name"
    assert set(steps[1].enum) == set(cc._settings())  # unfiltered until chosen

    steps = _form({"category": "plugin"}, monkeypatch)
    assert steps[1].enum == sorted(["brave_search_api_key", "title_delay_minutes"])

    steps = _form({"category": "user"}, monkeypatch)
    assert "skip_permissions" in steps[1].enum
    assert "stream_responses" not in steps[1].enum


def test_quicklink_args_skip_the_category_gate(monkeypatch):
    steps = _form({"setting_name": "stream_responses"}, monkeypatch)
    assert [s.name for s in steps][:2] == ["setting_name", "action"]


def test_one_shot_setting_name_still_parses(monkeypatch):
    _patch_plugins(monkeypatch)
    cmd = cc.ConfigCommand()

    args = parse_command_line("stream_responses", lambda a, c: cmd.form(a, _ctx()))

    assert args["setting_name"] == "stream_responses"
    assert args.get("category") is None


def test_one_shot_category_filters(monkeypatch):
    _patch_plugins(monkeypatch)
    cmd = cc.ConfigCommand()

    args = parse_command_line("plugin title_delay_minutes", lambda a, c: cmd.form(a, _ctx()))

    assert args["category"] == "plugin"
    assert args["setting_name"] == "title_delay_minutes"


def test_list_groups_by_category(monkeypatch):
    _patch_plugins(monkeypatch)
    context = SimpleNamespace(config={}, db=None, user_id=None)

    out = cc._list(context)
    assert "Kernel settings (config.json):" in out
    assert "Plugin settings (plugin_config.json):" in out
    assert "User settings (per-user):" in out

    out = cc._list(context, "plugin")
    assert "Kernel settings" not in out
    assert "brave_search_api_key" in out
