"""Structured frontend metadata and sandboxed ``/frontends`` coverage."""

from types import SimpleNamespace

from plugins import plugin_discovery
from sandbox import Sandbox
from sandbox.handlers.kernel import _plugin_list


class ReplAdapter:
    config_settings = [
        (
            "Prompt color",
            "repl_prompt_color_frontends_test",
            "Color used for prompts.",
            "blue",
            {"type": "text"},
        ),
        (
            "Hidden token",
            "repl_token_frontends_test",
            "Never expose this.",
            "",
            {"hidden": True},
        ),
    ]


def _context(tmp_path, monkeypatch, *, config=None):
    config = config or {
        "enabled_frontends": ["repl"],
        "frontend_profiles": {},
        "agent_profiles": {"writer": {}},
    }
    saved = {}
    monkeypatch.setattr(
        "config.config_manager.save", lambda values: saved.update(values))
    monkeypatch.setattr(
        "config.config_manager.load_plugin_config", lambda: {})
    monkeypatch.setattr(
        "config.config_manager.save_plugin_config",
        lambda values: saved.update(values),
    )
    manager = SimpleNamespace(
        available_frontends={"repl", "telegram"},
        adapters={"repl": ReplAdapter()},
    )
    runtime = SimpleNamespace(
        config=config,
        frontend_manager=manager,
        refresh_session_specs=lambda: None,
    )
    command_registry = SimpleNamespace(
        commands={"clear": object(), "tools": object()},
        list=lambda: ["clear", "tools"],
    )
    context = SimpleNamespace(
        config=dict(config),
        runtime=runtime,
        command_registry=command_registry,
        db=None,
        services={},
        session_key="chat",
        user_id=1,
    )
    return context, saved


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_frontends.py",
            "FrontendsCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_frontend_details_union_native_sandbox_and_config_names(
        tmp_path, monkeypatch):
    context, _ = _context(
        tmp_path,
        monkeypatch,
        config={
            "enabled_frontends": ["removed"],
            "frontend_profiles": {"profiled": {}},
        },
    )

    result = _plugin_list(
        context, {"category": "frontends", "details": True})

    assert [item["name"] for item in result.data] == [
        "profiled", "removed", "repl", "telegram"]
    repl = next(item for item in result.data if item["name"] == "repl")
    telegram = next(
        item for item in result.data if item["name"] == "telegram")
    assert repl["available"] and repl["loaded"]
    assert [item["key"] for item in repl["config_settings"]] == [
        "repl_prompt_color_frontends_test"]
    assert telegram["available"] and not telegram["loaded"]


def test_frontends_form_uses_runtime_cache_and_shared_setting_steps(
        tmp_path, monkeypatch):
    context, _ = _context(tmp_path, monkeypatch)

    initial = _run(context, {}, method="form")
    selected = _run(
        context, {"frontend_name": "repl"}, method="form")
    quicklink = _run(
        context,
        {
            "frontend_name": "repl",
            "action": "edit_setting:repl_prompt_color_frontends_test",
        },
        method="form",
    )
    commands = _run(
        context,
        {
            "frontend_name": "repl",
            "action": "configure",
            "field": "commands_list",
        },
        method="form",
    )

    assert initial.data[0]["enum"] == ["repl", "telegram"]
    assert selected.data[1]["enum"] == [
        "configure", "enable", "disable",
        "edit_setting:repl_prompt_color_frontends_test",
    ]
    assert quicklink.data[-1]["prompt"] == "Enter the new value."
    assert commands.data[-1]["type"] == "array"
    assert "clear, tools" in commands.data[-1]["prompt"]


def test_frontends_list_output_matches_native_wire_format(
        tmp_path, monkeypatch):
    context, _ = _context(tmp_path, monkeypatch)

    result = _run(context, {})

    assert result.data == (
        "Frontends:\n\n"
        "| Frontend | Status | Access |\n"
        "| --- | --- | --- |\n"
        "| repl | Enabled | agent default, all commands |\n"
        "| telegram | Disabled | agent default, all commands |"
    )


def test_frontend_enable_disable_and_last_guard(tmp_path, monkeypatch):
    context, _ = _context(tmp_path, monkeypatch)

    enabled = _run(
        context,
        {"frontend_name": "telegram", "action": "enable"},
        approve=lambda *_: True,
    )
    context.config["enabled_frontends"] = ["repl", "telegram"]
    disabled = _run(
        context,
        {"frontend_name": "telegram", "action": "disable"},
        approve=lambda *_: True,
    )
    context.config["enabled_frontends"] = ["repl"]
    guarded = _run(
        context,
        {"frontend_name": "repl", "action": "disable"},
        approve=lambda *_: True,
    )

    assert enabled.data == "Enabled frontend: telegram. Restart required."
    assert disabled.data == "Disabled frontend: telegram. Restart required."
    assert guarded.data == "Cannot disable the last enabled frontend."


def test_frontend_profile_merge_and_empty_whitelist_warning(
        tmp_path, monkeypatch):
    config = {
        "enabled_frontends": ["repl"],
        "frontend_profiles": {
            "telegram": {
                "agent_profile": "writer",
                "whitelist_or_blacklist_commands": "blacklist",
                "commands_list": [],
            },
        },
        "agent_profiles": {"writer": {}},
    }
    context, _ = _context(
        tmp_path, monkeypatch, config=config)

    result = _run(
        context,
        {
            "frontend_name": "repl",
            "action": "configure",
            "field": "whitelist_or_blacklist_commands",
            "value": "whitelist",
        },
        approve=lambda *_: True,
    )

    assert result.data == (
        "Updated repl profile: Command mode → whitelist\n"
        "Note: whitelist is empty — every command is now blocked on this "
        "frontend."
    )
    profiles = context.runtime.config["frontend_profiles"]
    assert profiles["telegram"]["agent_profile"] == "writer"
    assert profiles["repl"]["whitelist_or_blacklist_commands"] == "whitelist"


def test_frontend_quicklink_uses_config_write_and_restart_note(
        tmp_path, monkeypatch):
    context, _ = _context(tmp_path, monkeypatch)
    plugin_discovery._collect_config_settings(
        ReplAdapter(), plugin_type="frontend")
    try:
        result = _run(
            context,
            {
                "frontend_name": "repl",
                "action": (
                    "edit_setting:repl_prompt_color_frontends_test"
                ),
                "value": "green",
            },
            approve=lambda *_: True,
        )
    finally:
        _remove_setting()

    assert result.data == (
        "Set repl_prompt_color_frontends_test = green. Restart required.")
    assert context.runtime.config[
        "repl_prompt_color_frontends_test"] == "green"


def _remove_setting():
    key = "repl_prompt_color_frontends_test"
    plugin_discovery._plugin_settings[:] = [
        entry for entry in plugin_discovery._plugin_settings
        if entry[1] != key
    ]
    plugin_discovery._plugin_settings_keys.discard(key)
    plugin_discovery._plugin_setting_types.pop(key, None)
    plugin_discovery._setting_to_plugins.pop(key, None)
