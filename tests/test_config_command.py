"""Structured config metadata and sandboxed ``/config`` coverage."""

from types import SimpleNamespace

from plugins import plugin_discovery
from sandbox import Sandbox
from sandbox.handlers.kernel import _config_read


class DemoFrontend:
    name = "demo_frontend"
    config_settings = [
        (
            "Demo color",
            "demo_color_config_test",
            "Controls the demo color.",
            "blue",
            {"type": "text"},
        ),
        (
            "Hidden token",
            "demo_secret_config_test",
            "Never expose this.",
            "",
            {"hidden": True},
        ),
    ]


class SharedService:
    name = "shared_service"
    config_settings = [
        (
            "Demo color",
            "demo_color_config_test",
            "Controls the demo color.",
            "blue",
            {"type": "text"},
        ),
    ]


def _register():
    plugin_discovery._collect_config_settings(
        DemoFrontend(), plugin_type="frontend")
    plugin_discovery._collect_config_settings(
        SharedService(), service_names=["shared_service"],
        plugin_type="service")


def _remove():
    for key in ("demo_color_config_test", "demo_secret_config_test"):
        plugin_discovery._setting_to_services.pop(key, None)
        plugin_discovery._setting_to_plugins.pop(key, None)
        plugin_discovery._plugin_setting_types.pop(key, None)
        plugin_discovery._plugin_settings_keys.discard(key)
    plugin_discovery._plugin_settings[:] = [
        entry for entry in plugin_discovery._plugin_settings
        if entry[1] not in {
            "demo_color_config_test", "demo_secret_config_test"}]


def _context(monkeypatch):
    config = {
        "demo_color_config_test": "red",
        "stream_responses": True,
    }
    saved = {}
    rescans = []
    monkeypatch.setattr(
        "config.config_manager.save", lambda values: saved.update(values))
    monkeypatch.setattr(
        "config.config_manager.load_plugin_config", lambda: {})
    monkeypatch.setattr(
        "config.config_manager.save_plugin_config",
        lambda values: saved.update(values),
    )
    runtime = SimpleNamespace(
        config=config, refresh_session_specs=lambda: None)
    context = SimpleNamespace(
        config=dict(config),
        runtime=runtime,
        db=None,
        user_id=1,
        session_key="chat",
        orchestrator=SimpleNamespace(
            watcher=SimpleNamespace(rescan=lambda: rescans.append(True))),
        services={},
    )
    return context, saved, rescans


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_config.py",
            "ConfigCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_config_details_are_structured_redacted_and_hide_private_settings(
        monkeypatch):
    _register()
    context, _, _ = _context(monkeypatch)
    try:
        result = _config_read(context, {"details": True})
    finally:
        _remove()

    demo = next(
        item for item in result.data
        if item["key"] == "demo_color_config_test")
    assert demo["category"] == "plugin"
    assert demo["storage"] == "plugin_config.json"
    assert demo["owners"] == ["demo_frontend", "shared_service"]
    assert demo["restart_required"] is True
    assert not any(
        item["key"] == "demo_secret_config_test" for item in result.data)


def test_config_form_drills_through_category_plugin_and_typed_value(
        monkeypatch):
    _register()
    context, _, _ = _context(monkeypatch)
    try:
        initial = _run(context, {}, method="form")
        plugins = _run(
            context, {"category": "plugin"}, method="form")
        chosen = _run(
            context,
            {
                "category": "plugin",
                "plugin_name": "demo_frontend",
            },
            method="form",
        )
        edit = _run(
            context,
            {
                "setting_name": "demo_color_config_test",
                "action": "edit",
            },
            method="form",
        )
    finally:
        _remove()

    assert initial.data[0]["enum"] == [
        "kernel", "plugin", "user", "all"]
    assert "demo_frontend" in plugins.data[1]["enum"]
    assert "shared_service" in plugins.data[1]["enum"]
    assert chosen.data[-1]["enum"] == ["demo_color_config_test"]
    assert edit.data[-1]["type"] == "string"
    assert "Shared setting" in edit.data[-2]["prompt"]


def test_config_lists_categories_and_plugin_groups(monkeypatch):
    _register()
    context, _, _ = _context(monkeypatch)
    try:
        all_settings = _run(context, {})
        plugin_settings = _run(context, {"category": "plugin"})
        selected = _run(
            context,
            {"category": "plugin", "plugin_name": "demo_frontend"},
        )
    finally:
        _remove()

    assert "Kernel Settings (config.json):" in all_settings.data
    assert "Plugin Settings (plugin_config.json):" in all_settings.data
    assert "User Settings (per-user):" in all_settings.data
    assert "demo_frontend:" in plugin_settings.data
    assert "shared_service:" in plugin_settings.data
    assert selected.data.startswith("demo_frontend:")


def test_config_edit_preserves_output_and_restart_note(monkeypatch):
    _register()
    context, saved, _ = _context(monkeypatch)
    try:
        result = _run(
            context,
            {
                "setting_name": "demo_color_config_test",
                "action": "edit",
                "value": "green",
            },
            approve=lambda *_: True,
        )
    finally:
        _remove()

    assert result.data == (
        "Set demo_color_config_test = green. Restart required.")
    assert saved["demo_color_config_test"] == "green"


def test_config_write_rescans_watcher_settings(monkeypatch):
    context, _, rescans = _context(monkeypatch)

    result = _run(
        context,
        {
            "setting_name": "sync_directories",
            "action": "edit",
            "value": ["C:\\Notes"],
        },
        approve=lambda *_: True,
    )

    assert result.data == "Set sync_directories = C:\\Notes"
    assert rescans == [True]
