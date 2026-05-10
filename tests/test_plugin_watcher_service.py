from pathlib import Path

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED
from plugins.services.pluginWatcherService import PluginWatcherService


class _ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.unregistered = []

    def unregister(self, name):
        self.unregistered.append(name)


def _patch_plugin_dir(monkeypatch, directory):
    import plugins.helpers.plugin_paths as paths
    import plugins.services.pluginWatcherService as watcher_mod

    config = dict(paths.PLUGIN_CONFIG)
    built, sandbox, prefix, namespaces = config["tool"]
    config["tool"] = (Path(directory).resolve(), sandbox, prefix, namespaces)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)
    monkeypatch.setattr(watcher_mod, "iter_plugin_dirs", lambda: [("tool", Path(directory).resolve())])


def test_plugin_watcher_initial_scan_records_mtimes(monkeypatch):
    root_dir = Path(".codex_plugin_watcher")
    path = root_dir / "tool_demo.py"
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, root_dir)
        service = PluginWatcherService({})

        service._scan_existing()

        assert str(path.resolve()) in service._known_mtimes
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_plugin_watcher_add_or_edit_loads_plugin(monkeypatch):
    calls = []
    root_dir = Path(".codex_plugin_watcher")
    path = root_dir / "tool_demo.py"
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, root_dir)
        monkeypatch.setattr("plugins.services.pluginWatcherService.load_single_plugin", lambda *a, **k: calls.append((a, k)) or ("demo", None))
        monkeypatch.setattr("plugins.services.pluginWatcherService.PluginWatcherService._reconcile_plugin_config", lambda self: None)
        service = PluginWatcherService({})
        service.bind_runtime(tool_registry=_ToolRegistry())

        service.handle_create_or_modify(str(path))

        assert calls and calls[0][0][0] == "tool"
        assert calls[0][0][1] == path.resolve()
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_plugin_watcher_emits_registered_and_edit_messages(monkeypatch):
    messages = []
    root_dir = Path(".codex_plugin_watcher")
    path = root_dir / "tool_demo.py"
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda payload: messages.append(payload["message"]))
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, root_dir)
        monkeypatch.setattr("plugins.services.pluginWatcherService.load_single_plugin", lambda *a, **k: ("demo", None))
        monkeypatch.setattr("plugins.services.pluginWatcherService.PluginWatcherService._reconcile_plugin_config", lambda self: None)
        service = PluginWatcherService({})

        service.handle_create_or_modify(str(path))
        service._known_mtimes[str(path.resolve())] = path.stat().st_mtime - 1
        service.handle_create_or_modify(str(path))

        assert messages == ["Registered plugin: demo", "Registered plugin edit: demo"]
    finally:
        unsub()
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_plugin_watcher_emits_registration_failed_message(monkeypatch):
    messages = []
    root_dir = Path(".codex_plugin_watcher")
    path = root_dir / "tool_demo.py"
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda payload: messages.append(payload["message"]))
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, root_dir)
        monkeypatch.setattr("plugins.services.pluginWatcherService.load_single_plugin", lambda *a, **k: (None, "boom"))
        service = PluginWatcherService({})

        service.handle_create_or_modify(str(path))

        assert messages == ["Plugin registration failed: tool_demo.py"]
    finally:
        unsub()
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_plugin_watcher_unchanged_mtime_is_ignored(monkeypatch):
    calls = []
    root_dir = Path(".codex_plugin_watcher")
    path = root_dir / "tool_demo.py"
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, root_dir)
        monkeypatch.setattr("plugins.services.pluginWatcherService.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))
        service = PluginWatcherService({})
        service._known_mtimes[str(path.resolve())] = path.stat().st_mtime

        service.handle_create_or_modify(str(path))

        assert not calls
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_plugin_watcher_delete_unloads_by_source(monkeypatch):
    calls = []
    messages = []
    root_dir = Path(".codex_plugin_watcher")
    path = root_dir / "tool_demo.py"
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda payload: messages.append(payload["message"]))
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, root_dir)
        service = PluginWatcherService({})
        registry = _ToolRegistry()
        registry.tools["demo"] = type("DemoTool", (), {"_source_path": str(path.resolve())})()
        service.bind_runtime(tool_registry=registry)
        service._known_mtimes[str(path.resolve())] = path.stat().st_mtime
        path.unlink()
        monkeypatch.setattr("plugins.services.pluginWatcherService.unload_plugin", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr("plugins.services.pluginWatcherService.PluginWatcherService._reconcile_plugin_config", lambda self: None)

        service.handle_delete(str(path))

        assert calls and calls[0][0][0] == "tool"
        assert calls[0][1]["source_path"] == str(path.resolve())
        assert messages == ["Unregistered plugin: demo"]
    finally:
        unsub()
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_plugin_watcher_wrong_name_does_not_load(monkeypatch):
    calls = []
    root_dir = Path(".codex_plugin_watcher")
    path = root_dir / "demo.py"
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, root_dir)
        monkeypatch.setattr("plugins.services.pluginWatcherService.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))
        service = PluginWatcherService({})

        service.handle_create_or_modify(str(path))

        assert not calls
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_plugin_watcher_unload_cancels_pending_timers():
    service = PluginWatcherService({})
    handler = service._handler = _FakeHandler()

    service.unload()

    assert handler.cancelled


class _FakeHandler:
    cancelled = False

    def cancel_pending(self):
        self.cancelled = True
