"""Tests for the plugin hot-reload substrate (``service_plugin_watcher``).

The watcher is the install/uninstall mechanism the future plugin store builds
on: it scans the plugin dirs, debounces filesystem events, and loads/unloads
plugins by file presence. These tests fake the loader and assert the
scan/add/edit/delete/ignore paths and the user-facing chat notices.
"""

from pathlib import Path

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED
from plugins import plugin_discovery
from plugins.services.service_plugin_watcher import PluginWatcherService


class _ToolRegistry:
    """Tool registry."""
    def __init__(self):
        """Initialize the tool registry."""
        self.tools = {}
        self.unregistered = []

    def unregister(self, name):
        """Unregister tool registry."""
        self.unregistered.append(name)

    def register(self, tool):
        """Register tool registry."""
        self.tools[tool.name] = tool


def _watched_dir(tmp_path, monkeypatch, plugin_type="tool"):
    """Create a watched plugin dir under tmp_path and patch it in."""
    directory = tmp_path / "watched"
    directory.mkdir()
    _patch_plugin_dir(monkeypatch, directory, plugin_type)
    return directory


def _patch_plugin_dir(monkeypatch, directory, plugin_type="tool"):
    """Internal helper to handle patch plugin dir."""
    import plugins.helpers.plugin_paths as paths
    import plugins.services.service_plugin_watcher as watcher_mod

    config = dict(paths.PLUGIN_CONFIG)
    directory = Path(directory).resolve()
    family = directory.name
    root = paths.PluginRoot("test", directory.parent, "test_plugins")
    prefix = paths.PLUGIN_FAMILIES[plugin_type][1]
    config[plugin_type] = (paths.PluginDir(root, plugin_type, family, prefix),)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)
    monkeypatch.setattr(watcher_mod, "iter_plugin_dirs", lambda: [(plugin_type, Path(directory).resolve())])


def _patch_tool_discovery(monkeypatch, roots):
    """Patch tool discovery to use test roots."""
    import plugins.helpers.plugin_paths as paths

    plugin_roots = tuple(paths.PluginRoot(name, Path(root), module, built_in) for name, root, module, built_in in roots)
    config = dict(paths.PLUGIN_CONFIG)
    config["tool"] = tuple(paths.PluginDir(root, "tool", "tools", "tool_") for root in plugin_roots)
    monkeypatch.setattr(paths, "PLUGIN_ROOTS", plugin_roots)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)
    monkeypatch.setattr(plugin_discovery, "PLUGIN_ROOTS", plugin_roots)
    monkeypatch.setattr(plugin_discovery, "_TOOL_CONFIG", plugin_discovery._discovery_config("tool"))


class _CommandRegistry:
    """Command registry."""
    def __init__(self):
        """Initialize command registry."""
        self._commands = {}

    def register(self, command):
        """Register command."""
        self._commands[command.name] = command

    def unregister(self, name):
        """Unregister command."""
        self._commands.pop(name, None)

    def to_callable_specs(self):
        """Return command specs."""
        return dict(self._commands)


def test_plugin_watcher_initial_scan_records_mtimes(tmp_path, monkeypatch):
    """Verify plugin watcher initial scan records mtimes."""
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    service = PluginWatcherService({})

    service._scan_existing()

    assert str(path.resolve()) in service._known_mtimes


def test_plugin_watcher_add_or_edit_loads_plugin(tmp_path, monkeypatch):
    """Verify plugin watcher add or edit loads plugin."""
    calls = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    monkeypatch.setattr("plugins.services.service_plugin_watcher.load_single_plugin", lambda *a, **k: calls.append((a, k)) or ("demo", None))
    monkeypatch.setattr("plugins.services.service_plugin_watcher.PluginWatcherService._reconcile_plugin_config", lambda self: None)
    service = PluginWatcherService({})
    service.bind_runtime(tool_registry=_ToolRegistry())

    service.handle_create_or_modify(str(path))

    assert calls and calls[0][0][0] == "tool"
    assert calls[0][0][1] == path.resolve()


def test_plugin_watcher_emits_registered_and_edit_messages(tmp_path, monkeypatch):
    """Verify plugin watcher emits registered and edit messages."""
    messages = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda payload: messages.append(payload["message"]))
    try:
        monkeypatch.setattr("plugins.services.service_plugin_watcher.load_single_plugin", lambda *a, **k: ("demo", None))
        monkeypatch.setattr("plugins.services.service_plugin_watcher.PluginWatcherService._reconcile_plugin_config", lambda self: None)
        service = PluginWatcherService({})

        service.handle_create_or_modify(str(path))
        service._known_mtimes[str(path.resolve())] = path.stat().st_mtime - 1
        service.handle_create_or_modify(str(path))

        assert messages == ["✓ Registered plugin: demo", "✓ Registered plugin edit: demo"]
    finally:
        unsub()


def test_plugin_watcher_emits_registration_failed_message(tmp_path, monkeypatch):
    """Verify plugin watcher emits registration failed message."""
    messages = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda payload: messages.append(payload["message"]))
    try:
        monkeypatch.setattr("plugins.services.service_plugin_watcher.load_single_plugin", lambda *a, **k: (None, "boom"))
        service = PluginWatcherService({})

        service.handle_create_or_modify(str(path))

        assert messages == ["✕ Plugin registration failed: tool_demo.py\nboom"]
    finally:
        unsub()


def test_plugin_watcher_unchanged_mtime_is_ignored(tmp_path, monkeypatch):
    """Verify plugin watcher unchanged mtime is ignored."""
    calls = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    monkeypatch.setattr("plugins.services.service_plugin_watcher.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))
    service = PluginWatcherService({})
    service._known_mtimes[str(path.resolve())] = path.stat().st_mtime

    service.handle_create_or_modify(str(path))

    assert not calls


def test_plugin_watcher_delete_unloads_by_source(tmp_path, monkeypatch):
    """Verify plugin watcher delete unloads by source."""
    calls = []
    messages = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda payload: messages.append(payload["message"]))
    try:
        service = PluginWatcherService({})
        registry = _ToolRegistry()
        registry.tools["demo"] = type("DemoTool", (), {"_source_path": str(path.resolve())})()
        service.bind_runtime(tool_registry=registry)
        service._known_mtimes[str(path.resolve())] = path.stat().st_mtime
        path.unlink()
        monkeypatch.setattr("plugins.services.service_plugin_watcher.unload_plugin", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr("plugins.services.service_plugin_watcher.PluginWatcherService._reconcile_plugin_config", lambda self: None)

        service.handle_delete(str(path))

        assert calls and calls[0][0][0] == "tool"
        assert calls[0][1]["source_path"] == str(path.resolve())
        assert messages == ["Deregistered plugin: demo"]
    finally:
        unsub()


def test_plugin_watcher_wrong_name_does_not_load(tmp_path, monkeypatch):
    """Verify plugin watcher wrong name does not load."""
    calls = []
    path = _watched_dir(tmp_path, monkeypatch) / "demo.py"
    path.write_text("x", encoding="utf-8")
    monkeypatch.setattr("plugins.services.service_plugin_watcher.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))
    service = PluginWatcherService({})

    service.handle_create_or_modify(str(path))

    assert not calls


def test_plugin_watcher_accepts_llm_backend_provider(tmp_path, monkeypatch):
    """Verify service-family LLM backend files refresh profiles instead of failing."""
    from plugins.services.service_llm import LLMRouter
    messages = []
    path = _watched_dir(tmp_path, monkeypatch, "service") / "service_fake_llm.py"
    unsub = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda payload: messages.append(payload["message"]))
    try:
        path.write_text(
            "from plugins.services.service_llm import BaseLLM, LLMResponse\n\n"
            "class FakeBackend(BaseLLM):\n"
            "    is_llm_backend = True\n"
            "    def __init__(self, model_name, api_key=None, base_url=None): super().__init__(); self.model_name = model_name\n"
            "    def _load(self): self.loaded = True; return True\n"
            "    def unload(self): self.loaded = False\n"
            "    def invoke(self, messages, attachments=None, **kwargs): return LLMResponse(content='ok')\n"
            "    def stream(self, messages, attachments=None, **kwargs): return iter(())\n"
            "    def chat_with_tools(self, messages, tools=None, **kwargs): return LLMResponse(content='ok')\n",
            encoding="utf-8",
        )
        config = {"llm_profiles": {"model-x": {"llm_service_class": "FakeBackend"}}, "default_llm_profile": "model-x"}
        services = {}
        services["llm"] = LLMRouter(config, services)
        monkeypatch.setattr("plugins.services.service_plugin_watcher.PluginWatcherService._reconcile_plugin_config", lambda self: None)
        service = PluginWatcherService(config)
        service.services = services

        service.handle_create_or_modify(str(path))

        assert services["model-x"].loaded
        assert messages == ["✓ Registered plugin: LLM backends"]
        path.unlink()
        service.handle_delete(str(path))
        assert "model-x" not in services
    finally:
        unsub()


def test_plugin_watcher_refreshes_runtime_commands_on_command_load(tmp_path, monkeypatch):
    """Verify command hot-load updates the runtime command snapshot."""
    path = _watched_dir(tmp_path, monkeypatch, "command") / "command_agent.py"
    path.write_text("x", encoding="utf-8")
    registry = _CommandRegistry()
    runtime = type("Runtime", (), {"commands": {}, "refreshes": 0})()
    runtime.refresh_session_specs = lambda: setattr(runtime, "refreshes", runtime.refreshes + 1)

    def fake_load(plugin_type, _path, **kwargs):
        command = type("AgentCommand", (), {"name": "agent", "_source_path": str(path.resolve())})()
        kwargs["command_registry"].register(command)
        return "agent", None

    monkeypatch.setattr("plugins.services.service_plugin_watcher.load_single_plugin", fake_load)
    monkeypatch.setattr("plugins.services.service_plugin_watcher.PluginWatcherService._reconcile_plugin_config", lambda self: None)
    service = PluginWatcherService({})
    service.bind_runtime(command_registry=registry, runtime=runtime)

    service.handle_create_or_modify(str(path))

    assert "agent" in registry._commands
    assert "agent" in runtime.commands
    assert runtime.refreshes == 1


def test_plugin_watcher_refreshes_runtime_commands_on_command_delete(tmp_path, monkeypatch):
    """Verify command hot-unload updates the runtime command snapshot."""
    path = _watched_dir(tmp_path, monkeypatch, "command") / "command_agent.py"
    path.write_text("x", encoding="utf-8")
    command = type("AgentCommand", (), {"name": "agent", "_source_path": str(path.resolve())})()
    registry = _CommandRegistry()
    registry.register(command)
    runtime = type("Runtime", (), {"commands": {"agent": command}, "refreshes": 0})()
    runtime.refresh_session_specs = lambda: setattr(runtime, "refreshes", runtime.refreshes + 1)
    service = PluginWatcherService({})
    service.bind_runtime(command_registry=registry, runtime=runtime)
    service._known_mtimes[str(path.resolve())] = path.stat().st_mtime
    monkeypatch.setattr("plugins.services.service_plugin_watcher.unload_plugin", lambda *a, **k: registry.unregister("agent"))
    monkeypatch.setattr("plugins.services.service_plugin_watcher.PluginWatcherService._reconcile_plugin_config", lambda self: None)

    path.unlink()
    service.handle_delete(str(path))

    assert "agent" not in registry._commands
    assert "agent" not in runtime.commands
    assert runtime.refreshes == 1


def test_plugin_watcher_unload_cancels_pending_timers():
    """Verify plugin watcher unload cancels pending timers."""
    service = PluginWatcherService({})
    handler = service._handler = _FakeHandler()

    service.unload()

    assert handler.cancelled


def test_discovery_loads_sandbox_tree_relative_helpers(tmp_path, monkeypatch):
    """Verify sandbox_plugins tools can import family-local helpers relatively."""
    root = tmp_path / "sandbox_plugins"
    tools = root / "tools"
    helpers = tools / "helpers"
    helpers.mkdir(parents=True)
    (helpers / "answer.py").write_text('VALUE = "relative ok"\n', encoding="utf-8")
    (tools / "tool_relative.py").write_text(
        "from plugins.BaseTool import BaseTool, ToolResult\n"
        "from .helpers.answer import VALUE\n\n"
        "class RelativeTool(BaseTool):\n"
        "    name = 'relative_tool'\n"
        "    description = 'test'\n"
        "    parameters = {}\n"
        "    def run(self, context, **kwargs):\n"
        "        return ToolResult(llm_summary=VALUE)\n",
        encoding="utf-8",
    )
    _patch_tool_discovery(monkeypatch, (("sandbox", root, "sandbox_plugins", False),))
    registry = _ToolRegistry()

    plugin_discovery.discover_tools(tmp_path, registry, {}, reload=True)

    assert registry.tools["relative_tool"].run(None).llm_summary == "relative ok"


def test_discovery_precedence_prefers_sandbox_over_installed(tmp_path, monkeypatch):
    """Verify earlier roots win name collisions."""
    sandbox = tmp_path / "sandbox_plugins"
    installed = tmp_path / "installed_plugins"
    for root, label in ((sandbox, "sandbox"), (installed, "installed")):
        tools = root / "tools"
        tools.mkdir(parents=True)
        (tools / "tool_same.py").write_text(
            "from plugins.BaseTool import BaseTool\n\n"
            "class SameTool(BaseTool):\n"
            "    name = 'same_tool'\n"
            f"    description = '{label}'\n"
            "    parameters = {}\n",
            encoding="utf-8",
        )
    _patch_tool_discovery(
        monkeypatch,
        (("sandbox", sandbox, "sandbox_plugins", False), ("installed", installed, "installed_plugins", False)),
    )
    registry = _ToolRegistry()

    plugin_discovery.discover_tools(tmp_path, registry, {}, reload=True)

    assert registry.tools["same_tool"].description == "sandbox"


def test_load_single_tool_accepts_auto_register_false(tmp_path, monkeypatch):
    """Installing a tool that opts out of auto-registration is a no-op, not a
    failure: the file is on disk and something (e.g. plan mode) registers it on
    demand. Mirrors boot discovery, which silently skips such tools."""
    sandbox = tmp_path / "sandbox_plugins"
    tools = sandbox / "tools"
    tools.mkdir(parents=True)
    (tools / "tool_deferred.py").write_text(
        "from plugins.BaseTool import BaseTool, ToolResult\n\n"
        "class Deferred(BaseTool):\n"
        "    name = 'deferred'\n"
        "    description = 'test'\n"
        "    parameters = {}\n"
        "    auto_register = False\n"
        "    def run(self, context, **kwargs):\n"
        "        return ToolResult(data={})\n",
        encoding="utf-8",
    )
    _patch_tool_discovery(monkeypatch, (("sandbox", sandbox, "sandbox_plugins", False),))
    registry = _ToolRegistry()

    name, error = plugin_discovery._load_single_tool(tools / "tool_deferred.py", registry)

    assert error is None
    assert name == "deferred"
    assert registry.tools == {}  # opted out of the global registry


def test_load_single_tool_rejects_file_without_tool(tmp_path, monkeypatch):
    """A file with no BaseTool subclass at all is still a real failure."""
    sandbox = tmp_path / "sandbox_plugins"
    tools = sandbox / "tools"
    tools.mkdir(parents=True)
    (tools / "tool_empty.py").write_text("VALUE = 1\n", encoding="utf-8")
    _patch_tool_discovery(monkeypatch, (("sandbox", sandbox, "sandbox_plugins", False),))
    registry = _ToolRegistry()

    name, error = plugin_discovery._load_single_tool(tools / "tool_empty.py", registry)

    assert name is None
    assert "No BaseTool subclass found" in error


class _FakeHandler:
    """Fake handler."""
    cancelled = False

    def cancel_pending(self):
        """Cancel pending."""
        self.cancelled = True
