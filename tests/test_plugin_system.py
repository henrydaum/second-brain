"""Tests for the kernel plugin hot-reload substrate.

The watcher is the install/uninstall mechanism the future plugin store builds
on: it scans the plugin dirs, debounces filesystem events, and loads/unloads
plugins by file presence. These tests fake the loader and assert the
scan/add/edit/delete/ignore paths and the user-facing chat notices.
"""

from pathlib import Path

from events.event_bus import bus
from events.event_channels import NOTIFICATION_PUSHED
from plugins import plugin_discovery, plugin_watcher
from plugins.plugin_watcher import PluginWatcher


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


def test_plugin_watcher_is_kernel_infrastructure():
    """The watcher must not drift back into the service registry."""
    from plugins.native.service import BaseService
    import plugins.plugin_watcher as watcher_module

    assert not issubclass(PluginWatcher, BaseService)
    assert not hasattr(watcher_module, "build_services")


def _watched_dir(tmp_path, monkeypatch, plugin_type="tool"):
    """Create a watched plugin dir under tmp_path and patch it in."""
    directory = tmp_path / "watched"
    directory.mkdir()
    _patch_plugin_dir(monkeypatch, directory, plugin_type)
    return directory


def _patch_plugin_dir(monkeypatch, directory, plugin_type="tool"):
    """Internal helper to handle patch plugin dir."""
    import trees
    import plugins.plugin_paths as paths

    config = dict(paths.PLUGIN_CONFIG)
    directory = Path(directory).resolve()
    family = directory.name
    root = paths.PluginRoot("test", directory.parent, "test_plugins")
    prefix = paths.PLUGIN_FAMILIES[plugin_type][1]
    config[plugin_type] = (paths.PluginDir(root, plugin_type, family, prefix),)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)
    # The watcher walks the layout rather than a family list, so narrowing it
    # to one directory is done at the table.
    tree_root = trees.roots_by_name[paths.PLUGIN_FAMILIES[plugin_type][0]]
    monkeypatch.setattr(
        trees, "iter_root_dirs",
        lambda watched_only=False: [(root, tree_root, directory)])


def _patch_tool_discovery(monkeypatch, roots):
    """Patch tool discovery to use test roots."""
    import plugins.plugin_paths as paths

    plugin_roots = tuple(paths.PluginRoot(name, Path(root), module, builtin=built_in)
                         for name, root, module, built_in in roots)
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
    service = PluginWatcher({})

    service._scan_existing()

    assert str(path.resolve()) in service._known_mtimes


def test_plugin_watcher_add_or_edit_loads_plugin(tmp_path, monkeypatch):
    """Verify plugin watcher add or edit loads plugin."""
    calls = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    monkeypatch.setattr("plugins.plugin_watcher.load_single_plugin", lambda *a, **k: calls.append((a, k)) or ("demo", None))
    monkeypatch.setattr("plugins.plugin_watcher.PluginWatcher._reconcile_plugin_config", lambda self: None)
    service = PluginWatcher({})
    service.bind_runtime(tool_registry=_ToolRegistry())

    outcome = service.register(str(path))

    assert calls and calls[0][0][0] == "tool"
    assert calls[0][0][1] == path.resolve()
    assert outcome == {
        "ok": True,
        "name": "demo",
        "family": "tool",
        "path": str(path.resolve()),
    }


def test_kernel_coordinator_live_registers_and_unregisters_sandbox_tool(
    tmp_path,
    monkeypatch,
):
    """The real discovery bridge is shared by watcher and SDK mutations."""
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text(
        '"""A live registration fixture."""\n\n'
        "from guest.bases import BaseTool\n\n"
        "class DemoTool(BaseTool):\n"
        '    """A tiny sandboxed tool."""\n'
        '    name = "demo"\n'
        '    description = "Demo."\n'
        "    parameters = {}\n\n"
        "    def run(self, sdk):\n"
        '        """Return a stable answer."""\n'
        '        return "ok"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "plugins.plugin_watcher.PluginWatcher._reconcile_plugin_config",
        lambda self: None,
    )
    registry = _ToolRegistry()
    watcher = PluginWatcher({}, tool_registry=registry)

    loaded = watcher.register(path)
    unloaded = watcher.unregister(path)

    assert loaded["ok"] and loaded["name"] == "demo"
    assert unloaded["ok"] and unloaded["names"] == ["demo"]
    assert registry.unregistered == ["demo"]


def test_plugin_watcher_emits_registered_and_edit_messages(tmp_path, monkeypatch):
    """Verify plugin watcher emits registered and edit messages."""
    messages = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    unsub = bus.subscribe(NOTIFICATION_PUSHED, messages.append)
    try:
        monkeypatch.setattr("plugins.plugin_watcher.load_single_plugin", lambda *a, **k: ("demo", None))
        monkeypatch.setattr("plugins.plugin_watcher.PluginWatcher._reconcile_plugin_config", lambda self: None)
        service = PluginWatcher({})

        service.handle_create_or_modify(str(path))
        service._known_mtimes[str(path.resolve())] = path.stat().st_mtime - 1
        service.handle_create_or_modify(str(path))

        # A registry change is the system announcing itself, not the
        # conversation, so it travels as a notification — and the source is
        # stamped rather than described in the text.
        assert [(m["title"], m["body"]) for m in messages] == [
            ("Plugin registered", "demo"), ("Plugin reloaded", "demo")]
        assert {m["source"] for m in messages} == {"plugin_watcher"}
        assert {m["level"] for m in messages} == {"success"}
    finally:
        unsub()


def test_plugin_watcher_emits_registration_failed_message(tmp_path, monkeypatch):
    """Verify plugin watcher emits registration failed message."""
    messages = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    unsub = bus.subscribe(NOTIFICATION_PUSHED, messages.append)
    try:
        monkeypatch.setattr("plugins.plugin_watcher.load_single_plugin", lambda *a, **k: (None, "boom"))
        service = PluginWatcher({})

        service.handle_create_or_modify(str(path))

        # The ✕ is gone from the text because ``level`` carries it now: a
        # frontend styles a failure rather than reading a glyph out of a string.
        assert messages[0]["title"] == "Plugin registration failed"
        assert messages[0]["body"] == "tool_demo.py\n\nboom"
        assert messages[0]["level"] == "error"
    finally:
        unsub()


def test_plugin_watcher_unchanged_mtime_is_ignored(tmp_path, monkeypatch):
    """Verify plugin watcher unchanged mtime is ignored."""
    calls = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    monkeypatch.setattr("plugins.plugin_watcher.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))
    service = PluginWatcher({})
    service._known_mtimes[str(path.resolve())] = path.stat().st_mtime

    service.handle_create_or_modify(str(path))

    assert not calls


def test_plugin_watcher_delete_unloads_by_source(tmp_path, monkeypatch):
    """Verify plugin watcher delete unloads by source."""
    calls = []
    messages = []
    path = _watched_dir(tmp_path, monkeypatch) / "tool_demo.py"
    path.write_text("x", encoding="utf-8")
    unsub = bus.subscribe(NOTIFICATION_PUSHED, messages.append)
    try:
        service = PluginWatcher({})
        registry = _ToolRegistry()
        registry.tools["demo"] = type("DemoTool", (), {"_source_path": str(path.resolve())})()
        service.bind_runtime(tool_registry=registry)
        service._known_mtimes[str(path.resolve())] = path.stat().st_mtime
        path.unlink()
        monkeypatch.setattr("plugins.plugin_watcher.unload_plugin", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr("plugins.plugin_watcher.PluginWatcher._reconcile_plugin_config", lambda self: None)

        service.handle_delete(str(path))

        assert calls and calls[0][0][0] == "tool"
        assert calls[0][1]["source_path"] == str(path.resolve())
        assert [(m["title"], m["body"]) for m in messages] == [
            ("Plugin removed", "demo")]
    finally:
        unsub()


def test_plugin_watcher_wrong_name_does_not_load(tmp_path, monkeypatch):
    """Verify plugin watcher wrong name does not load."""
    calls = []
    path = _watched_dir(tmp_path, monkeypatch) / "demo.py"
    path.write_text("x", encoding="utf-8")
    monkeypatch.setattr("plugins.plugin_watcher.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))
    service = PluginWatcher({})

    service.handle_create_or_modify(str(path))

    assert not calls


def test_plugin_watcher_refreshes_sandboxed_llm_backends(
    tmp_path,
    monkeypatch,
):
    """A change under llm/ rescans the kernel registry."""

    import trees
    from tests.support import retarget_trees

    directory = retarget_trees(monkeypatch, tmp_path)["workspace"] / "llm"
    directory.mkdir(parents=True)
    path = directory / "llm_fake.py"
    path.write_text("display_name = 'Fake'\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr("llm.discover", lambda: calls.append("discover"))
    monkeypatch.setattr(
        "llm.refresh",
        lambda config, **kwargs: calls.append(
            ("refresh", config, kwargs)
        ),
    )
    service = PluginWatcher({"llm_profiles": {}})

    service.handle_create_or_modify(str(path))
    path.unlink()
    service.handle_delete(str(path))

    assert calls == [
        "discover",
        ("refresh", service.config, {"force": True}),
        "discover",
        ("refresh", service.config, {"force": True}),
    ]


def test_a_batch_loads_parsers_and_backends_before_plugins(tmp_path, monkeypatch):
    """Ordering, and the fact that ranking a rootless file does not raise.

    ``_priority`` has to rank files ``plugin_info`` knows nothing about — a
    parser and an LLM backend belong to no family — so it asks the layout
    first. That branch is reached on *every* debounced batch, and it went
    unexercised long enough for a refactor to leave it calling a method that no
    longer existed: an ``AttributeError`` on any group of saves, with a green
    suite. This pins both halves.
    """
    from plugins.plugin_watcher import _PluginEventHandler
    from tests.support import retarget_trees

    workspace = retarget_trees(monkeypatch, tmp_path)["workspace"]
    written = []
    for rel in ("tools/tool_late.py", "parsers/parse_early.py",
                "llm/llm_early.py", "commands/command_mid.py"):
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")
        written.append(str(path))

    watcher = PluginWatcher({})
    handler = _PluginEventHandler(watcher)
    handler.pending = set(written)
    loaded = []
    monkeypatch.setattr(watcher, "handle_create_or_modify", loaded.append)
    handler._fire_batch()

    kinds = [Path(p).parent.name for p in loaded]
    assert len(loaded) == len(written), "every queued file must be loaded"
    assert set(kinds[:2]) == {"parsers", "llm"}, (
        f"parsers and backends must load before plugins, got {kinds}")
    # Deliberately not asserting the order of the remaining two. Ranking a
    # family goes through ``plugin_info``, which resolves against the *real*
    # PLUGIN_CONFIG and cannot place a file in a retargeted tree — so they tie
    # and fall back to set iteration order, which pytest-randomly reseeds. The
    # family ordering is ``_LOAD_PRIORITY``'s to keep; this test is about the
    # rootless branch ahead of it.


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

    monkeypatch.setattr("plugins.plugin_watcher.load_single_plugin", fake_load)
    monkeypatch.setattr("plugins.plugin_watcher.PluginWatcher._reconcile_plugin_config", lambda self: None)
    service = PluginWatcher({})
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
    service = PluginWatcher({})
    service.bind_runtime(command_registry=registry, runtime=runtime)
    service._known_mtimes[str(path.resolve())] = path.stat().st_mtime
    monkeypatch.setattr("plugins.plugin_watcher.unload_plugin", lambda *a, **k: registry.unregister("agent"))
    monkeypatch.setattr("plugins.plugin_watcher.PluginWatcher._reconcile_plugin_config", lambda self: None)

    path.unlink()
    service.handle_delete(str(path))

    assert "agent" not in registry._commands
    assert "agent" not in runtime.commands
    assert runtime.refreshes == 1


def test_plugin_watcher_stop_cancels_pending_timers():
    """Stopping kernel observation cancels pending reload batches."""
    service = PluginWatcher({})
    handler = service._handler = _FakeHandler()

    service.stop()

    assert handler.cancelled


def test_discovery_loads_a_tool_declaring_a_family_local_helper(tmp_path, monkeypatch):
    """A declared helper must not stop a tool being discovered.

    Whether the *import* resolves is the box's business — ``dependencies_files``
    is what puts the helper on the box's path, and the sandbox suite tests
    that. What is asserted here is the discovery half: the declaration is read
    off the file and carried onto the adapter, so a tool that has one still
    registers.
    """
    root = tmp_path / "sandbox_plugins"
    tools = root / "tools"
    helpers = tools / "helpers"
    helpers.mkdir(parents=True)
    (helpers / "answer.py").write_text('VALUE = "relative ok"\n', encoding="utf-8")
    (tools / "tool_relative.py").write_text(
        "dependencies_files = ['tools/helpers/answer.py']\n\n"
        "from guest.bases import BaseTool\n\n"
        "class RelativeTool(BaseTool):\n"
        "    name = 'relative_tool'\n"
        "    description = 'test'\n"
        "    parameters = {}\n"
        "    def run(self, sdk, **kwargs):\n"
        "        from .helpers.answer import VALUE\n"
        "        return sdk.ok(VALUE)\n",
        encoding="utf-8",
    )
    _patch_tool_discovery(monkeypatch, (("sandbox", root, "sandbox_plugins", False),))
    registry = _ToolRegistry()

    plugin_discovery.discover_tools(registry, reload=True)

    assert registry.tools["relative_tool"].dependencies_files == [
        "tools/helpers/answer.py"]


def test_discovery_precedence_prefers_sandbox_over_installed(tmp_path, monkeypatch):
    """Verify earlier roots win name collisions."""
    sandbox = tmp_path / "sandbox_plugins"
    installed = tmp_path / "installed_plugins"
    for root, label in ((sandbox, "sandbox"), (installed, "installed")):
        tools = root / "tools"
        tools.mkdir(parents=True)
        (tools / "tool_same.py").write_text(
            "from guest.bases import BaseTool\n\n"
            "class SameTool(BaseTool):\n"
            "    name = 'same_tool'\n"
            f"    description = '{label}'\n"
            "    parameters = {}\n"
            "    def run(self, sdk, **kwargs):\n"
            "        return sdk.ok(None)\n",
            encoding="utf-8",
        )
    _patch_tool_discovery(
        monkeypatch,
        (("sandbox", sandbox, "sandbox_plugins", False), ("installed", installed, "installed_plugins", False)),
    )
    registry = _ToolRegistry()

    plugin_discovery.discover_tools(registry, reload=True)

    assert registry.tools["same_tool"].description == "sandbox"


def test_load_single_tool_accepts_auto_register_false(tmp_path, monkeypatch):
    """Installing a tool that opts out of auto-registration is a no-op, not a
    failure: the file is on disk and something (e.g. plan mode) registers it on
    demand. Mirrors boot discovery, which silently skips such tools."""
    sandbox = tmp_path / "sandbox_plugins"
    tools = sandbox / "tools"
    tools.mkdir(parents=True)
    (tools / "tool_deferred.py").write_text(
        "from guest.bases import BaseTool\n\n"
        "class Deferred(BaseTool):\n"
        "    name = 'deferred'\n"
        "    description = 'test'\n"
        "    parameters = {}\n"
        "    auto_register = False\n"
        "    def run(self, sdk, **kwargs):\n"
        "        return sdk.ok({})\n",
        encoding="utf-8",
    )
    _patch_tool_discovery(monkeypatch, (("sandbox", sandbox, "sandbox_plugins", False),))
    registry = _ToolRegistry()

    name, error = plugin_discovery._load_single_tool(tools / "tool_deferred.py", registry)

    assert error is None
    assert name == "deferred"
    assert registry.tools == {}  # opted out of the global registry


def test_load_single_tool_rejects_a_file_that_is_not_sdk_code(tmp_path, monkeypatch):
    """A tool the sandbox will not carry is a tool that does not install.

    This used to reach the subclass scan and fail there, because the loader
    would import anything and only the absence of a ``BaseTool`` gave it away.
    The refusal is earlier now — the file never runs at all — which is the
    whole point of dropping the native path.
    """
    sandbox = tmp_path / "sandbox_plugins"
    tools = sandbox / "tools"
    tools.mkdir(parents=True)
    (tools / "tool_empty.py").write_text("VALUE = 1\n", encoding="utf-8")
    _patch_tool_discovery(monkeypatch, (("sandbox", sandbox, "sandbox_plugins", False),))
    registry = _ToolRegistry()

    name, error = plugin_discovery._load_single_tool(tools / "tool_empty.py", registry)

    assert name is None
    assert "tool_empty.py" in error


def test_load_single_tool_rejects_sdk_code_declaring_no_tool(tmp_path, monkeypatch):
    """Importing the SDK is not the same as being a tool.

    The bridge needs a plugin class to build an adapter around, so a file that
    imports ``guest.bases`` and then declares nothing is refused as surely as
    one that never mentioned it — a distinction worth pinning, since the two
    failures now share a message.
    """
    sandbox = tmp_path / "sandbox_plugins"
    tools = sandbox / "tools"
    tools.mkdir(parents=True)
    (tools / "tool_classless.py").write_text(
        "from guest.bases import BaseTool\n\nVALUE = 1\n", encoding="utf-8")
    _patch_tool_discovery(monkeypatch, (("sandbox", sandbox, "sandbox_plugins", False),))
    registry = _ToolRegistry()

    name, error = plugin_discovery._load_single_tool(
        tools / "tool_classless.py", registry)

    assert name is None
    assert "tool_classless.py" in error
    assert registry.tools == {}


class _FakeHandler:
    """Fake handler."""
    cancelled = False

    def cancel_pending(self):
        """Cancel pending."""
        self.cancelled = True


# ────────────────────────────────────────────────────────────────────
# Pipeline wiring (was test_pipeline.py)
# ────────────────────────────────────────────────────────────────────

from events.event_channels import SERVICE_LOADED
from pipeline.orchestrator import Orchestrator


def test_orchestrator_stop_unsubscribes_service_loaded_handler():
    """Verify orchestrator stop unsubscribes service loaded handler."""
    before = len(bus._subs.get(SERVICE_LOADED, []))
    orch = Orchestrator(None, {})
    assert len(bus._subs.get(SERVICE_LOADED, [])) == before + 1
    orch.stop()
    assert len(bus._subs.get(SERVICE_LOADED, [])) == before


# ────────────────────────────────────────────────────────────────────
# Service lifecycle (was test_service_lifecycle.py)
# ────────────────────────────────────────────────────────────────────

from plugins.native.service import BaseService, EXTENSION, forget_stale_autoloads, is_user_managed_service, should_autoload_service


class ManagedService(BaseService):
    """Managed service."""


class ExtensionService(BaseService):
    """Extension service."""
    lifecycle = EXTENSION


def test_base_service_has_default_noop_lifecycle():
    service = BaseService()

    assert service.load() is True
    assert service.loaded is True
    service.unload()
    assert service.loaded is False


def test_extension_services_autoload_without_config_entry():
    managed = ManagedService()
    extension = ExtensionService()

    assert not should_autoload_service("managed", managed, {"autoload_services": []})
    assert should_autoload_service("managed", managed, {"autoload_services": ["managed"]})
    assert should_autoload_service("extension", extension, {"autoload_services": []})
    assert is_user_managed_service(managed)
    assert not is_user_managed_service(extension)


def test_a_stale_autoload_entry_is_dropped_rather_than_warned_about(monkeypatch):
    """A name no live service answers to can only ever be skipped.

    Warning about it on every boot forever is how a person learns to scroll
    past boot warnings — and until the package manager learned to read a
    service's declared name it wrote the *filename* here, so the entries were
    ones no amount of reinstalling would fix. Same answer bootstrap already
    gives for enabled_frontends.
    """
    saved = {}
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))
    config = {"autoload_services": ["timekeeper", "drive", "embed"]}

    gone = forget_stale_autoloads(config, {"timekeeper": object()})

    assert sorted(gone) == ["drive", "embed"]
    assert config["autoload_services"] == ["timekeeper"]
    assert saved["autoload_services"] == ["timekeeper"]


def test_nothing_is_saved_when_every_autoload_entry_resolves(monkeypatch):
    """A boot that changes nothing must not rewrite the config file."""
    writes = []
    monkeypatch.setattr("config.config_manager.save", lambda config: writes.append(config))
    config = {"autoload_services": ["timekeeper"]}

    assert forget_stale_autoloads(config, {"timekeeper": object()}) == []
    assert writes == []


# ────────────────────────────────────────────────────────────────────
# Reinstalling a service that is already registered.
# ────────────────────────────────────────────────────────────────────

_MIGRATED_SERVICE = '''
"""A migrated service."""

from guest.bases import BaseService


class Counter(BaseService):
    """Counts."""

    name = "counter"
    description = "counts"
    exports = ["bump"]

    def start(self, sdk):
        """Nothing to acquire."""
        return True

    def bump(self, sdk, by=1):
        """Add."""
        return by
'''


def test_reinstalling_a_bridged_service_keeps_its_runtime(tmp_path, monkeypatch):
    """One attribute name, two meanings — and the collision only fired live.

    ``_load_single_service`` preserved the bindings of the instance it was
    replacing by reading ``svc._runtime``. The only writer of that attribute
    is the bridge, which stores the live ``ConversationRuntime`` there because
    ``_sync_hooks`` needs ``runtime.hooks``. While services were native
    nothing set it and the read produced ``{}``; once every service is bridged
    it finds a runtime object, and ``dict()`` of one raises
    "'ConversationRuntime' object is not iterable" — so *every reinstall of
    an already-registered service* failed at registration, with the install
    itself already done and the files already on disk.
    """
    from types import SimpleNamespace

    import sandbox  # noqa: F401  - installs the ``guest`` alias
    from sandbox import Sandbox
    from sandbox.bridge import configure

    services_dir = tmp_path / "services"
    services_dir.mkdir()
    _patch_plugin_dir(monkeypatch, services_dir, "service")

    configure(Sandbox())
    try:
        path = services_dir / "service_counter.py"
        path.write_text(_MIGRATED_SERVICE, encoding="utf-8")
        runtime = SimpleNamespace(hooks=None)
        services = {}

        first, error = plugin_discovery._load_single_service(
            path, services, {}, {"runtime": runtime})
        assert error is None, error
        assert first == "counter"
        assert services["counter"]._runtime is runtime

        # The second install is the one that used to fail.
        again, error = plugin_discovery._load_single_service(
            path, services, {}, {"runtime": runtime})
        assert error is None, error
        assert again == "counter"
        # And the bindings survived the replacement, which is what the
        # preservation existed for in the first place.
        assert services["counter"]._runtime is runtime
    finally:
        configure(None)


# ── A root that is not Python ─────────────────────────────────────────
#
# ``widgets/`` is the first one, and everything below is a place that used to
# say ``.py`` out loud. Each of those literals was true of every root when it
# was written, which is exactly why none of them looked like a filter — and a
# widget that is installed but that the watcher never sees is invisible in the
# one direction nothing reports.


def _widget_tree(tmp_path, monkeypatch):
    """A workspace tree with a widget in it, and the layout pointed at it."""
    from tests.support import retarget_trees

    roots = retarget_trees(monkeypatch, tmp_path)
    directory = roots["workspace"] / "widgets"
    directory.mkdir(parents=True)
    path = directory / "widget_file_explorer.html"
    path.write_text("export default {};\n", encoding="utf-8")
    return path


def test_the_layout_places_a_widget_by_its_own_extension(tmp_path, monkeypatch):
    """``root_for`` answers for a root whose files are not Python.

    And answers None for the two near-misses, which is the half that matters:
    a ``.py`` in ``widgets/`` is not a widget, and a helper is not in a root
    at all.
    """
    import trees

    path = _widget_tree(tmp_path, monkeypatch)

    assert trees.root_for(path).name == "widgets"
    assert trees.root_for(path.with_suffix(".py")) is None
    assert trees.root_for(path.parent / "helpers" / "x.html") is None


def test_the_watcher_registers_a_widget_and_loads_nothing(tmp_path, monkeypatch):
    """A widget is announced, and no plugin loader is reached.

    Registering one is *only* the notification: the kernel never loads a
    widget, so there is nothing to refresh and deliberately no registry. The
    negative is the point — an HTML file handed to ``plugin_info`` is an AST
    parse of markup, which reports a working widget as a broken plugin.
    """
    path = _widget_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "plugins.plugin_watcher.load_single_plugin",
        lambda *a, **k: pytest.fail("a widget must not reach the plugin loader"))
    notices = []
    watcher = PluginWatcher({})
    watcher._notify = lambda title, body="", level="info": notices.append(title)

    registered = watcher.register(str(path))
    removed = watcher.unregister(str(path))

    assert registered == {"ok": True, "name": "widget_file_explorer",
                          "family": "widget", "path": str(path.resolve())}
    assert removed["family"] == "widget"
    assert notices == ["Widget registered", "Widget removed"]


def test_the_watcher_baselines_and_hoists_a_widget(tmp_path, monkeypatch):
    """The scan seeds its mtime, and a batch loads it before the families.

    Without the baseline its first save after start-up reads as a brand-new
    file. The hoist is ``plugin_info``'s problem again: it cannot rank a file
    belonging to no family, and for this one cannot even read it.
    """
    path = _widget_tree(tmp_path, monkeypatch)
    watcher = PluginWatcher({})

    watcher._scan_existing()

    assert str(path.resolve()) in watcher._known_mtimes
    assert watcher._root_of(path) in plugin_watcher._UNFAMILIED_ROOTS


def test_the_store_installs_a_widget_and_checks_it_against_its_own_root():
    """A store path is valid against the root's extension, not against Python."""
    from bundled.commands.helpers import package_manager

    assert package_manager._validate_rel_path(
        "widgets/widget_file_explorer.html") == "widgets/widget_file_explorer.html"
    assert not package_manager._is_valid_tree_rel("widgets/widget_x.py")
    assert not package_manager._is_valid_tree_rel("widgets/x.html")
    assert not package_manager._is_valid_tree_rel("tools/tool_x.html")
    # A helper is Python whatever its family holds: it is imported by the
    # plugin beside it, and nothing in a browser imports from this tree.
    assert package_manager._is_valid_tree_rel("tools/helpers/x.py")


def test_the_sdk_lists_widgets_from_every_tree_first_match_winning(tmp_path, monkeypatch):
    """``plugin.list(source="widgets")`` is how the browser learns what exists.

    Precedence is every other discoverer's — workspace over installed over
    bundled — but the shadowed file is *reported* rather than dropped,
    because "my widget is not showing up" cannot be answered from a browser
    that can read no disk, and neither can "did my override take".

    """
    from tests.support import retarget_trees
    from sandbox.handlers.kernel import _plugin_list

    roots = retarget_trees(monkeypatch, tmp_path)
    for tree, name in (("installed", "widget_file_explorer.html"),
                       ("workspace", "widget_file_explorer.html"),
                       ("workspace", "widget_notes.html")):
        directory = roots[tree] / "widgets"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text("", encoding="utf-8")

    rows = _plugin_list(None, {"source": "widgets"}).data

    assert [row["name"] for row in rows] == ["file_explorer", "notes"]
    assert rows[0]["tree"] == "workspace"
    assert rows[0]["shadowed"] == [
        str(roots["installed"] / "widgets" / "widget_file_explorer.html")]
    assert rows[1]["extension"] == ".html"
    # No URL. Fetching one is the transport's business, and a second client
    # would have a different one.
    assert not any("url" in row for row in rows)


# ── Precedence: the most specific tree wins ──────────────────────────
#
# One ordering, read by everything: ``trees.TREES``. The tests below pin the
# table itself, then the three behaviours that hang off it — a draft
# overriding what it is a draft of, a *broken* draft not taking anything down
# with it, and the validator not calling a deliberate override a mistake.


def _tool_source(name: str, label: str) -> str:
    return ("from guest.bases import BaseTool\n\n"
            "class SameTool(BaseTool):\n"
            f"    name = '{name}'\n"
            f"    description = '{label}'\n"
            "    parameters = {}\n"
            "    def run(self, sdk, **kwargs):\n"
            "        return sdk.ok(None)\n")


def _discover_over_trees(monkeypatch, roots):
    """Run tool discovery across real retargeted trees, in table order."""
    import trees
    import plugins.plugin_paths as paths

    _patch_tool_discovery(
        monkeypatch,
        tuple((tree.name, roots[tree.name], tree.module, tree.builtin)
              for tree in trees.TREES),
    )
    registry = _ToolRegistry()
    plugin_discovery.discover_tools(registry, reload=True)
    return registry


def test_the_layout_table_ranks_the_workspace_first():
    """Precedence is a property of ``TREES`` and nothing restates it.

    Every discoverer walks this tuple and keeps a seen-set, so the order here
    *is* the answer to "who wins a name collision". Pinned as a literal
    because the direction was reversed deliberately: the more specific tree
    wins, which is what makes a workspace draft the way to revise a bundled
    capability that cannot be uninstalled.
    """
    import trees

    assert [tree.name for tree in trees.TREES] == [
        "workspace", "installed", "bundled"]


def test_script_resolution_reads_the_same_direction_as_discovery():
    """``resolve_script`` used to reverse the table and no longer may.

    It resolves a *filename* an agent typed, and the agent means the one it
    wrote — which used to disagree with discovery resolving a *capability
    name*. The two agree now, so a ``reversed`` here would silently hand back
    the bundled copy of a script the agent had just drafted.
    """
    import inspect

    from sandbox import isolation

    assert "reversed" not in inspect.getsource(isolation.resolve_script)


def test_a_workspace_draft_overrides_the_bundled_tool_it_revises(
        tmp_path, monkeypatch):
    """The whole point: shadowing is how you revise what you cannot uninstall.

    An installed package can be removed and replaced. A bundled one cannot, so
    without this there is no way to try a change to a kernel tool except by
    editing the kernel's own tree — which is editing the app rather than
    writing a plugin.
    """
    from tests.support import retarget_trees

    roots = retarget_trees(monkeypatch, tmp_path)
    for tree in ("bundled", "installed", "workspace"):
        tools = roots[tree] / "tools"
        tools.mkdir(parents=True)
        (tools / "tool_same.py").write_text(
            _tool_source("same_tool", tree), encoding="utf-8")

    registry = _discover_over_trees(monkeypatch, roots)

    assert registry.tools["same_tool"].description == "workspace"


def test_a_draft_that_will_not_load_shadows_nothing(tmp_path, monkeypatch):
    """Failure falls back, and it is the loop that says so rather than a rule.

    A file that does not load never reaches the seen-set, so the tree below it
    registers as though the draft were not there. That is what keeps a
    half-written override from taking a working kernel tool off the air —
    the failure mode that would otherwise make overriding too sharp to use.
    """
    from tests.support import retarget_trees

    roots = retarget_trees(monkeypatch, tmp_path)
    for tree in ("bundled", "workspace"):
        (roots[tree] / "tools").mkdir(parents=True)
    (roots["bundled"] / "tools" / "tool_same.py").write_text(
        _tool_source("same_tool", "bundled"), encoding="utf-8")
    (roots["workspace"] / "tools" / "tool_same.py").write_text(
        "class Broken(:\n", encoding="utf-8")

    registry = _discover_over_trees(monkeypatch, roots)

    assert registry.tools["same_tool"].description == "bundled"


def test_the_validator_does_not_call_a_deliberate_override_a_duplicate(
        tmp_path, monkeypatch):
    """``plugin.validate`` must not refuse the one thing precedence allows.

    Its duplicate-name check exists to catch an *accidental* collision. A
    workspace file taking a bundled name is not one — it is the override the
    ordering is for — so the name is dropped before the check runs. The
    reverse is still reported: there the file being checked is the one that
    loses, which is the case an author cannot otherwise see.
    """
    from types import SimpleNamespace

    from sandbox.handlers.kernel import _known_names
    from tests.support import retarget_trees

    roots = retarget_trees(monkeypatch, tmp_path)
    bundled = roots["bundled"] / "tools" / "tool_same.py"
    workspace = roots["workspace"] / "tools" / "tool_same.py"
    for path in (bundled, workspace):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    registry = SimpleNamespace(tools={})
    ctx = SimpleNamespace(tool_registry=registry)

    registry.tools = {"same_tool": SimpleNamespace(_source_path=str(bundled))}
    assert _known_names(ctx, workspace) == []

    registry.tools = {"same_tool": SimpleNamespace(_source_path=str(workspace))}
    assert _known_names(ctx, bundled) == ["same_tool"]


class _LiveToolRegistry(_ToolRegistry):
    """A registry that actually forgets, unlike the recording double above."""

    def unregister(self, name):
        super().unregister(name)
        self.tools.pop(name, None)


def test_deleting_an_override_restores_what_it_was_shadowing(
        tmp_path, monkeypatch):
    """Reverting is deleting the draft, and it must not need a restart.

    Unloading alone unregisters by source path and re-scans nothing, so the
    capability was simply *gone* until the next boot — which makes overriding
    a kernel plugin a one-way door in exactly the case you most want to back
    out of. A deleted parser or LLM backend already came back, because those
    watcher paths answer with a full rescan; only the five families stopped at
    the unload.
    """
    import trees

    roots = {}
    for tree in trees.TREES:
        tools = tmp_path / tree.name / "tools"
        tools.mkdir(parents=True)
        roots[tree.name] = tmp_path / tree.name
        (tools / "tool_same.py").write_text(
            _tool_source("same_tool", tree.name), encoding="utf-8")

    _patch_tool_discovery(
        monkeypatch,
        tuple((tree.name, roots[tree.name], tree.module, tree.builtin)
              for tree in trees.TREES),
    )
    registry = _LiveToolRegistry()
    plugin_discovery.discover_tools(registry, reload=True)
    assert registry.tools["same_tool"].description == "workspace"

    draft = roots["workspace"] / "tools" / "tool_same.py"
    draft.unlink()
    watcher = PluginWatcher({}, tool_registry=registry)
    outcome = watcher.unregister(draft)

    assert outcome["restored"] == "same_tool"
    # One tree down, not all the way to bundled: reverting a draft hands back
    # the package it was a draft *of*.
    assert registry.tools["same_tool"].description == "installed"


def test_deleting_a_shadowed_file_restores_nothing(tmp_path, monkeypatch):
    """A file that held no name had nothing to give back.

    Deleting the *bundled* copy while a workspace draft is live must not
    reload anything: the draft is still registered and still correct, and a
    restore here would either be a no-op or would quietly demote it.
    """
    import trees

    roots = {}
    for tree in trees.TREES:
        tools = tmp_path / tree.name / "tools"
        tools.mkdir(parents=True)
        roots[tree.name] = tmp_path / tree.name
        (tools / "tool_same.py").write_text(
            _tool_source("same_tool", tree.name), encoding="utf-8")

    _patch_tool_discovery(
        monkeypatch,
        tuple((tree.name, roots[tree.name], tree.module, tree.builtin)
              for tree in trees.TREES),
    )
    registry = _LiveToolRegistry()
    plugin_discovery.discover_tools(registry, reload=True)

    bundled = roots["bundled"] / "tools" / "tool_same.py"
    bundled.unlink()
    outcome = PluginWatcher({}, tool_registry=registry).unregister(bundled)

    assert outcome["names"] == []
    assert outcome["restored"] is None
    assert registry.tools["same_tool"].description == "workspace"
