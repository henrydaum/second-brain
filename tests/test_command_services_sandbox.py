"""Service lifecycle SDK and sandboxed ``/services`` coverage."""

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from pipeline.database import Database
from plugins import plugin_discovery
from plugins.BaseService import BaseService, EXTENSION
from plugins.frontends.helpers.command_registry import CommandRegistry
from plugins.plugin_discovery import discover_commands
from runtime.context import build_context
from runtime.conversation_runtime import ConversationRuntime
from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.console import CONSOLE
from sandbox.guest.requests import (
    SERVICE_LOAD,
    SERVICE_UNLOAD,
    Request,
)
from sandbox.handlers.kernel import _service_list
from sandbox.policy import ALWAYS_UNSAFE


class ManagedService(BaseService):
    model_name = "Embedder"
    config_settings = [
        (
            "Embed Model",
            "embed_model_name_services_test",
            "Model.",
            "default-model",
            {"type": "text"},
        ),
    ]


class ExtensionService(BaseService):
    model_name = "Watcher"
    lifecycle = EXTENSION


class Orchestrator:
    def __init__(self):
        self.clears = 0

    def clear_skip_cache(self):
        self.clears += 1


def _context(services, config=None):
    config = config or {"autoload_services": []}
    runtime = SimpleNamespace(config=config)
    return SimpleNamespace(
        services=services,
        config=config,
        runtime=runtime,
        orchestrator=Orchestrator(),
        session_key="chat",
        db=None,
        user_id=1,
    )


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_services.py",
            "ServicesCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_service_list_details_is_structured_and_default_is_unchanged():
    managed = ManagedService()
    context = _context({"embedder": managed})

    plain = _service_list(context, {})
    detailed = _service_list(context, {"details": True})

    assert plain.data == {"embedder": False}
    assert detailed.data == [{
        "name": "embedder",
        "loaded": False,
        "model_name": "Embedder",
        "lifecycle": "managed",
        "config_settings": [{
            "title": "Embed Model",
            "key": "embed_model_name_services_test",
            "description": "Model.",
            "default": "default-model",
            "info": {"type": "text"},
            "current": None,
        }],
    }]


def test_services_form_uses_guest_formstep_for_dependent_actions():
    managed = ManagedService()
    context = _context({"embedder": managed})

    initial = _run(context, {}, method="form")
    selected = _run(
        context, {"service_name": "embedder"}, method="form")
    editing = _run(
        context,
        {
            "service_name": "embedder",
            "action": "edit_setting:embed_model_name_services_test",
        },
        method="form",
    )

    assert [step["name"] for step in initial.data] == ["service_name"]
    assert [step["name"] for step in selected.data] == [
        "service_name", "action"]
    assert selected.data[1]["enum"] == [
        "toggle_loaded",
        "toggle_autoload",
        "edit_setting:embed_model_name_services_test",
    ]
    assert selected.data[1]["enum_labels"] == [
        "Load it", "Autoload on startup", "Edit Embed Model"]
    assert [step["name"] for step in editing.data] == [
        "service_name", "action", "value"]


def test_service_lifecycle_requests_are_unsafe_and_clear_task_cache():
    managed = ManagedService()
    context = _context({"embedder": managed})

    loaded = _run(
        context,
        {"service_name": "embedder", "action": "toggle_loaded"},
        approve=lambda *_: True,
    )
    unloaded = _run(
        context,
        {"service_name": "embedder", "action": "toggle_loaded"},
        approve=lambda *_: True,
    )

    assert {SERVICE_LOAD, SERVICE_UNLOAD} <= ALWAYS_UNSAFE
    assert not Request(SERVICE_LOAD, {"name": "embedder"}).read_only
    assert loaded.data == "Loaded service: embedder"
    assert unloaded.data == "Unloaded service: embedder"
    assert context.orchestrator.clears == 2
    assert managed.loaded is False


def test_services_extension_has_no_lifecycle_actions():
    extension = ExtensionService()
    extension.load()
    context = _context({"extension": extension})

    form = _run(
        context, {"service_name": "extension"}, method="form")
    shown = _run(context, {"service_name": "extension"})
    blocked = _run(
        context, {"service_name": "extension", "action": "unload"})

    assert [step["name"] for step in form.data] == ["service_name"]
    assert "| Status | Extension |" in shown.data
    assert blocked.data == (
        "extension is an installed extension and is loaded automatically.")


def test_services_toggle_autoload_persists_through_config_request(
        monkeypatch):
    saved = {}
    monkeypatch.setattr(
        "config.config_manager.save", lambda config: saved.update(config))
    context = _context(
        {"embedder": ManagedService()},
        {"autoload_services": ["llm"]},
    )

    enabled = _run(
        context,
        {"service_name": "embedder", "action": "toggle_autoload"},
        approve=lambda *_: True,
    )
    disabled = _run(
        context,
        {"service_name": "embedder", "action": "toggle_autoload"},
        approve=lambda *_: True,
    )

    assert enabled.data == (
        "embedder will now load automatically on startup.")
    assert disabled.data == (
        "embedder will no longer load automatically on startup.")
    assert saved["autoload_services"] == ["llm"]


def test_services_setting_quicklink_uses_config_write(monkeypatch):
    service = ManagedService()
    plugin_discovery._collect_config_settings(
        service, service_names=["embedder"], plugin_type="service")
    saved = {}
    monkeypatch.setattr(
        "config.config_manager.save", lambda config: saved.update(config))
    monkeypatch.setattr(
        "config.config_manager.load_plugin_config", lambda: {})
    monkeypatch.setattr(
        "config.config_manager.save_plugin_config",
        lambda config: saved.update(config),
    )
    context = _context({"embedder": service})
    try:
        result = _run(
            context,
            {
                "service_name": "embedder",
                "action": (
                    "edit_setting:embed_model_name_services_test"
                ),
                "value": "new-model",
            },
            approve=lambda *_: True,
        )
    finally:
        key = "embed_model_name_services_test"
        plugin_discovery._plugin_settings[:] = [
            entry for entry in plugin_discovery._plugin_settings
            if entry[1] != key
        ]
        plugin_discovery._plugin_settings_keys.discard(key)
        plugin_discovery._plugin_setting_types.pop(key, None)
        plugin_discovery._setting_to_services.pop(key, None)
        plugin_discovery._setting_to_plugins.pop(key, None)

    assert result.data == (
        "Set embed_model_name_services_test = new-model")
    assert saved["embed_model_name_services_test"] == "new-model"


def test_live_repl_collects_services_form_and_describes_extension(
        tmp_path, monkeypatch):
    extension = ExtensionService()
    extension.load()
    services = {"extension": extension}
    config = {"autoload_services": []}
    db = Database(str(tmp_path / "services-live.db"))
    holder = {}
    registry = CommandRegistry(
        lambda key=None: build_context(
            db, config, services, runtime=holder.get("runtime"),
            root_dir=tmp_path, session_key=key,
        )
    )
    discover_commands(tmp_path, registry, config)
    runtime = ConversationRuntime(
        db=db,
        services=services,
        config=config,
        commands=registry.to_callable_specs(),
    )
    holder["runtime"] = runtime

    sandbox = Sandbox()
    configure(sandbox)
    written = []
    original_claim = CONSOLE.claim

    class PacedInput(io.StringIO):
        def readline(self, *args, **kwargs):
            if self.tell():
                time.sleep(0.25)
            return super().readline(*args, **kwargs)

    def claim(token, source=None, writer=None):
        return original_claim(
            token,
            source=PacedInput("/services\nextension\n"),
            writer=written.append,
        )

    monkeypatch.setattr(CONSOLE, "claim", claim)
    module = adapt(Path("plugins/frontends/frontend_repl.py").resolve())
    frontend_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    frontend = frontend_cls(shutdown_event=threading.Event())
    frontend.bind(runtime, registry, config)
    thread = threading.Thread(target=frontend.start, daemon=True)

    try:
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline and not any(
                "Watcher" in text for text in written):
            time.sleep(0.01)
        output = "".join(written)
        assert "Select a service." in output
        assert "extension" in output
        assert "Status" in output
        assert "Extension" in output
        assert "Watcher" in output
    finally:
        frontend.unbind()
        frontend.stop()
        thread.join(timeout=2)
        sandbox.shutdown()
        configure(None)
