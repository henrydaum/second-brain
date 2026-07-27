"""Kernel package requests and sandboxed ``/packages`` coverage."""

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from pipeline.database import Database
from plugins.commands.helpers import package_manager
from plugins.frontends.helpers.command_registry import CommandRegistry
from plugins.plugin_discovery import discover_commands
from runtime.context import build_context
from runtime.conversation_runtime import ConversationRuntime
from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.console import CONSOLE
from sandbox.guest.requests import (
    PLUGIN_INSTALL,
    PLUGIN_UNINSTALL,
    PLUGIN_UPDATE,
    Request,
)
from sandbox.policy import ALWAYS_UNSAFE


ITEMS = [
    {
        "id": "tool_alpha",
        "name": "tool_alpha",
        "path": "tools/tool_alpha.py",
        "family": "tools",
        "helper": False,
        "installed": False,
    },
    {
        "id": "shared",
        "name": "shared",
        "path": "tools/helpers/shared.py",
        "family": "tools",
        "helper": True,
        "installed": False,
    },
]


def _context(tmp_path, pushes=None):
    runtime = SimpleNamespace(
        sessions={},
        push_message=lambda key, message, **kwargs: (
            pushes.append((key, message, kwargs)) if pushes is not None
            else None
        ),
    )
    return SimpleNamespace(
        root_dir=tmp_path,
        runtime=runtime,
        session_key="chat",
        services={},
        config={},
        db=None,
    )


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_packages.py",
            "PackagesCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def _patch_catalog(monkeypatch):
    monkeypatch.setattr(
        package_manager, "search_packages",
        lambda _root: [dict(item) for item in ITEMS],
    )
    monkeypatch.setattr(package_manager, "installed_packages", lambda: [])
    monkeypatch.setattr(package_manager, "removable_packages", lambda: [
        {**ITEMS[0], "installed": True},
    ])
    monkeypatch.setattr(package_manager, "search_bundles", lambda _root: [{
        "id": "bundle_starter",
        "name": "Starter",
        "path": "bundles/bundle_starter.json",
        "family": "bundles",
        "helper": False,
        "installed": False,
    }])


def test_packages_form_recomputes_dependent_steps(tmp_path, monkeypatch):
    _patch_catalog(monkeypatch)
    context = _context(tmp_path)

    initial = _run(context, {}, method="form")
    available = _run(context, {"action": "available"}, method="form")
    install = _run(context, {"action": "install"}, method="form")
    uninstall = _run(context, {"action": "uninstall"}, method="form")
    update = _run(context, {"action": "update"}, method="form")

    assert initial.ok
    assert [step["name"] for step in initial.data] == ["action"]
    assert [step["name"] for step in available.data] == [
        "action", "category"]
    assert "Available files by category:" in available.data[1]["prompt"]
    assert "| Tools | 2 | agent-callable tools |" in available.data[1]["prompt"]
    assert [step["name"] for step in install.data] == [
        "action", "package_id"]
    assert uninstall.data[1]["enum"] == [
        "tool_alpha", "bundle_starter"]
    assert [step["name"] for step in update.data] == ["action"]


def test_packages_browse_output_matches_native_wire_format(
        tmp_path, monkeypatch):
    _patch_catalog(monkeypatch)

    result = _run(
        _context(tmp_path), {"action": "available", "category": "tools"})

    assert result.ok, result.error
    assert result.data == (
        "Available tool plugins:\n\n"
        "| Name | Path |\n"
        "| --- | --- |\n"
        "| tool_alpha | tools/tool_alpha.py |\n"
        "| shared (helper) | tools/helpers/shared.py |\n\n"
        "Install with `/packages install <name>`."
    )


def test_package_mutations_are_unsafe_and_forward_progress(
        tmp_path, monkeypatch):
    pushes = []

    class Outcome:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

    def install(root, target, context, progress=None, **kwargs):
        assert root == tmp_path
        assert target == "tool_alpha"
        progress("Copying package files")
        return Outcome("Installed file: tools/tool_alpha.py")

    monkeypatch.setattr(package_manager, "install_package", install)
    result = _run(
        _context(tmp_path, pushes),
        {"action": "install", "package_id": "tool_alpha"},
        approve=lambda *_: True,
    )

    assert {PLUGIN_INSTALL, PLUGIN_UNINSTALL, PLUGIN_UPDATE} <= ALWAYS_UNSAFE
    assert Request(PLUGIN_UPDATE).read_only is False
    assert result.ok, result.error
    assert result.data == "Installed file: tools/tool_alpha.py"
    assert pushes == [(
        "chat", "Copying package files", {"source": "packages"})]


def test_packages_preserves_store_failure_message(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise package_manager.PackageError("missing package: nope")

    monkeypatch.setattr(package_manager, "install_package", fail)
    result = _run(
        _context(tmp_path),
        {"action": "install", "package_id": "nope"},
        approve=lambda *_: True,
    )

    assert result.ok
    assert result.data == "Package install failed: missing package: nope"


def test_packages_adapter_keeps_agent_prompt():
    module = adapt(Path("plugins/commands/command_packages.py").resolve())
    command_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    assert "next turn" in command_cls().agent_prompt_for(None)


def test_live_repl_collects_multistep_packages_form(
        tmp_path, monkeypatch):
    """REPL -> action step -> category step -> sandboxed command."""
    _patch_catalog(monkeypatch)
    db = Database(str(tmp_path / "packages-live.db"))
    holder = {}
    registry = CommandRegistry(
        lambda key=None: build_context(
            db, {}, {}, runtime=holder.get("runtime"),
            root_dir=tmp_path, session_key=key,
        )
    )
    discover_commands(tmp_path, registry, {})
    runtime = ConversationRuntime(
        db=db, services={}, config={}, commands=registry.to_callable_specs())
    holder["runtime"] = runtime

    sandbox = Sandbox()
    configure(sandbox)
    written = []
    original_claim = CONSOLE.claim

    class PacedInput(io.StringIO):
        """A terminal user waits for each dependent prompt."""

        def readline(self, *args, **kwargs):
            if self.tell():
                time.sleep(0.25)
            return super().readline(*args, **kwargs)

    def claim(token, source=None, writer=None):
        return original_claim(
            token,
            source=PacedInput("/packages\navailable\ntools\n"),
            writer=written.append,
        )

    monkeypatch.setattr(CONSOLE, "claim", claim)
    module = adapt(Path("plugins/frontends/frontend_repl.py").resolve())
    frontend_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    frontend = frontend_cls(shutdown_event=threading.Event())
    frontend.bind(runtime, registry, {})
    thread = threading.Thread(target=frontend.start, daemon=True)

    try:
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline and not any(
                "Available tool plugins" in text for text in written):
            time.sleep(0.01)
        output = "".join(written)
        assert "Choose a package action." in output
        assert "Choose a category." in output
        assert "Available tool plugins:" in output
        assert "tool_alpha" in output
    finally:
        frontend.unbind()
        frontend.stop()
        thread.join(timeout=2)
        sandbox.shutdown()
        configure(None)
