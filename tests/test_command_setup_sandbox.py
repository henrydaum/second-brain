"""Sandboxed ``/setup`` onboarding and supporting SDK coverage."""

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
from sandbox import Result, Sandbox
from sandbox.bridge import adapt, configure
from sandbox.console import CONSOLE
from sandbox.guest.requests import NET_HTTP
from sandbox.handlers import HANDLERS
from sandbox.handlers.kernel import _config_write, _plugin_list


def _context(tmp_path, config=None):
    config = config or {}
    runtime = SimpleNamespace(config=config, refresh_session_specs=lambda: None)
    return SimpleNamespace(
        root_dir=tmp_path,
        config=config,
        runtime=runtime,
        services={},
        session_key="chat",
        db=None,
        user_id=1,
    )


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_setup.py",
            "SetupCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_plugin_list_can_discover_llm_backend_role(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "llm.backend_names",
        lambda: ["LiteLLMService", "OtherBackend"],
    )

    result = _plugin_list(
        _context(tmp_path),
        {
            "source": "registered",
            "category": "services",
            "role": "llm_backend",
        },
    )

    assert result.ok
    assert result.data == ["LiteLLMService", "OtherBackend"]


def test_setup_form_preserves_dependent_atlas_steps(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "llm.backend_names",
        lambda: ["LiteLLMService"],
    )
    monkeypatch.setattr(package_manager, "installed_packages", lambda: [])
    context = _context(tmp_path)

    initial = _run(context, {}, method="form")
    atlas = _run(
        context, {"llm_choice": "atlas"}, method="form")
    direct = _run(
        context,
        {"llm_choice": "atlas", "key_source": "direct"},
        method="form",
    )

    assert [step["name"] for step in initial.data] == ["llm_choice"]
    assert [step["name"] for step in atlas.data] == [
        "llm_choice", "key_source"]
    assert [step["name"] for step in direct.data] == [
        "llm_choice", "key_source", "api_key", "model_name"]


def test_setup_fresh_form_starts_with_package_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "llm.backend_names", lambda: [])
    context = _context(tmp_path)

    initial = _run(context, {}, method="form")
    starter = _run(
        context, {"install_choice": "bundle_starter"}, method="form")
    completed = _run(
        context,
        {
            "install_choice": "bundle_starter",
            "llm_choice": "other",
            "other_model_name": "openai/test",
            "other_service_class": "LiteLLMService",
        },
        method="form",
    )

    assert [step["name"] for step in initial.data] == ["install_choice"]
    assert [step["name"] for step in starter.data][:2] == [
        "install_choice", "llm_choice"]
    assert completed.data[-1]["name"] == "telegram_choice"


def test_setup_profile_merge_stays_kernel_side_and_preserves_secrets(
        monkeypatch, tmp_path):
    saved_plugin = {}
    monkeypatch.setattr(
        "config.config_manager.save", lambda _config: None)
    monkeypatch.setattr(
        "config.config_manager.load_plugin_config",
        lambda: {"unrelated": True},
    )
    monkeypatch.setattr(
        "config.config_manager.save_plugin_config",
        lambda values: saved_plugin.update(values),
    )
    config = {
        "llm_profiles": {
            "existing": {
                "secret_llm_api_key": "must-not-cross-the-sandbox",
            },
        },
    }
    context = _context(tmp_path, config)

    result = _run(
        context,
        {
            "llm_choice": "other",
            "other_model_name": "openai/test",
            "other_service_class": "LiteLLMService",
            "telegram_choice": "skip",
        },
        approve=lambda *_: True,
    )

    assert result.ok, result.error
    assert config["llm_profiles"]["existing"]["secret_llm_api_key"] == (
        "must-not-cross-the-sandbox")
    assert "openai/test" in config["llm_profiles"]
    assert config["default_llm_profile"] == "openai/test"
    assert saved_plugin["unrelated"] is True
    assert saved_plugin["llm_profiles"] == config["llm_profiles"]
    # Preserve the native command's early return for the generic-provider
    # branch; its LLM confirmation is the complete response.
    assert result.data.startswith(
        "LLM: profile `openai/test` added and set as default.")


def test_config_write_merge_rejects_non_mapping(tmp_path):
    context = _context(tmp_path, {"value": "plain"})

    result = _config_write(
        context, {"key": "value", "value": {"x": 1}, "merge": True})

    assert not result.ok
    assert result.error == "config setting 'value' is not a mapping"


def test_setup_skip_output_is_exact(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "llm.backend_names", lambda: [])

    result = _run(_context(tmp_path), {"install_choice": "skip"})

    assert result.data == (
        "Skipped package install.\n\n"
        "Second Brain needs at least an LLM backend before it can do anything. "
        "When you're ready:\n"
        "  /packages install bundle_starter   — the recommended baseline\n"
        "  /packages install bundle_full      — everything\n"
        "  /packages available        — browse the store by category\n\n"
        "Then run /setup again to configure your LLM and Telegram."
    )


def test_setup_bundle_install_uses_kernel_store(monkeypatch, tmp_path):
    class Outcome:
        def text(self):
            return "Installed 3 files."

    monkeypatch.setitem(
        HANDLERS, NET_HTTP, lambda _ctx, _args: Result(data=""))
    monkeypatch.setattr(
        package_manager,
        "install_package",
        lambda _root, package_id, _context, **_kwargs: (
            Outcome() if package_id == "bundle_starter" else None
        ),
    )
    context = _context(tmp_path)

    result = _run(
        context,
        {"install_choice": "bundle_starter"},
        approve=lambda *_: True,
    )

    assert result.ok, result.error
    assert result.data.startswith(
        "Installed the `bundle_starter` bundle.\n  Installed 3 files.")


def test_setup_denied_config_write_stays_denied(tmp_path):
    result = _run(
        _context(tmp_path),
        {
            "llm_choice": "other",
            "other_model_name": "openai/test",
            "other_service_class": "LiteLLMService",
        },
        approve=lambda *_: False,
    )

    assert not result.ok
    assert "config.write" in result.error
    assert "denied" in result.error.lower()


def test_live_repl_collects_setup_package_choice(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "llm.backend_names", lambda: [])
    db = Database(str(tmp_path / "setup-live.db"))
    config = {}
    services = {}
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
            source=PacedInput("/setup\nskip\n"),
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
                "Then run /setup again" in text for text in written):
            time.sleep(0.01)
        output = "".join(written)
        assert "Welcome to Second Brain." in output
        assert "Install the starter bundle" in output
        assert "Skipped package install." in output
        assert "Then run /setup again" in output
    finally:
        frontend.unbind()
        frontend.stop()
        thread.join(timeout=2)
        sandbox.shutdown()
        configure(None)
