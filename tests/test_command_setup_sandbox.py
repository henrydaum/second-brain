"""Sandboxed ``/setup`` onboarding and supporting SDK coverage."""

from types import SimpleNamespace

from plugins.commands.helpers import package_manager
from sandbox import Result, Sandbox
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
