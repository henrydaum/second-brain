"""Secret-safe config proxies and sandboxed ``/llm`` coverage."""

from types import SimpleNamespace

from plugins import plugin_discovery
from sandbox import Sandbox
from sandbox.handlers.kernel import _config_read


class FakeLlm:
    lifecycle = "managed"

    def __init__(self, loaded=False):
        self.loaded = loaded
        self.model_name = "demo"
        self.config_settings = []

    def load(self):
        self.loaded = True
        return True

    def unload(self):
        self.loaded = False


def _context(monkeypatch, profiles=None, default="a"):
    if profiles is None:
        profiles = {
            "a": {
                "llm_service_class": "FakeBackend",
                "secret_llm_api_key": "TOP-SECRET",
                "llm_context_size": 0,
            },
            "b": {"llm_service_class": "FakeBackend"},
        }
    config = {
        "llm_profiles": profiles,
        "default_llm_profile": default,
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
    monkeypatch.setattr(
        "plugins.services.service_llm.refresh_llm_profile_services",
        lambda services, values: False,
    )
    monkeypatch.setattr(
        "plugins.services.service_llm.llm_backend_names",
        lambda: ["FakeBackend"],
    )
    services = {"a": FakeLlm(), "b": FakeLlm()}
    runtime = SimpleNamespace(
        config=config, refresh_session_specs=lambda: None)
    context = SimpleNamespace(
        config=dict(config),
        runtime=runtime,
        services=services,
        db=None,
        user_id=1,
        session_key="chat",
    )
    return context, saved


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_llm.py",
            "LlmCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_nested_secret_config_values_are_proxied(monkeypatch):
    context, _ = _context(monkeypatch)

    result = _config_read(context, {"key": "llm_profiles"})

    key = result.data["a"]["secret_llm_api_key"]
    assert key.startswith("<secret:path_")
    assert "TOP-SECRET" not in str(result.data)


def test_llm_form_and_description_use_structured_sdk_state(monkeypatch):
    context, _ = _context(monkeypatch)

    selected = _run(context, {"model_name": "a"}, method="form")
    adding = _run(context, {"model_name": "add"}, method="form")
    editing = _run(
        context,
        {"model_name": "a", "action": "edit"},
        method="form",
    )

    assert selected.data[0]["prompt"].endswith("Default: a")
    assert selected.data[1]["enum"] == [
        "edit", "set_default", "load", "unload", "remove"]
    assert adding.data[1]["enum"] == ["FakeBackend"]
    assert "secret_llm_api_key" in [
        step["name"] for step in adding.data]
    assert editing.data[-1]["type"] == "string"


def test_llm_add_preserves_secret_name_and_sets_first_default(monkeypatch):
    context, saved = _context(monkeypatch, profiles={}, default="")

    result = _run(
        context,
        {
            "model_name": "add",
            "new_model_name": "openai/demo",
            "llm_service_class": "FakeBackend",
            "llm_endpoint": "",
            "secret_llm_api_key": "OPENAI_API_KEY",
            "llm_context_size": 0,
            "llm_capability_image": True,
        },
        approve=lambda *_: True,
    )

    assert result.data == "Added LLM profile: openai/demo"
    profile = saved["llm_profiles"]["openai/demo"]
    assert profile["secret_llm_api_key"] == "OPENAI_API_KEY"
    assert profile["llm_capabilities"] == {"image": True}
    assert saved["default_llm_profile"] == "openai/demo"


def test_llm_edit_round_trips_proxy_without_losing_secret(monkeypatch):
    context, saved = _context(monkeypatch)

    result = _run(
        context,
        {
            "model_name": "a",
            "action": "edit",
            "field": "llm_endpoint",
            "value": "https://example.test",
        },
        approve=lambda *_: True,
    )

    assert result.data == "Updated LLM profile: a"
    assert saved["llm_profiles"]["a"]["secret_llm_api_key"] == "TOP-SECRET"
    assert saved["llm_profiles"]["a"]["llm_endpoint"] == (
        "https://example.test")


def test_llm_rename_default_remove_and_lifecycle(monkeypatch):
    context, saved = _context(monkeypatch)

    renamed = _run(
        context,
        {
            "model_name": "a",
            "action": "edit",
            "field": "llm_model_name",
            "value": "c",
        },
        approve=lambda *_: True,
    )
    renamed_default = saved["default_llm_profile"]
    context.config.update(saved)
    context.runtime.config.update(saved)
    context.services["c"] = context.services.pop("a")
    loaded = _run(
        context, {"model_name": "c", "action": "load"},
        approve=lambda *_: True)
    unloaded = _run(
        context, {"model_name": "c", "action": "unload"},
        approve=lambda *_: True)
    removed = _run(
        context, {"model_name": "c", "action": "remove"},
        approve=lambda *_: True)

    assert renamed.data == "Updated LLM profile: c"
    assert renamed_default == "c"
    assert loaded.data == "Loaded LLM profile: c"
    assert unloaded.data == "Unloaded LLM profile: c"
    assert removed.data == "Removed LLM profile: c"


def test_legacy_llm_api_key_is_migrated_on_write(monkeypatch):
    context, saved = _context(
        monkeypatch,
        profiles={"old": {"llm_api_key": "LEGACY"}},
        default="old",
    )

    result = _run(
        context,
        {
            "model_name": "old",
            "action": "edit",
            "field": "llm_endpoint",
            "value": "https://example.test",
        },
        approve=lambda *_: True,
    )

    assert result.data == "Updated LLM profile: old"
    profile = saved["llm_profiles"]["old"]
    assert profile["secret_llm_api_key"] == "LEGACY"
    assert "llm_api_key" not in profile
