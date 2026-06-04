"""Tests for the kernel LLM router and profile loader."""

from plugins.services.service_llm import (
    BaseLLM,
    LLMResponse,
    LLMRouter,
    _build_llm_from_profile,
    build_services,
    is_context_limit_error,
    refresh_llm_profile_services,
)


class FakeBackend(BaseLLM):
    is_llm_backend = True

    def __init__(self, model_name, api_key=None, base_url=None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.load_count = 0
        self.unload_count = 0

    def _load(self):
        self.load_count += 1
        self.loaded = True
        return True

    def unload(self):
        self.unload_count += 1
        self.loaded = False

    def invoke(self, messages, attachments=None, **kwargs):
        return LLMResponse(content=f"{self.model_name}:invoke")

    def stream(self, messages, attachments=None, **kwargs):
        yield f"{self.model_name}:stream"

    def chat_with_tools(self, messages, tools=None, **kwargs):
        return LLMResponse(content=f"{self.model_name}:tools")


def test_build_from_profile_uses_installed_backend_and_profile_fields(monkeypatch):
    monkeypatch.setattr("plugins.services.service_llm._llm_backend_classes", lambda: {"FakeBackend": FakeBackend})
    monkeypatch.setenv("FAKE_API_KEY", "sk-env")

    llm = _build_llm_from_profile("model-a", {
        "llm_service_class": "FakeBackend",
        "llm_api_key": "FAKE_API_KEY",
        "llm_endpoint": "https://example.test",
        "llm_context_size": 32000,
        "llm_capabilities": {"image": True, "audio": False, "other": True},
    })

    assert isinstance(llm, FakeBackend)
    assert llm.api_key == "sk-env"
    assert llm.base_url == "https://example.test"
    assert llm.context_size == 32000
    assert llm.capabilities == {"image": True, "audio": False, "video": None}


def test_build_services_registers_profiles_and_default_router(monkeypatch):
    monkeypatch.setattr("plugins.services.service_llm._llm_backend_classes", lambda: {"FakeBackend": FakeBackend})
    config = {"llm_profiles": {"model-a": {"llm_service_class": "FakeBackend"}}}

    services = build_services(config)

    assert set(services) == {"model-a", "llm"}
    assert config["default_llm_profile"] == "model-a"
    assert services["llm"].active is services["model-a"]


def test_router_loads_and_delegates_to_default_profile():
    config = {"llm_profiles": {"a": {}, "b": {}}, "default_llm_profile": "b"}
    services = {"a": FakeBackend("a"), "b": FakeBackend("b")}
    router = LLMRouter(config, services)
    services["llm"] = router

    assert router.load() is True
    assert services["b"].loaded
    assert not services["a"].loaded
    assert router.model_name == "b (b)"
    assert router.invoke([]).content == "b:invoke"
    assert "".join(router.stream([])) == "b:stream"
    assert router.chat_with_tools([], []).content == "b:tools"


def test_router_falls_back_to_first_registered_profile():
    config = {"llm_profiles": {"a": {}, "b": {}}, "default_llm_profile": "missing"}
    services = {"a": FakeBackend("a"), "b": FakeBackend("b")}
    router = LLMRouter(config, services)

    assert router.active is services["a"]


def test_router_reports_not_loaded_without_active_llm():
    router = LLMRouter({"llm_profiles": {}, "default_llm_profile": ""}, {})

    assert router.load() is False
    assert router.invoke([]).error_code == "not_loaded"
    assert router.chat_with_tools([], []).error_code == "not_loaded"
    assert list(router.stream([])) == []


def test_refresh_llm_profile_services_adds_and_removes_backend(monkeypatch):
    config = {"llm_profiles": {"model-x": {"llm_service_class": "FakeBackend"}}, "default_llm_profile": "model-x"}
    services = {}
    services["llm"] = LLMRouter(config, services)
    monkeypatch.setattr("plugins.services.service_llm._llm_backend_classes", lambda: {"FakeBackend": FakeBackend})

    assert refresh_llm_profile_services(services, config)
    assert isinstance(services["model-x"], FakeBackend)
    assert services["model-x"].loaded

    monkeypatch.setattr("plugins.services.service_llm._llm_backend_classes", lambda: {})
    assert refresh_llm_profile_services(services, config)
    assert "model-x" not in services
    assert services["llm"].active is None


def test_model_plan_error_is_not_context_limit():
    err = "your current token plan not support model, MiniMax-M2.7 (2061)"
    assert not is_context_limit_error(err)


def test_prompt_token_limit_is_context_limit():
    err = "prompt tokens exceed model token limit"
    assert is_context_limit_error(err)
