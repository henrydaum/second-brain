from types import SimpleNamespace

from agent.system_prompt import _loaded_services_for_prompt, _model_status


def test_unloaded_services_are_prompt_silent_unless_opted_in():
    silent = SimpleNamespace(loaded=False)
    speaking = SimpleNamespace(loaded=False, prompt_when_unloaded=True)
    loaded = SimpleNamespace(loaded=True)

    picked = _loaded_services_for_prompt({"a": silent, "b": speaking, "c": loaded})

    assert picked == [speaking, loaded]


def test_model_status_reports_effective_native_attachment_capabilities():
    active = SimpleNamespace(
        model_name="MiniMax-M3",
        capabilities={"image": True, "audio": True, "video": False},
        native_attachment_modalities={"image", "video"},
    )
    router = SimpleNamespace(_active_name="m3", active=active)

    status = _model_status({"llm": router})

    assert "Current model: m3 (MiniMax-M3)." in status
    assert "images: yes" in status
    assert "audio: no" in status
    assert "video: no" in status


def test_model_status_reports_unavailable_without_llm():
    assert _model_status({}) == "Current model: unavailable."
