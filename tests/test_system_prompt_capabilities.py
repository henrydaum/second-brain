from types import SimpleNamespace

from agent.system_prompt import _model_status


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

def test_model_status_prefers_session_resolved_llm_over_router():
    # A profile pinning a non-default LLM must be described as itself: the
    # router (default profile) is only the fallback when no caller context.
    pinned = SimpleNamespace(
        model_name="minimax/MiniMax-M3",
        capabilities={"image": True},
        native_attachment_modalities={"image"},
    )
    router = SimpleNamespace(
        _active_name="deepseek/deepseek-chat",
        active=SimpleNamespace(model_name="deepseek/deepseek-chat",
                               capabilities={}, native_attachment_modalities=set()),
    )
    status = _model_status({"llm": router}, pinned)
    assert "minimax/MiniMax-M3" in status and "deepseek" not in status
    assert "images: yes" in status


def test_session_prompt_names_the_profile_pinned_llm(tmp_path):
    # End to end: a session whose profile pins a non-default LLM gets that
    # model in its prompt's model-status line, not the router default.
    import state_machine  # noqa: F401 — break the runtime<->state_machine import cycle
    from runtime.conversation_runtime import ConversationRuntime
    from runtime.runtime_config import session_system_prompt

    pinned = SimpleNamespace(model_name="minimax/MiniMax-M3", loaded=True,
                             capabilities={}, native_attachment_modalities=set())
    router = SimpleNamespace(_active_name="deepseek/deepseek-chat",
                             active=SimpleNamespace(model_name="deepseek/deepseek-chat",
                                                    capabilities={},
                                                    native_attachment_modalities=set()))
    from pipeline.database import Database
    db = Database(str(tmp_path / "prompt.db"))
    rt = ConversationRuntime(
        db=db,
        services={"llm": router, "minimax/MiniMax-M3": pinned},
        config={"agent_profiles": {"research": {"llm": "minimax/MiniMax-M3"}},
                "llm_profiles": {"minimax/MiniMax-M3": {}},
                "default_llm_profile": "deepseek/deepseek-chat"},
    )
    session = rt.load_conversation("s", db.create_conversation("x"))
    session.profile_override = "research"
    prompt = session_system_prompt(rt, session)()
    dynamic = prompt[1]["content"]
    assert "minimax/MiniMax-M3" in dynamic.split("Current model:")[1].splitlines()[0]
