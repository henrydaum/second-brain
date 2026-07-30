"""Memory prompt section: folder + index model.

The kernel inlines only ``memory/MEMORY.md`` (the index) and lists topic
files by name — topic bodies stay out of the prompt and are read on demand
via the store ``memory`` tool, whose own ``agent_prompt`` carries the usage
instructions (plugin guidance stays out of the kernel).
"""

import pytest

import plugins.memory_paths as memory_paths
from agent.system_prompt import _agent_memory
from pipeline.database import DEFAULT_USER_ID


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_paths, "DATA_DIR", tmp_path)
    return tmp_path


def test_empty_install_shows_empty_index(data_dir):
    text = _agent_memory()
    assert "## Memory" in text
    assert "(empty)" in text


def test_index_inlined_topics_listed_not_inlined(data_dir):
    root = data_dir / "memory"
    root.mkdir()
    (root / "MEMORY.md").write_text("- [proj](proj.md) - the project", encoding="utf-8")
    (root / "proj.md").write_text("SECRET TOPIC BODY", encoding="utf-8")
    text = _agent_memory()
    assert "- [proj](proj.md) - the project" in text
    assert "proj" in text.split("Topic files:")[-1]
    assert "SECRET TOPIC BODY" not in text


def test_no_plugin_guidance_in_kernel_section(data_dir):
    (data_dir / "memory").mkdir()
    assert "`memory` tool" not in _agent_memory()


def test_memory_root_is_per_user_ready(data_dir):
    assert memory_paths.memory_root() == data_dir / "memory"
    assert memory_paths.memory_root(DEFAULT_USER_ID) == data_dir / "memory"
    assert memory_paths.memory_root(7) == data_dir / "memory" / "users" / "7"


def test_topic_path_validates_names(data_dir):
    (data_dir / "memory").mkdir()
    assert memory_paths.topic_path("project-x").name == "project-x.md"
    assert memory_paths.topic_path("notes.md").name == "notes.md"
    for bad in ("", "..", "../evil", "a/b", "MEMORY", ".hidden", "C:\\x"):
        with pytest.raises(ValueError):
            memory_paths.topic_path(bad)


def test_list_topics_excludes_index_and_other_users(data_dir):
    root = data_dir / "memory"
    (root / "users" / "7").mkdir(parents=True)
    (root / "MEMORY.md").write_text("idx", encoding="utf-8")
    (root / "a.md").write_text("a", encoding="utf-8")
    (root / "users" / "7" / "b.md").write_text("b", encoding="utf-8")
    assert [p.stem for p in memory_paths.list_topics()] == ["a"]
    assert [p.stem for p in memory_paths.list_topics(7)] == ["b"]


# ────────────────────────────────────────────────────────────────────
# Capability gating (was test_system_prompt_capabilities.py)
# ────────────────────────────────────────────────────────────────────

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
