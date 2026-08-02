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
    # A modality counts only when both halves agree: the model ingests it
    # (capabilities) and the backend can put it on the wire (native_modalities).
    brain = SimpleNamespace(
        model_name="MiniMax-M3",
        capabilities={"image": True, "audio": True, "video": False},
        native_modalities={"image", "video"},
    )

    status = _model_status(brain)

    assert "Current model: MiniMax-M3." in status
    assert "images: yes" in status
    assert "audio: no" in status      # model reads it, backend cannot send it
    assert "video: no" in status      # backend sends it, model cannot read it


def test_model_status_reports_unavailable_without_llm():
    assert _model_status(None) == "Current model: unavailable."


def test_model_status_reads_the_attributes_brain_actually_publishes():
    """The names in the prompt must be the names routing uses.

    This asked a ``Brain`` for ``native_attachment_modalities``, which no Brain
    has ever had. ``getattr`` answered its default, so every model was told it
    was blind while ``_route_attachments`` sent it images perfectly well. No
    ``SimpleNamespace`` fake can catch that — a fake has whatever attribute the
    test gives it — so pin the real class instead.
    """
    from llm.registry import Brain

    for attr in ("model_name", "capabilities", "native_modalities"):
        assert isinstance(getattr(Brain, attr, None), property), attr


def test_session_prompt_names_the_profile_pinned_llm(tmp_path):
    # End to end: a session whose profile pins a non-default LLM gets that
    # model in its prompt's model-status line, not the router default.
    import state_machine  # noqa: F401 — break the runtime<->state_machine import cycle
    from runtime.conversation_runtime import ConversationRuntime
    from runtime.runtime_config import session_system_prompt

    pinned = SimpleNamespace(model_name="minimax/MiniMax-M3", loaded=True,
                             capabilities={}, native_modalities=set())
    router = SimpleNamespace(model_name="deepseek/deepseek-chat", loaded=True,
                             capabilities={}, native_modalities=set())
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
