"""Memory prompt section: kernel-owned index only.

Topic paths, validation and enumeration belong to the store memory tool.
"""

import pytest

from agent.system_prompt import MEMORY_INDEX_CAP, _agent_memory


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    import paths
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    return tmp_path


def test_empty_install_shows_empty_index(data_dir):
    text = _agent_memory()
    assert "## Memory" in text
    assert "(empty)" in text


def test_index_inlined_without_reading_topic_files(data_dir):
    root = data_dir / "workspace" / "memory"
    root.mkdir(parents=True)
    (root / "MEMORY.md").write_text("- [proj](proj.md) - the project", encoding="utf-8")
    (root / "proj.md").write_text("SECRET TOPIC BODY", encoding="utf-8")
    text = _agent_memory()
    assert "- [proj](proj.md) - the project" in text
    assert "SECRET TOPIC BODY" not in text
    assert "Topic files:" not in text


def test_a_runaway_index_cannot_grow_the_prompt_without_bound(data_dir):
    """What the index says is the agent's business; what it costs is not.

    This text is inlined on every call, so an index nobody prunes would push
    the prompt over on its own. Truncation is visible so the agent can see
    that its own index has outgrown the window.
    """
    from agent.system_prompt import MEMORY_INDEX_CAP

    root = data_dir / "workspace" / "memory"
    root.mkdir(parents=True)
    (root / "MEMORY.md").write_text("- [x](x.md) - filler\n" * 2000,
                                    encoding="utf-8")

    text = _agent_memory()

    assert len(text) < MEMORY_INDEX_CAP + 500
    assert "prune MEMORY.md" in text


def test_the_cap_is_a_setting_so_a_curator_can_learn_it(data_dir):
    """The budget has two readers and neither may hold its own copy.

    Whatever curates ``MEMORY.md`` has to prune it to the same figure the
    kernel truncates at, and a sandboxed plugin cannot import this module to
    find out — so the number lives in config, where both sides can read it.
    A constant on each side is the drift this arrangement exists to prevent.
    """
    from config.config_data import SETTINGS_DATA

    declared = {entry[1]: entry[3] for entry in SETTINGS_DATA}
    assert declared["memory_index_cap"] == MEMORY_INDEX_CAP, (
        "the fallback constant and the declared default must agree")

    root = data_dir / "workspace" / "memory"
    root.mkdir(parents=True)
    (root / "MEMORY.md").write_text("- fact\n" * 500, encoding="utf-8")

    tight = _agent_memory({"memory_index_cap": 200})
    loose = _agent_memory({"memory_index_cap": 3000})

    assert "truncated at 200 characters" in tight
    assert "truncated at 3000 characters" in loose
    assert len(tight) < len(loose)


def test_an_index_under_the_cap_is_untouched(data_dir):
    root = data_dir / "workspace" / "memory"
    root.mkdir(parents=True)
    (root / "MEMORY.md").write_text("- [proj](proj.md) - the project",
                                    encoding="utf-8")

    text = _agent_memory()

    assert "- [proj](proj.md) - the project" in text
    assert "truncated" not in text


def test_no_plugin_guidance_in_kernel_section(data_dir):
    (data_dir / "workspace" / "memory").mkdir(parents=True)
    assert "`memory` tool" not in _agent_memory()


# ────────────────────────────────────────────────────────────────────
# Capability gating (was test_system_prompt_capabilities.py)
# ────────────────────────────────────────────────────────────────────

from types import SimpleNamespace
from agent.system_prompt import _model_status


def test_filesystem_access_distinguishes_agent_and_user_owned_paths(monkeypatch, tmp_path):
    from agent.system_prompt import _filesystem_access
    import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    user_project = tmp_path / "user-project"
    text = _filesystem_access({"fs_writable_dirs": [str(user_project)]})

    assert f"Agent-owned workspace (free write): {tmp_path / 'data' / 'workspace'}" in text
    assert "User-owned writable folders" in text
    assert str(user_project) in text
    assert "remain protected" in text


def test_filesystem_access_names_an_empty_grant(monkeypatch, tmp_path):
    from agent.system_prompt import _filesystem_access
    import paths

    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    assert "- None configured." in _filesystem_access({})


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


# ────────────────────────────────────────────────────────────────────
# Which block a plugin's guidance lands in
# ────────────────────────────────────────────────────────────────────


def _prompting_tools():
    """Two tools contributing the same guidance in the two allowed shapes."""
    fixed = SimpleNamespace(name="fixed", description="", parameters={},
                            agent_prompt="GUIDANCE-FROM-A-STRING")
    live = SimpleNamespace(name="live", description="", parameters={},
                           agent_prompt=lambda ctx: "GUIDANCE-FROM-A-METHOD")
    return fixed, live


def _sections_with(tools):
    """Build a prompt whose only in-scope plugins are these tools."""
    from agent.system_prompt import build_prompt_sections

    registry = SimpleNamespace(_visible_tools=lambda: list(tools), tools={})
    return build_prompt_sections(None, None, registry, {})


def test_a_fixed_contribution_stays_in_the_cacheable_prefix(data_dir):
    """A string is settled at load, so it belongs in the position-0 message.

    That message is the one providers cache across a conversation; text that
    cannot change has no reason to leave it.
    """
    fixed, _ = _prompting_tools()
    system, dynamic = _sections_with([fixed])

    assert "GUIDANCE-FROM-A-STRING" in system["content"]
    assert "GUIDANCE-FROM-A-STRING" not in dynamic["content"]


def test_a_live_contribution_rides_in_the_dynamic_block(data_dir):
    """A method exists because its answer moves — so it must not sit in the prefix.

    Left in the position-0 message, every refresh would rewrite the one thing
    the provider caches, and the fix for staleness would cost a cache miss on
    every subsequent call of the conversation. This is the same argument
    ``_mode_suffix`` makes for itself in ``runtime/runtime_config.py``.
    """
    _, live = _prompting_tools()
    system, dynamic = _sections_with([live])

    assert "GUIDANCE-FROM-A-METHOD" in dynamic["content"]
    assert "GUIDANCE-FROM-A-METHOD" not in system["content"]


def test_both_shapes_are_collected_when_both_are_present(data_dir):
    """The partition splits the populations; it must not drop half of them.

    Enumerating in-scope plugins once and collecting twice is the kind of
    refactor where one shape silently stops arriving — the exact failure mode
    ``_collect``'s tolerance of two shapes exists to prevent.
    """
    fixed, live = _prompting_tools()
    system, dynamic = _sections_with([fixed, live])

    assert "GUIDANCE-FROM-A-STRING" in system["content"]
    assert "GUIDANCE-FROM-A-METHOD" in dynamic["content"]


def test_the_static_prompt_stays_within_its_budget():
    """The static prompt is paid on every turn, including the ones that have
    nothing to do with this codebase.

    It had grown to 10 KB, most of it a plugin-authoring tutorial duplicating
    ``docs/SDK.md`` and a research procedure the model already knows — enough
    preamble that the agent remarked on it while answering a mundane question.
    The rule that keeps it down is *keep what is Second-Brain-specific and
    could not be inferred*: paths, grants, catalogs, kernel behaviour, the
    ``[SYSTEM CONTEXT UPDATE]`` structure. Long-form guidance belongs in
    ``docs/``, which the file itself points at and the agent can read on
    demand.

    A cap rather than a golden file: the wording should stay free to change,
    and only the budget is worth defending. Raise it deliberately, or not at
    all.
    """
    from agent.system_prompt import _STATIC_PROMPT_PATH

    size = len(_STATIC_PROMPT_PATH.read_text(encoding="utf-8"))

    assert size <= 7000, (
        f"agent/system_prompt_static.md is {size} chars. Anything long-form "
        "belongs in docs/ with a pointer here, not inlined into every turn."
    )
