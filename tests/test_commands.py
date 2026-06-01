"""Tests for the kernel slash commands.

Only the REPL/introspection commands ship in the kernel. These exercise the
two with non-trivial logic: ``/llm`` (profile management form + handler) and
``/debug`` (live state-machine snapshot + recent log tail). Both stub their
context dependencies.
"""

from types import SimpleNamespace

from plugins.commands import command_debug
from plugins.commands.command_debug import DebugCommand
from plugins.commands.command_llm import LlmCommand
from state_machine.conversation import ConversationState, Participant


# ── /llm ─────────────────────────────────────────────────────────────

def test_llm_command_can_set_default(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda config: saved.append(dict(config)))
    context = SimpleNamespace(config={"llm_profiles": {"a": {}, "b": {}}, "default_llm_profile": "a"}, services={})

    steps = LlmCommand().form({"model_name": "b"}, context)
    result = LlmCommand().run({"model_name": "b", "action": "set_default"}, context)

    assert steps[0].prompt == "Select an LLM profile, or add a new one.\nDefault: a"
    assert steps[1].enum == ["edit", "set_default", "remove"]
    assert steps[1].enum_labels == ["Edit", "Set default", "Remove"]
    assert result == "Default LLM profile set to: b"
    assert context.config["default_llm_profile"] == "b"
    assert saved[-1]["default_llm_profile"] == "b"


def test_llm_command_add_stores_declared_capabilities(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda config: saved.append(dict(config)))
    context = SimpleNamespace(config={"llm_profiles": {}, "default_llm_profile": ""}, services={})

    steps = LlmCommand().form({"model_name": "add"}, context)
    result = LlmCommand().run({
        "model_name": "add",
        "new_model_name": "openai/gpt-4o",
        "llm_service_class": "LiteLLMService",
        "llm_endpoint": "",
        "llm_api_key": "OPENAI_API_KEY",
        "llm_context_size": 0,
        "llm_capability_image": True,
        "llm_capability_audio": False,
    }, context)

    profile = context.config["llm_profiles"]["openai/gpt-4o"]
    assert [s.name for s in steps][-3:] == ["llm_capability_image", "llm_capability_audio", "llm_capability_video"]
    assert result == "Added LLM profile: openai/gpt-4o"
    assert profile["llm_capabilities"] == {"image": True, "audio": False}
    assert saved[-1]["llm_profiles"]["openai/gpt-4o"] == profile


# ── /debug ───────────────────────────────────────────────────────────

def _session_context(tmp_path, monkeypatch, cs, **session_attrs):
    monkeypatch.setattr(command_debug, "DATA_DIR", tmp_path)
    (tmp_path / "app.log").write_text(
        "01:00PM | Main         | INFO  | ok\n"
        "01:01PM | Discovery    | WARNING | Plugin registration failed: demo\n"
        "01:02PM | Main         | ERROR | Auto-load failed for 'llm': boom\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(cs=cs, **session_attrs)
    return SimpleNamespace(runtime=SimpleNamespace(sessions={"chat": session}), session_key="chat")


def test_debug_reports_state_machine_snapshot_and_log_tail(tmp_path, monkeypatch):
    cs = ConversationState([Participant("user", "user"), Participant("agent", "agent")])
    context = _session_context(tmp_path, monkeypatch, cs, plan_mode=True, full_permissions_this_turn=False)

    out = DebugCommand().run({}, context)

    assert "Conversation state:" in out
    assert "Turn: user (user)" in out
    assert f"Phase: {cs.phase}" in out
    assert "Participants: user(user), agent(agent)" in out
    assert "Session: plan mode" in out
    # The valuable part of the old /doctor: recent warnings/errors.
    assert "Recent log warnings/errors:" in out
    assert "Plugin registration failed: demo" in out
    assert "Auto-load failed for 'llm': boom" in out
    assert "INFO  | ok" not in out  # info lines are filtered out


def test_debug_handles_no_active_session(tmp_path, monkeypatch):
    monkeypatch.setattr(command_debug, "DATA_DIR", tmp_path)
    context = SimpleNamespace(runtime=SimpleNamespace(sessions={}), session_key="chat")

    out = DebugCommand().run({}, context)

    assert "(no active session)" in out
    assert "No log file found" in out
