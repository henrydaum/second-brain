"""Tests for the kernel slash commands.

Only the REPL/introspection commands ship in the kernel. These exercise the
two with non-trivial logic: ``/llm`` (profile management form + handler) and
``/debug`` (live state-machine snapshot + recent log tail). Both stub their
context dependencies.
"""

from types import SimpleNamespace

from plugins.commands import command_debug
from plugins.commands.command_frontends import FrontendsCommand
from plugins.commands.command_agent import AgentCommand
from plugins.commands.command_debug import DebugCommand
from plugins.commands.command_llm import LlmCommand
from state_machine.conversation import ConversationState, Participant


# ── /llm ─────────────────────────────────────────────────────────────

def test_llm_command_can_set_default(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda context: saved.append(dict(context.config)))
    context = SimpleNamespace(config={"llm_profiles": {"a": {}, "b": {}}, "default_llm_profile": "a"}, services={})

    steps = LlmCommand().form({"model_name": "b"}, context)
    result = LlmCommand().run({"model_name": "b", "action": "set_default"}, context)

    assert steps[0].prompt == "Select an LLM profile, or add a new one.\nDefault: a"
    assert steps[1].enum == ["edit", "set_default", "remove"]
    assert steps[1].enum_labels == ["Edit", "Set default", "Remove"]
    assert result == "Default LLM profile set to: b"
    assert context.config["default_llm_profile"] == "b"
    assert saved[-1]["default_llm_profile"] == "b"


def test_llm_command_set_default_writes_through_to_runtime_config(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm.config_manager.load_plugin_config", lambda: {"kept": True})
    monkeypatch.setattr("plugins.commands.command_llm.config_manager.save_plugin_config", lambda values: saved.append(dict(values)))
    runtime = SimpleNamespace(config={"llm_profiles": {"a": {}, "b": {}}, "default_llm_profile": ""})
    context = SimpleNamespace(config={"llm_profiles": {"a": {}, "b": {}}, "default_llm_profile": ""}, services={}, runtime=runtime)

    result = LlmCommand().run({"model_name": "b", "action": "set_default"}, context)

    assert result == "Default LLM profile set to: b"
    assert saved[-1]["kept"] is True
    assert saved[-1]["default_llm_profile"] == "b"
    assert runtime.config["default_llm_profile"] == "b"


def test_llm_command_add_stores_declared_capabilities(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda context: saved.append(dict(context.config)))
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
    assert context.config["default_llm_profile"] == "openai/gpt-4o"
    assert profile["llm_capabilities"] == {"image": True, "audio": False}
    assert not any(k.startswith("llm_capability_") for k in profile)
    assert saved[-1]["llm_profiles"]["openai/gpt-4o"] == profile
    assert saved[-1]["default_llm_profile"] == "openai/gpt-4o"

def test_llm_command_can_rename_profile(monkeypatch):
    saved, removed, added = [], [], []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda context: saved.append(dict(context.config)))
    router = SimpleNamespace(remove_llm=lambda name: removed.append(name), add_llm=lambda name, profile: added.append((name, profile)))
    context = SimpleNamespace(config={"llm_profiles": {"bad": {"llm_endpoint": "https://api.atlascloud.ai/v1"}}, "default_llm_profile": "bad"}, services={"llm": router})

    steps = LlmCommand().form({"model_name": "bad", "action": "edit"}, context)
    result = LlmCommand().run({"model_name": "bad", "action": "edit", "field": "llm_model_name", "value": "deepseek-ai/deepseek-v4-pro"}, context)

    assert "llm_model_name" in next(s.enum for s in steps if s.name == "field")
    assert result == "Updated LLM profile: deepseek-ai/deepseek-v4-pro"
    assert "bad" not in context.config["llm_profiles"]
    assert context.config["default_llm_profile"] == "deepseek-ai/deepseek-v4-pro"
    assert removed == ["bad"]
    assert added[-1][0] == "deepseek-ai/deepseek-v4-pro"
    assert saved[-1]["default_llm_profile"] == "deepseek-ai/deepseek-v4-pro"


def test_llm_command_remove_default_selects_next_profile(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda context: saved.append(dict(context.config)))
    context = SimpleNamespace(config={"llm_profiles": {"a": {}, "b": {}, "c": {}}, "default_llm_profile": "b"}, services={})

    result = LlmCommand().run({"model_name": "b", "action": "remove"}, context)

    assert result == "Removed LLM profile: b"
    assert context.config["default_llm_profile"] == "c"
    assert saved[-1]["default_llm_profile"] == "c"


def test_llm_command_add_does_not_replace_existing_default(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda context: saved.append(dict(context.config)))
    context = SimpleNamespace(config={"llm_profiles": {"a": {}}, "default_llm_profile": "a"}, services={})

    result = LlmCommand().run({"model_name": "add", "new_model_name": "b"}, context)

    assert result == "Added LLM profile: b"
    assert context.config["default_llm_profile"] == "a"
    assert saved[-1]["default_llm_profile"] == "a"


def test_llm_command_remove_last_default_blanks_default(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda context: saved.append(dict(context.config)))
    context = SimpleNamespace(config={"llm_profiles": {"a": {}}, "default_llm_profile": "a"}, services={})

    result = LlmCommand().run({"model_name": "a", "action": "remove"}, context)

    assert result == "Removed LLM profile: a"
    assert context.config["default_llm_profile"] == ""
    assert saved[-1]["default_llm_profile"] == ""


# ── /agent ───────────────────────────────────────────────────────────

def test_agent_command_can_rename_profile(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_agent._save", lambda config: saved.append(dict(config)))
    session = SimpleNamespace(active_agent_profile="builder", profile_override="builder")
    runtime = SimpleNamespace(sessions={"chat": session}, refresh_session_specs=lambda: None)
    context = SimpleNamespace(config={"agent_profiles": {"builder": {"llm": "default"}}, "active_agent_profile": "builder"}, runtime=runtime)

    steps = AgentCommand().form({"profile_name": "builder", "action": "edit"}, context)
    result = AgentCommand().run({"profile_name": "builder", "action": "edit", "field": "agent_profile_name", "value": "writer"}, context)

    assert "agent_profile_name" in next(s.enum for s in steps if s.name == "field")
    assert result == "Updated agent profile: writer"
    assert "builder" not in context.config["agent_profiles"]
    assert context.config["active_agent_profile"] == "writer"
    assert session.active_agent_profile == "writer"
    assert session.profile_override == "writer"
    assert saved[-1]["active_agent_profile"] == "writer"


# ── /frontends ───────────────────────────────────────────────────────

def test_frontends_form_uses_runtime_cache_without_discovery(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("frontend discovery should not run while rendering form hints")

    monkeypatch.setattr("plugins.plugin_discovery.discover_frontends", boom)
    manager = SimpleNamespace(available_frontends={"repl", "telegram"}, adapters={"repl": object()})
    runtime = SimpleNamespace(frontend_manager=manager)
    context = SimpleNamespace(config={"enabled_frontends": ["repl"], "frontend_profiles": {}}, runtime=runtime)

    steps = FrontendsCommand().form({}, context)

    assert steps[0].enum == ["repl", "telegram"]


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
    return SimpleNamespace(runtime=SimpleNamespace(sessions={"chat": session}), session_key="chat", services={})


def test_debug_reports_state_machine_snapshot_and_log_tail(tmp_path, monkeypatch):
    cs = ConversationState([Participant("user", "user"), Participant("agent", "agent")])
    service = SimpleNamespace(debug_flags=lambda _session: ["sample extension"])
    context = _session_context(tmp_path, monkeypatch, cs)
    context.services["sample"] = service

    out = DebugCommand().run({}, context)

    assert "Conversation state:" in out
    assert "Turn: user (user)" in out
    assert f"Phase: {cs.phase}" in out
    assert "Participants: user(user), agent(agent)" in out
    assert "Session: sample extension" in out
    # The valuable part of the old /doctor: recent warnings/errors.
    assert "Recent log warnings/errors:" in out
    assert "Plugin registration failed: demo" in out
    assert "Auto-load failed for 'llm': boom" in out
    assert "INFO  | ok" not in out  # info lines are filtered out


def test_debug_handles_no_active_session(tmp_path, monkeypatch):
    monkeypatch.setattr(command_debug, "DATA_DIR", tmp_path)
    context = SimpleNamespace(runtime=SimpleNamespace(sessions={}), session_key="chat", services={})

    out = DebugCommand().run({}, context)

    assert "(no active session)" in out
    assert "No log file found" in out
