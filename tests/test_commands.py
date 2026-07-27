"""Remaining native `/agent` command coverage."""

from types import SimpleNamespace

from plugins.commands.command_agent import AgentCommand


def test_agent_command_can_rename_profile(monkeypatch):
    saved = []
    monkeypatch.setattr(
        "plugins.commands.command_agent._save",
        lambda config: saved.append(dict(config)),
    )
    session = SimpleNamespace(
        active_agent_profile="builder", profile_override="builder")
    runtime = SimpleNamespace(
        sessions={"chat": session}, refresh_session_specs=lambda: None)
    context = SimpleNamespace(
        config={
            "agent_profiles": {"builder": {"llm": "default"}},
            "active_agent_profile": "builder",
        },
        runtime=runtime,
    )

    steps = AgentCommand().form(
        {"profile_name": "builder", "action": "edit"}, context)
    result = AgentCommand().run({
        "profile_name": "builder",
        "action": "edit",
        "field": "agent_profile_name",
        "value": "writer",
    }, context)

    assert "agent_profile_name" in next(
        step.enum for step in steps if step.name == "field")
    assert result == "Updated agent profile: writer"
    assert "builder" not in context.config["agent_profiles"]
    assert context.config["active_agent_profile"] == "writer"
    assert session.active_agent_profile == "writer"
    assert session.profile_override == "writer"
    assert saved[-1]["active_agent_profile"] == "writer"


def test_agent_command_contributes_prompt_guidance():
    assert "profile" in AgentCommand().agent_prompt_for(None)
