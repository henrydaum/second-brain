from types import SimpleNamespace

from plugins.commands.command_agent import AgentCommand
from plugins.commands.command_llm import LlmCommand


def test_agent_command_prompts_explain_each_step():
    context = SimpleNamespace(
        config={"agent_profiles": {"default": {"llm": "default"}}, "llm_profiles": {"m": {}}},
        tool_registry=SimpleNamespace(tools={"search": object()}),
    )

    add_steps = AgentCommand().form({"profile_name": "add"}, context)
    edit_steps = AgentCommand().form({"profile_name": "default", "action": "edit"}, context)

    assert add_steps[0].prompt == "Select an agent profile, or add a new one."
    assert add_steps[0].enum_labels == ["default", "Add profile"]
    assert "short name" in add_steps[1].prompt
    assert "system prompt" in add_steps[3].prompt
    assert edit_steps[-2].prompt == "Choose which part of the agent profile to edit."
    assert edit_steps[-2].enum_labels == ["LLM", "Prompt suffix", "Tool mode", "Tool list"]


def test_llm_command_prompts_explain_add_and_edit_steps():
    context = SimpleNamespace(config={"llm_profiles": {"m": {}}, "default_llm_profile": "m"}, services={})

    add_steps = LlmCommand().form({"model_name": "add"}, context)
    edit_steps = LlmCommand().form({"model_name": "m", "action": "edit"}, context)

    assert add_steps[0].prompt.endswith("Default: m")
    assert add_steps[0].enum_labels == ["m (default)", "Add profile"]
    assert "model name exactly" in add_steps[1].prompt
    assert "API key" in add_steps[4].prompt
    assert edit_steps[-2].prompt == "Choose which LLM setting to edit."
    assert edit_steps[-2].enum_labels == ["Endpoint", "API key", "Context size", "Service class"]
