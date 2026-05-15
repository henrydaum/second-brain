"""Regression tests for command copy."""

from types import SimpleNamespace
import json

from config import config_manager
from config.config_data import SETTINGS_DATA
from plugins.commands import command_config
from plugins.commands.command_agent import AgentCommand
from plugins.commands.command_config import ConfigCommand
from plugins.commands.command_llm import LlmCommand
from state_machine.form_display import form_step_display


def test_agent_command_prompts_explain_each_step():
    """Verify agent command prompts explain each step."""
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
    """Verify LLM command prompts explain add and edit steps."""
    context = SimpleNamespace(config={"llm_profiles": {"m": {}}, "default_llm_profile": "m"}, services={})

    add_steps = LlmCommand().form({"model_name": "add"}, context)
    edit_steps = LlmCommand().form({"model_name": "m", "action": "edit"}, context)

    assert add_steps[0].prompt.endswith("Default: m")
    assert add_steps[0].enum_labels == ["m (default)", "Add profile"]
    assert "model name exactly" in add_steps[1].prompt
    assert "API key" in add_steps[4].prompt
    assert "prompt cache routing key" in add_steps[6].prompt
    assert add_steps[7].enum == ["", "in_memory", "24h"]
    assert edit_steps[-2].prompt == "Choose which LLM setting to edit."
    assert edit_steps[-2].enum_labels == ["Endpoint", "API key", "Context size", "Service class", "Cache key", "Cache retention"]


def test_config_list_setting_uses_multiline_array_form():
    """Verify config list setting uses multiline array form."""
    context = SimpleNamespace(config={"autoload_services": []})
    step = ConfigCommand().form({"setting_name": "autoload_services", "action": "edit"}, context)[-1]
    display = form_step_display(step)

    assert step.type == "array"
    assert step.prompt == "Enter a list of items, one on each line, like so:\n\nitem 1\nitem 2"
    assert display["assist"] == ""


def test_config_list_setting_parses_one_item_per_line(monkeypatch):
    """Verify config list setting parses one item per line."""
    saved = []
    monkeypatch.setattr("plugins.commands.command_config.config_manager.save", lambda config: saved.append(dict(config)))

    context = SimpleNamespace(config={"autoload_services": []})
    result = ConfigCommand().run({"setting_name": "autoload_services", "action": "edit", "value": "llm\nparser"}, context)

    assert context.config["autoload_services"] == ["llm", "parser"]
    assert saved[-1]["autoload_services"] == ["llm", "parser"]
    assert result == "Set autoload_services = ['llm', 'parser']"


def test_config_load_normalizes_string_autoload_services(tmp_path):
    """Verify config load normalizes string autoload services."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(config_manager.DEFAULTS, autoload_services="web_search_provider")))

    config = config_manager.load(str(path))

    assert config["autoload_services"] == ["web_search_provider"]
    assert json.loads(path.read_text())["autoload_services"] == ["web_search_provider"]


def test_config_save_partial_update_preserves_existing_values(tmp_path):
    """Verify partial config saves do not clobber existing config."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"enabled_frontends": ["repl"], "tool_timeout": 123}))

    config_manager.save({"last_active_conversation_id": 9}, str(path))
    saved = json.loads(path.read_text())

    assert saved["enabled_frontends"] == ["repl"]
    assert saved["tool_timeout"] == 123
    assert saved["last_active_conversation_id"] == 9


def test_config_frontend_plugin_setting_saves_with_restart_notice(monkeypatch):
    """Verify config frontend plugin setting saves with restart notice."""
    setting = ("Telegram Allowed User ID", "telegram_allowed_user_id", "Only this user can interact with the bot.", 0, {"type": "text"})
    saved_core, saved_plugin = [], []
    monkeypatch.setattr(command_config, "get_plugin_settings", lambda: [setting])
    monkeypatch.setattr(command_config, "get_plugin_setting_type", lambda key: "frontend" if key == "telegram_allowed_user_id" else None)
    monkeypatch.setattr(command_config.config_manager, "save", lambda config: saved_core.append(dict(config)))
    monkeypatch.setattr(command_config.config_manager, "load_plugin_config", lambda: {})
    monkeypatch.setattr(command_config.config_manager, "save_plugin_config", lambda config: saved_plugin.append(dict(config)))

    context = SimpleNamespace(config={"telegram_allowed_user_id": 0})
    result = ConfigCommand().run({"setting_name": "telegram_allowed_user_id", "action": "edit", "value": "123"}, context)

    assert context.config["telegram_allowed_user_id"] == "123"
    assert saved_core[-1]["telegram_allowed_user_id"] == "123"
    assert saved_plugin[-1] == {"telegram_allowed_user_id": "123"}
    assert result == "Set telegram_allowed_user_id = 123. Restart required."


def test_telegram_credentials_are_not_core_settings():
    """Verify Telegram credentials are not core settings."""
    core_keys = {entry[1] for entry in SETTINGS_DATA}
    assert "telegram_bot_token" not in core_keys
    assert "telegram_allowed_user_id" not in core_keys
