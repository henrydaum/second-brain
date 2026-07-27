"""Structured command catalogue and sandboxed ``/commands`` coverage."""

from types import SimpleNamespace

from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.command_registry import (
    CommandRegistry,
    frontend_command_filter,
)
from sandbox.facade import Sandbox
from state_machine.conversation import FormStep


class _Alpha(BaseCommand):
    name = "alpha"
    description = "Alpha command"
    category = "Conversation"

    def run(self, args, context):
        return None


class _Deploy(BaseCommand):
    name = "deploy"
    description = "Deploy a target"
    category = "Other"

    def form(self, args, context):
        return [
            FormStep("target", "Target", True),
            FormStep("force", "Force", False, type="boolean", default=False),
        ]

    def run(self, args, context):
        return None


class _Hidden(BaseCommand):
    name = "hidden"
    description = "Not listed"
    hide_from_help = True

    def run(self, args, context):
        return None


def _rig():
    config = {
        "frontend_profiles": {
            "telegram": {
                "whitelist_or_blacklist_commands": "blacklist",
                "commands_list": ["deploy"],
            }
        }
    }
    session = SimpleNamespace(frontend_name="telegram")
    runtime = SimpleNamespace(sessions={"chat": session})
    context = SimpleNamespace(
        config=config,
        runtime=runtime,
        session_key="chat",
    )
    registry = CommandRegistry(lambda _key: context)
    context.command_registry = registry
    for command in (_Alpha(), _Deploy(), _Hidden()):
        registry.register(command)
    return context, registry


def test_commands_command_matches_registry_help_and_frontend_filter():
    context, registry = _rig()
    expected = registry.help_text(
        frontend_command_filter(context.config, "telegram")
    )
    sandbox = Sandbox(context=context)
    try:
        result = sandbox.run(
            "plugins/commands/command_commands.py",
            "CommandsCommand",
            kwargs={"args": {}},
        )
    finally:
        sandbox.shutdown()

    assert result.ok, result.error
    assert result.data == expected
    assert "/alpha" in result.data
    assert "/deploy" not in result.data
    assert "/hidden" not in result.data


def test_commands_command_preserves_form_hints_without_filter():
    context, registry = _rig()
    context.runtime.sessions["chat"].frontend_name = "repl"
    expected = registry.help_text()

    sandbox = Sandbox(context=context)
    try:
        result = sandbox.run(
            "plugins/commands/command_commands.py",
            "CommandsCommand",
            kwargs={"args": {}},
        )
    finally:
        sandbox.shutdown()

    assert result.ok, result.error
    assert result.data == expected
    assert "/deploy <target> [force]" in result.data


def test_commands_command_keeps_missing_registry_message():
    sandbox = Sandbox(context=SimpleNamespace(command_registry=None))
    try:
        result = sandbox.run(
            "plugins/commands/command_commands.py",
            "CommandsCommand",
            kwargs={"args": {}},
        )
    finally:
        sandbox.shutdown()

    assert result.ok
    assert result.data == "No command registry is available."
