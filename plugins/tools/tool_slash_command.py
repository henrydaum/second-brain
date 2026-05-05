"""Slash command tool — one-shot invocation of user slash commands by the agent.

The same `/agent`, `/configure`, `/schedule`, `/llm`, etc. that a human runs
through the UI's step-by-step form. The agent supplies the full structured
arguments in a single call; no form is shown.
"""

import logging

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("SlashCommand")

_BLOCKED = {"new", "cancel", "help", "refresh", "restart", "quit", "exit", "slash_command"}


class SlashCommand(BaseTool):
    name = "slash_command"
    description = (
        "Invoke a user slash command (the same ones a human runs in the UI) in "
        "one shot, supplying all arguments at once. Examples: change a setting "
        "via /configure, add an LLM via /llm add, edit an agent profile via "
        "/agent edit, schedule a job via /schedule create. Use /help (via the "
        "registry below) to discover available commands. UI-only commands "
        "(new, cancel, refresh, etc.) are blocked.\n\n"
        "Pass the command name without the leading slash. The args dict mirrors "
        "the fields the form would collect; see each command's form for the "
        "exact keys."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Command name without the leading slash. E.g. 'configure', 'agent', 'llm'.",
            },
            "args": {
                "type": "object",
                "description": (
                    "Structured arguments for the command. Keys mirror the form "
                    "fields the human UI would prompt for. Pass {} for arg-less "
                    "commands like 'services' or 'tools'."
                ),
            },
        },
        "required": ["name"],
    }
    requires_services = []
    max_calls = 20
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        name = (kwargs.get("name") or "").strip().lstrip("/")
        args = kwargs.get("args") or {}
        if not name:
            return ToolResult.failed("A command name is required.")
        if name in _BLOCKED:
            return ToolResult.failed(f"Command '/{name}' is not callable from an agent.")
        if not isinstance(args, dict):
            return ToolResult.failed("'args' must be an object.")

        runtime = getattr(context, "runtime", None) or getattr(getattr(context, "tool_registry", None), "runtime", None)
        registry = getattr(runtime, "command_registry", None) if runtime else None
        if registry is None:
            return ToolResult.failed("Command registry is not available in this context.")

        session_key = None
        sessions = getattr(runtime, "sessions", {}) or {}
        if sessions:
            session_key = next(iter(sessions))

        try:
            output = registry.dispatch_dict(name, args, session_key=session_key)
        except Exception as e:
            logger.exception(f"slash_command '{name}' failed")
            return ToolResult.failed(f"Command '/{name}' raised: {e}")

        text = "" if output is None else str(output)
        return ToolResult(data={"command": name, "output": text}, llm_summary=text or f"/{name} ran.")
