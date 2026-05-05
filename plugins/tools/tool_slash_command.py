"""Slash command tool — one-shot invocation of registered command plugins."""

import json
import logging

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("SlashCommand")

_BLOCKED = {"restart", "quit", "slash_command"}


class SlashCommand(BaseTool):
    name = "slash_command"
    description = (
        "Invoke a registered user slash-command plugin in one shot, supplying "
        "all arguments at once. Commands are discovered from BaseCommand "
        "plugins. Host commands such as quit and restart are blocked.\n\n"
        "Pass the command name without the leading slash. The args dict mirrors "
        "the fields the form would collect; see each command's form for the "
        "exact keys."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Command name without the leading slash.",
            },
            "args": {
                "type": "object",
                "description": (
                    "Structured arguments for the command. Keys mirror the form "
                    "fields the human UI would prompt for. Pass {} for arg-less commands."
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
        if getattr((getattr(registry, "_commands", {}) or {}).get(name), "require_approval", False):
            return ToolResult.failed(f"Command '/{name}' requires user approval and is not callable from an agent.")

        approve_fn = getattr(context, "approve_command", None)
        if approve_fn is None:
            return ToolResult.failed(
                "Slash command execution is not available — no approval handler is configured."
            )
        detail = _format_args(args)
        try:
            approved = approve_fn(f"Run /{name}", detail)
        except Exception as e:
            logger.error(f"Approval callback failed: {e}")
            return ToolResult.failed(f"Approval dialog error: {e}")
        if not approved:
            return ToolResult.failed(
                f"User denied /{name}. STOP — do not retry. Ask the user what to do instead."
            )

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


def _format_args(args: dict) -> str:
    if not args:
        return "(no arguments)"
    try:
        return json.dumps(args, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(args)
