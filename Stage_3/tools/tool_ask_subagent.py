"""
Ask Subagent tool.

Lets the main agent offload a task to a fresh conversation with its own
context window. The subagent has access to all tools except ask_subagent
(no infinite regress) and the approval-gated tools (build_plugin, run_command).
"""

import logging

from Stage_3.BaseTool import BaseTool, ToolResult
from Stage_3.subagent import run_subagent

logger = logging.getLogger("AskSubagent")


class AskSubagent(BaseTool):
    name = "ask_subagent"
    description = (
        "Spawn a subagent in a fresh conversation to handle a self-contained task, "
        "then return its final text answer. Use this to offload context-heavy work "
        "(triaging many files, multi-step research) without filling your own context. "
        "The subagent has access to the same tools you do, minus ask_subagent, "
        "build_plugin, and run_command. Give it a complete, standalone prompt — "
        "it cannot see your conversation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The complete task description for the subagent. Must be self-contained.",
            },
            "title": {
                "type": "string",
                "description": "Short title for the subagent's conversation (for your own bookkeeping). Optional.",
            },
        },
        "required": ["prompt"],
    }
    requires_services = ["llm"]
    agent_enabled = True
    max_calls = 5

    def run(self, context, **kwargs) -> ToolResult:
        prompt = (kwargs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult.failed("'prompt' is required.")

        title = (kwargs.get("title") or "").strip() or "[subagent]"

        try:
            result, conv_id = run_subagent(
                prompt=prompt,
                title=title,
                db=context.db,
                config=context.config,
                services=context.services,
                tool_registry=context.tool_registry,
                orchestrator=context.orchestrator,
            )
        except Exception as e:
            logger.error(f"Subagent failed: {e}", exc_info=True)
            return ToolResult.failed(f"Subagent failed: {e}")

        return ToolResult(
            llm_summary=result or "(subagent returned no text)",
            data={"conversation_id": conv_id},
        )

