"""
Run Command tool.

Gives the LLM agent general-purpose terminal access with mandatory
user approval. Every command requires explicit Allow/Deny from the user
via a confirmation dialog (GUI) or console prompt (REPL).

Replaces the old read_source_code and install_package tools.
"""

import logging
import subprocess

from Stage_3.BaseTool import BaseTool, ToolResult
from paths import ROOT_DIR

logger = logging.getLogger("RunCommand")


class RunCommand(BaseTool):
    name = "run_command"
    description = (
        "Run a terminal command. Requires user approval before execution. "
        "Use this to read source files (e.g. cat, type), search code (e.g. grep, findstr), "
        "install packages for plugins (e.g. pip install), check environment state, and more. "
        "Always provide an honest justification so the user can make an informed decision."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "justification": {
                "type": "string",
                "description": "Plain English explanation of what this command does and why.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Max seconds to wait for the command to finish. Defaults to 30. "
                    "Use higher values for slow operations like 'pip install torch' (300+). Max 600."
                ),
            },
        },
        "required": ["command", "justification"],
    }
    requires_services = []
    agent_enabled = True
    max_calls = 10

    def run(self, context, **kwargs) -> ToolResult:
        command = kwargs.get("command", "").strip()
        justification = kwargs.get("justification", "").strip()
        timeout = min(max(int(kwargs.get("timeout", 30)), 5), 600)

        if not command:
            return ToolResult.failed("No command provided.")
        if not justification:
            return ToolResult.failed("A justification is required for every command.")

        # Request user approval
        approve_fn = context.approve_command
        if approve_fn is None:
            return ToolResult.failed(
                "Command execution is not available — no approval handler is configured."
            )

        try:
            approved = approve_fn(command, justification)
        except Exception as e:
            logger.error(f"Approval callback failed: {e}")
            return ToolResult.failed(f"Approval dialog error: {e}")

        if not approved:
            return ToolResult.failed("Command denied by user.")

        # Execute the command
        logger.info(f"Running: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(ROOT_DIR),
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failed(f"Command timed out after {timeout} seconds.")
        except Exception as e:
            return ToolResult.failed(f"Command execution error: {e}")

        # Build summary
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            parts.append(f"(exit code {result.returncode})")

        output = "\n".join(parts) if parts else "(no output)"

        return ToolResult(
            data={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode},
            llm_summary=output,
        )
