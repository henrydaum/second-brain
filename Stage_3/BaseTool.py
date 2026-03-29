"""
Tool interface.

Tools are the query layer — they accept arguments, read from the database
and parsers, call other tools, and return structured results.

Unlike tasks (which run in the background on every file), tools are
called on demand and return results immediately to a caller.

Tools become LLM function calls with zero translation:
    - name        -> function name
    - description -> function description
    - parameters  -> JSON schema for arguments

The same tool can be exposed via REST API, CLI, WebSocket, or LLM tool call.
The interface is the same — only the transport layer changes.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Tool")


@dataclass
class ToolResult:
    """
    What a tool returns.

    success:     Did it work?
    error:       Error message if failed.
    data:        Rich result payload for GUI display (tables, lists, etc.).
                 Never sent to Claude directly.
    llm_summary: Text sent to the LLM. All tools must populate this.
                 Standardized human-readable format regardless of tool type.
    gui_display_paths:       Flat list of file paths for GUI file-preview rendering in Flet. Never sent to the LLM, unless the file path is an image, since the LLM can receive those.
    """
    success: bool = True
    error: str = ""
    data: Any = None
    llm_summary: str = ""  # What the LLM will see as the tool result
    gui_display_paths: list[str] = field(default_factory=list)  # What the GUI will render as file previews

    def to_dict(self) -> dict:
        """Serialize for HTTP API responses."""
        return {
            "success": self.success,
            "error": self.error,
            "data": self.data,
            "llm_summary": self.llm_summary,
        }

    @staticmethod
    def failed(error: str) -> "ToolResult":
        return ToolResult(success=False, error=error)


class BaseTool:
    """
    The contract every tool implements.

    Class attributes (override these):
        name              Unique identifier. "keyword_search", "vector_search", etc.
        description       Natural language description. Doubles as the LLM tool description.
        parameters        JSON schema dict describing the input arguments.
        requires_services List of services that must be loaded. Same as tasks.

    Methods (override these):
        run(context, **kwargs) -> ToolResult
    """

    # --- Identity ---
    name: str = ""
    description: str = ""
    parameters: dict = {}

    # --- Service requirements ---
    requires_services: list[str] = []

    # --- Agent controls ---
    agent_enabled: bool = True   # Whether the LLM can see and call this tool
    max_calls: int = 3           # Max times the agent can call this tool per message

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr in ("parameters", "requires_services"):
            value = getattr(cls, attr)
            if isinstance(value, (dict, list)):
                setattr(cls, attr, value.copy())

    def run(self, context, **kwargs) -> ToolResult:
        raise NotImplementedError(f"Tool '{self.name}' must implement run()")

    def to_schema(self) -> dict:
        """Export as an OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


