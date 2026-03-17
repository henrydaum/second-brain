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
import time
from dataclasses import dataclass, field
from typing import Any

from context import DataRefineryContext, build_context

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

    def run(self, context: DataRefineryContext, **kwargs) -> ToolResult:
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


class ToolRegistry:
    """
    Manages all registered tools and handles dispatch.

    The registry:
        1. Stores tool instances by name
        2. Dispatches calls (including tool-to-tool calls via context.call_tool)
        3. Exports schemas for LLM function calling
    """

    def __init__(self, db, config: dict, services: dict = None):
        self.db = db
        self.config = config
        self.services = services or {}
        self.tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool. Overwrites if name already exists."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def call(self, name: str, **kwargs) -> ToolResult:
        """
        Call a tool by name. This is the single dispatch point.

        Used by:
            - External callers (API, CLI, LLM)
            - Other tools via context.call_tool
        """
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult.failed(f"Unknown tool: {name}")

        # Check service requirements
        if tool.requires_services:
            not_ready = []
            for svc_name in tool.requires_services:
                svc = self.services.get(svc_name)
                if svc is None or not svc.loaded:
                    not_ready.append(svc_name)
            if not_ready:
                return ToolResult.failed(f"Required services not available: {not_ready}")

        # Build context with call_tool pointing back to this registry
        context = build_context(self.db, self.config, self.services, call_tool=self.call)

        t0 = time.time()
        try:
            result = tool.run(context, **kwargs)
            logger.debug(f"Tool '{name}' completed in {time.time() - t0:.3f}s")
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' failed after {time.time() - t0:.3f}s: {e}")
            return ToolResult.failed(str(e))

    @property
    def max_tool_calls(self) -> int:
        """Total tool call budget for the agent (sum of per-tool max_calls)."""
        return sum(t.max_calls for t in self.tools.values() if t.agent_enabled)

    def get_all_schemas(self) -> list[dict]:
        """Export tool schemas for LLM function calling (agent_enabled tools only)."""
        return [tool.to_schema() for tool in self.tools.values() if tool.agent_enabled]

    def get_schema(self, name: str) -> dict | None:
        tool = self.tools.get(name)
        return tool.to_schema() if tool else None

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())