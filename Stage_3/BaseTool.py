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

from context import DataRefineryContext

logger = logging.getLogger("Tool")


@dataclass
class ToolResult:
    """
    What a tool returns.

    success:  Did it work?
    data:     The actual result — a dict, list, whatever the tool produces.
    error:    Error message if failed.
    metadata: Optional extra info (timing, result count, sources used, etc.)
    """
    success: bool = True
    data: Any = None
    error: str = ""
    metadata: dict = field(default_factory=dict)

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

    def __init__(self, db, config: dict, services: dict = {}):
        self.db = db
        self.config = config
        self.services = services
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
        from Stage_1.registry import parse
        context = DataRefineryContext(
            db=self.db,
            config=self.config,
            services=self.services,
            call_tool=self.call,
            parse=lambda path, modality=None, config=None: parse(path, modality, config, self.services),  # Passes services automatically to parsers
        )

        try:
            result = tool.run(context, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return ToolResult.failed(str(e))

    def get_all_schemas(self) -> list[dict]:
        """Export all tool schemas for LLM function calling."""
        return [tool.to_schema() for tool in self.tools.values()]

    def get_schema(self, name: str) -> dict | None:
        tool = self.tools.get(name)
        return tool.to_schema() if tool else None

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())