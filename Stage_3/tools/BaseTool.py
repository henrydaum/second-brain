"""
Tool interface.

Tools are the query layer — they accept arguments, read from the database
and parsers, call other tools, and return structured results.

Unlike tasks (which run in the background on every file), tools are
called on demand and return results immediately to a caller.

Three tiers:
    Tier 1 - Data tools:     Read from task output tables. Pure queries.
    Tier 2 - Compute tools:  Do work at call time. Embed a query, call an LLM.
    Tier 3 - Composite tools: Orchestrate other tools. Research, construct tasks.

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

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """
    What a tool receives when it runs.

    db:         Database access — read task outputs, file info, etc.
    config:     Global settings.
    call_tool:  Invoke another tool by name. Returns that tool's result dict.
                Usage: context.call_tool("keyword_search", query="neural nets", limit=10)
    parse:      Call a parser directly.
                Usage: context.parse("report.pdf", "text")
    """
    db: Any = None
    config: dict = field(default_factory=dict)
    call_tool: Any = None   # callable(name: str, **kwargs) -> dict
    parse: Any = None       # callable(path: str, modality: str) -> ParseResult


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
        name            Unique identifier. "keyword_search", "vector_search", etc.
        description     Natural language description. Doubles as the LLM tool description.
                        Be specific — the LLM uses this to decide when to call the tool.
        parameters      JSON schema dict describing the input arguments.
                        Follows the OpenAI function calling format.

    Methods (override these):
        run(**kwargs, context) -> ToolResult
    """

    # --- Identity ---
    name: str = ""
    description: str = ""
    parameters: dict = {}

    def run(self, context: ToolContext, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            context:    ToolContext with db access, config, call_tool, parse.
            **kwargs:   The arguments matching self.parameters schema.

        Returns:
            ToolResult with the data.
        """
        raise NotImplementedError(f"Tool '{self.name}' must implement run()")

    def to_schema(self) -> dict:
        """
        Export as an LLM-compatible function schema.
        Works with OpenAI, Anthropic, and similar function calling formats.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """
    Manages all registered tools and handles dispatch.

    The registry:
        1. Stores tool instances by name
        2. Dispatches calls (including tool-to-tool calls)
        3. Exports schemas for LLM function calling
        4. Provides the call_tool function that composite tools use
    """

    def __init__(self, db, config: dict):
        self.db = db
        self.config = config
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

        # Build context with call_tool pointing back to this registry
        context = ToolContext(
            db=self.db,
            config=self.config,
            call_tool=self.call,  # recursive — tools can call tools
            parse=self._get_parse_fn(),
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
        """Export a single tool's schema."""
        tool = self.tools.get(name)
        return tool.to_schema() if tool else None

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def _get_parse_fn(self):
        """Lazy import to avoid circular dependency."""
        try:
            from Stage_1.registry import parse
            return parse
        except ImportError:
            return None