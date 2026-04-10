"""
Tool registry — manages tool registration, dispatch, and schema export.

Separated from BaseTool.py so that the tool template stays clean for
LLM consumption. Only infrastructure code lives here.
"""

import logging
import threading
import time

from context import build_context
from Stage_3.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("Tool")


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
        self._lock = threading.Lock()
        self.on_approve_command = None  # callable(command, justification) -> bool
        self.orchestrator = None        # set after construction in main.pyw

    def register(self, tool: BaseTool):
        """Register a tool. Overwrites if name already exists."""
        with self._lock:
            self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str):
        """Remove a tool from the registry (used by build_plugin on delete)."""
        with self._lock:
            removed = self.tools.pop(name, None)
        if removed:
            logger.info(f"Unregistered tool: {name}")

    def call(self, name: str, mcp_context=None, **kwargs) -> ToolResult:
        """
        Call a tool by name. This is the single dispatch point.

        Used by:
            - External callers (API, CLI, LLM)
            - Other tools via context.call_tool
        """
        with self._lock:
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
        
        # STRICTLY FOR MODEL CONTEXT PROTOCOL (SAMPLING):
        def _sample_wrapper(prompt: str, max_tokens: int = 500) -> str:
            if not mcp_context:
                return "Error: Sampling not available outside of MCP host."
            
            import asyncio
            async def _do_sample():
                res = await mcp_context.session.create_message(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                return res.content.text
            return asyncio.run(_do_sample())

        # Build context with call_tool pointing back to this registry
        context = build_context(self.db, self.config, self.services,
                                call_tool=self.call,
                                approve_command=self.on_approve_command,
                                tool_registry=self,
                                orchestrator=self.orchestrator,
                                sample_llm=_sample_wrapper)

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
