"""
Tool registry.

Owns tool registration, dispatch, and schema export. Separated from
BaseTool.py so the base contract stays lightweight and the tool template
can focus on authoring guidance instead of runtime plumbing.
"""

import logging
import threading
import time

from context import build_context
from Stage_3.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("Tool")


class ToolRegistry:
    """
    Registry and execution entry point for tools.

    Responsibilities:
        1. Store tool instances by name
        2. Dispatch tool calls, including tool-to-tool composition
        3. Export LLM-visible schemas for agent use
    """

    def __init__(self, db, config: dict, services: dict = None):
        self.db = db
        self.config = config
        self.services = services or {}
        self.tools: dict[str, BaseTool] = {}
        self._lock = threading.Lock()
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

    def call(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Used by:
            - External callers such as the REPL, API, or agent
            - Other tools via context.call_tool
        """
        with self._lock:
            tool = self.tools.get(name)
        if tool is None:
            return ToolResult.failed(f"Unknown tool: {name}")

        # Gate on required services before building a runtime context.
        if tool.requires_services:
            not_ready = []
            for svc_name in tool.requires_services:
                svc = self.services.get(svc_name)
                if svc is None or not svc.loaded:
                    not_ready.append(svc_name)
            if not_ready:
                return ToolResult.failed(f"Required services not available: {not_ready}")
        
        # Build a fresh runtime context for this invocation. call_tool points
        # back to the registry, and approve_command is wired inside build_context.
        context = build_context(self.db, self.config, self.services,
                                call_tool=self.call,
                                tool_registry=self,
                                orchestrator=self.orchestrator)

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
        """Return the agent's total tool-call budget for one message."""
        return sum(t.max_calls for t in self.tools.values() if t.agent_enabled)

    def get_all_schemas(self) -> list[dict]:
        """Export schemas for agent-enabled tools."""
        return [tool.to_schema() for tool in self.tools.values() if tool.agent_enabled]

    def get_schema(self, name: str) -> dict | None:
        tool = self.tools.get(name)
        return tool.to_schema() if tool else None

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())
