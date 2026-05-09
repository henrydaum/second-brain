"""
Tool registry.

Owns tool registration, dispatch, and schema export. Separated from
BaseTool.py so the base contract stays lightweight and the tool template
can focus on authoring guidance instead of runtime plumbing.
"""

import concurrent.futures
import logging
import threading
import time

from runtime.context import build_context
from plugins.BaseTool import BaseTool, ToolResult
from events.event_bus import bus
from events.event_channels import TOOLS_CHANGED

logger = logging.getLogger("Tool")

# Thread-local flag so reentrant tool calls (tool -> context.call_tool -> tool)
# skip the timeout wrapper. Only top-level calls get wrapped, otherwise nested
# calls would consume extra executor threads and could deadlock.
_exec_state = threading.local()


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
        self.visible_tool_names: set[str] | None = None
        self._lock = threading.Lock()
        self.orchestrator = None        # set after construction in main.pyw
        self.runtime = None             # ConversationRuntime, set by frontend bootstrap

    def register(self, tool: BaseTool):
        """Register a tool. Overwrites if name already exists."""
        with self._lock:
            self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
        bus.emit(TOOLS_CHANGED, {"name": tool.name, "action": "registered"})

    def unregister(self, name: str):
        """Remove a tool from the registry (used by build_plugin on delete)."""
        with self._lock:
            removed = self.tools.pop(name, None)
        if removed:
            logger.info(f"Unregistered tool: {name}")
            bus.emit(TOOLS_CHANGED, {"name": name, "action": "unregistered"})

    def call(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Used by:
            - External callers such as the REPL, API, or agent
            - Other tools via context.call_tool
        """
        session_key = kwargs.pop("_session_key", None)
        user_initiated = bool(kwargs.pop("_user_initiated", False))
        with self._lock:
            tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult.failed(f"Unknown tool: {tool_name}")

        # Background-safety gate: tools marked background_safe=False are
        # interactive (they need a human watching). Refuse if the call is
        # coming from a session that isn't the currently active one.
        if (not getattr(tool, "background_safe", True)
                and session_key is not None
                and self.runtime is not None
                and session_key != getattr(self.runtime, "active_session_key", None)):
            return ToolResult.failed(
                f"Tool '{tool_name}' requires an active conversation and cannot run in the background."
            )

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
        # back to the registry, and approvals go through the owning session.
        context = build_context(self.db, self.config, self.services,
                                call_tool=self.call,
                                tool_registry=self,
                                orchestrator=self.orchestrator,
                                runtime=self.runtime,
                                session_key=session_key,
                                user_initiated=user_initiated)

        t0 = time.time()

        # Reentrant calls (tool -> call_tool -> tool) run inline — the outer
        # call already owns a timeout budget, and a nested submit would double
        # the thread count without adding safety.
        if getattr(_exec_state, "in_tool", False):
            try:
                result = tool.run(context, **kwargs)
                logger.debug(f"Tool '{tool_name}' completed in {time.time() - t0:.3f}s")
                return result
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed after {time.time() - t0:.3f}s: {e}")
                return ToolResult.failed(str(e))

        timeout = int(self.config.get("tool_timeout", 600))
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"sb-tool-{tool_name}")

        def _run_with_flag():
            _exec_state.in_tool = True
            try:
                return tool.run(context, **kwargs)
            finally:
                _exec_state.in_tool = False

        try:
            future = executor.submit(_run_with_flag)
            try:
                result = future.result(timeout=timeout)
                logger.debug(f"Tool '{tool_name}' completed in {time.time() - t0:.3f}s")
                return result
            except concurrent.futures.TimeoutError:
                logger.error(f"Tool '{tool_name}' timed out after {timeout}s — abandoning thread")
                # Don't wait for the zombie thread — it may never finish.
                return ToolResult.failed(
                    f"Tool '{tool_name}' timed out after {timeout}s and was abandoned.")
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed after {time.time() - t0:.3f}s: {e}")
                return ToolResult.failed(str(e))
        finally:
            executor.shutdown(wait=False)

    @property
    def max_tool_calls(self) -> int:
        """Return the agent's total tool-call budget for one message."""
        return sum(t.max_calls for t in self._visible_tools())

    def get_all_schemas(self) -> list[dict]:
        """Export schemas for every agent-visible tool."""
        return [tool.to_schema() for tool in self._visible_tools()]

    def get_schema(self, name: str) -> dict | None:
        if self.visible_tool_names is not None and name not in self.visible_tool_names:
            return None
        tool = self.tools.get(name)
        return tool.to_schema() if tool else None

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def _visible_tools(self):
        if self.visible_tool_names is None:
            return self.tools.values()
        return [tool for name, tool in self.tools.items() if name in self.visible_tool_names]
