"""
Tool registry.

Owns tool registration, dispatch, and schema export. Separated from
BaseTool.py so the base contract stays lightweight and the tool template
can focus on authoring guidance instead of runtime plumbing.
"""

import logging
import threading
import time

from runtime.context import build_context
from plugins.native.tool import BaseTool, ToolResult
from events.event_bus import bus
from events.event_channels import TOOLS_CHANGED

logger = logging.getLogger("Tool")

# Fallback for `default_tool_max_calls` when no config reaches the registry.
# Mirrors the setting's own default in config_data.
DEFAULT_TOOL_MAX_CALLS = 25


class ToolRegistry:
    """
    Registry and execution entry point for tools.

    Responsibilities:
        1. Store tool instances by name
        2. Dispatch tool calls, including tool-to-tool composition
        3. Export LLM-visible schemas for agent use
    """

    def __init__(self, db, config: dict, services: dict = None):
        """Initialize the tool registry."""
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
        """Remove a tool from the registry during plugin unload/delete."""
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
        # ``narration`` is a reserved parameter name: a tool declares it in its
        # schema so the model can say *why* it is calling, and the kernel renders
        # that beside the status line. The tool never receives it — the kernel
        # owns the rendering, and tool signatures are explicit by house style
        # (``def run(self, sdk, url)``), so an unstripped kwarg is a TypeError in
        # every tool that declares one.
        #
        # This pop is the single point covering native *and* sandboxed tools: it
        # is upstream of the bridge's ``run_tool`` -> ``_forward`` -> the guest's
        # ``run(sdk, **kwargs)``. It mutates the fresh dict built by the ``**args``
        # unpack in ``runtime_config.tool_specs_for``, not the ``content["args"]`` the
        # conversation loop holds — so the status payload and the history row keep
        # the narration and only the call loses it.
        kwargs.pop("narration", None)
        with self._lock:
            tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult.failed(f"Unknown tool: {tool_name}")

        # There was a background-safety gate here: a tool declaring
        # ``background_safe = False`` was refused outright from an unattended
        # session. It predated the policy function and duplicated it — an
        # unattended chain already refuses every unsafe Request
        # (``sandbox/approval.py`` step 3), which is where ``ui.ask`` and the
        # rest are actually decided. It was also a declaration *by the code
        # being contained* about its own containment, the same shape as the
        # retired ``isolation`` attribute. The ``unattended_call`` hook stage
        # survives; the approver is its other producer.

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
                                user_initiated=user_initiated,
                                current_tool_name=tool_name)

        t0 = time.time()

        # A tool runs on the calling thread, including the reentrant case
        # (tool -> call_tool -> tool).
        try:
            result = tool.run(context, **kwargs)
            logger.debug(f"Tool '{tool_name}' completed in {time.time() - t0:.3f}s")
            return result
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed after {time.time() - t0:.3f}s: {e}")
            return ToolResult.failed(str(e))

    def call_limit(self, tool) -> int:
        """How many times `tool` may be called in one message.

        The single place the declaration-or-default question is answered, so
        the budget the loop enforces and the budget the iteration bound is
        derived from cannot drift. An undeclared `max_calls` is the normal
        case: a tool says a number only when its own nature bounds it.
        """
        declared = getattr(tool, "max_calls", None)
        if declared is not None:
            try:
                return max(1, int(declared))
            except (TypeError, ValueError):
                pass
        try:
            return max(1, int((self.config or {}).get("default_tool_max_calls")
                              or DEFAULT_TOOL_MAX_CALLS))
        except (TypeError, ValueError):
            return DEFAULT_TOOL_MAX_CALLS

    @property
    def max_tool_calls(self) -> int:
        """Return the agent's total tool-call budget for one message."""
        return sum(self.call_limit(t) for t in self._visible_tools())

    def get_all_schemas(self) -> list[dict]:
        """Export schemas for every agent-visible tool."""
        return [tool.to_schema() for tool in self._visible_tools()]

    def get_schema(self, name: str) -> dict | None:
        """Get schema."""
        if self.visible_tool_names is not None and name not in self.visible_tool_names:
            return None
        tool = self.tools.get(name)
        return tool.to_schema() if tool else None

    def list_tools(self) -> list[str]:
        """List tools."""
        return list(self.tools.keys())

    def _visible_tools(self):
        """Internal helper to handle visible tools."""
        if self.visible_tool_names is None:
            return self.tools.values()
        return [tool for name, tool in self.tools.items() if name in self.visible_tool_names]
