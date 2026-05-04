"""
Runtime context passed into tools and tasks.

The context packages together the database handle, config, shared
services, and a few runtime helpers so plugins do not need to know how
the surrounding application is wired. Parsing is reached uniformly via
``context.services.get("parser").parse(path, modality)`` — no special
shortcut on the context.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecondBrainContext:
    """
    The runtime context every task and tool receives.

    db:
        Database instance for reads and writes.
    config:
        Global settings dict.
    services:
        Mapping of service name to service instance. Includes the
        "parser" service for file parsing.
    call_tool:
        Helper for tool-to-tool composition. Only populated for tools.
        Example:
        context.call_tool("hybrid_search", query="revenue") -> ToolResult
    approve_command:
        Helper for user approval on sensitive actions. Tools only. None means
        no subscribed UI is available, so tools should treat that as deny.
    is_subagent:
        True when this context belongs to a scheduled / unattended subagent
        run. Tools that need to apply tighter authority (e.g. restrict mail
        access to a scoped alias) should branch on this flag.
    """
    db: Any = None
    config: dict = field(default_factory=dict)
    services: dict = field(default_factory=dict)
    call_tool: Any = None        # callable(name, **kwargs) -> ToolResult (tools only)
    approve_command: Any = None  # callable(command, justification) -> bool (tools only)
    tool_registry: Any = None    # ToolRegistry instance (tools only)
    orchestrator: Any = None     # Orchestrator instance (tools only)
    is_subagent: bool = False    # True inside task_run_subagent execution
    runtime: Any = None          # ConversationRuntime — present for tasks that
                                 # need to drive a state-machine session
                                 # (scheduled subagents in particular).


def build_context(db, config: dict, services: dict, call_tool=None,
                   tool_registry=None, orchestrator=None,
                   is_subagent: bool = False, runtime=None) -> SecondBrainContext:
    """
    Build a fully wired runtime context.

    approve_command is auto-wired to the event bus. If nothing is
    subscribed to APPROVAL_REQUESTED, the field stays None so tools can
    detect that no approval UI is available.

    Usage:
        # In orchestrator (tasks — no call_tool):
        context = build_context(self.db, self.config, self.services)

        # In tool registry (tools — with call_tool):
        context = build_context(self.db, self.config, self.services, call_tool=self.call)
    """
    from events.event_bus import bus
    from events.event_channels import APPROVAL_REQUESTED
    from events.approval_request import ApprovalRequest

    approve_command = None
    if call_tool is not None and bus.has_subscribers(APPROVAL_REQUESTED):
        def approve_command(command: str, justification: str) -> bool:
            req = ApprovalRequest(command, justification)
            bus.emit(APPROVAL_REQUESTED, req)
            if not req.wait(timeout=300.0):
                req.metadata["timed_out"] = True
                req.resolve(False)
                return False
            return req.approved

    return SecondBrainContext(
        db=db,
        config=config,
        services=services,
        call_tool=call_tool,
        approve_command=approve_command,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        is_subagent=is_subagent,
        runtime=runtime,
    )
