from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecondBrainContext:
    """
    What every task and tool receives when it runs.

    db:         Database instance. Read/write access.
    config:     Global settings dict.
    services:   Dict of {name: service_instance}. Access shared models.
    parse:      Call a parser directly.
                Usage: context.parse("report.pdf", "text") -> ParseResult
    call_tool:  Invoke another tool by name. Only populated for tools.
                Usage: context.call_tool("keyword_search", query="revenue") -> ToolResult
    approve_command: Request human approval for a destructive action.
                Tools-only. None if no UI is subscribed (tools should treat
                that as auto-deny). Backed by the event bus (APPROVAL_REQUESTED).
    """
    db: Any = None
    config: dict = field(default_factory=dict)
    services: dict = field(default_factory=dict)
    parse: Any = None            # callable(path, modality, config) -> ParseResult
    call_tool: Any = None        # callable(name, **kwargs) -> ToolResult (tools only)
    approve_command: Any = None  # callable(command, justification) -> bool (tools only)
    tool_registry: Any = None    # ToolRegistry instance (tools only)
    orchestrator: Any = None     # Orchestrator instance (tools only)


def build_context(db, config: dict, services: dict, call_tool=None,
                   tool_registry=None, orchestrator=None) -> SecondBrainContext:
    """
    Factory that wires up a fully functional context.

    The parse callable is built here, with services baked in, so callers
    don't need to import Stage_1.registry or build the lambda themselves.

    approve_command is auto-wired to the event bus. If no subscriber is
    registered on APPROVAL_REQUESTED (e.g. headless/REPL mode), the field
    stays None so tools can detect "no UI available" and auto-deny.

    Usage:
        # In orchestrator (tasks — no call_tool):
        context = build_context(self.db, self.config, self.services)

        # In tool registry (tools — with call_tool):
        context = build_context(self.db, self.config, self.services, call_tool=self.call)
    """
    from Stage_1.registry import parse as _parse
    from event_bus import bus
    from event_channels import APPROVAL_REQUESTED

    approve_command = None
    if call_tool is not None and bus.has_subscribers(APPROVAL_REQUESTED):
        def approve_command(command: str, justification: str) -> bool:
            reply = bus.request(
                APPROVAL_REQUESTED,
                {"command": command, "reason": justification},
                timeout=300.0,
            )
            return bool(reply)

    return SecondBrainContext(
        db=db,
        config=config,
        services=services,
        parse=lambda path, modality=None, config=None: _parse(path, modality, config, services),
        call_tool=call_tool,
        approve_command=approve_command,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
    )
