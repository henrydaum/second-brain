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
    """
    db: Any = None
    config: dict = field(default_factory=dict)
    services: dict = field(default_factory=dict)
    parse: Any = None        # callable(path, modality, config) -> ParseResult
    call_tool: Any = None    # callable(name, **kwargs) -> ToolResult (tools only)


# Backward-compat alias so existing plugins using the old name still work
DataRefineryContext = SecondBrainContext


def build_context(db, config: dict, services: dict, call_tool=None) -> SecondBrainContext:
    """
    Factory that wires up a fully functional context.

    The parse callable is built here, with services baked in, so callers
    don't need to import Stage_1.registry or build the lambda themselves.

    Usage:
        # In orchestrator (tasks — no call_tool):
        context = build_context(self.db, self.config, self.services)

        # In tool registry (tools — with call_tool):
        context = build_context(self.db, self.config, self.services, call_tool=self.call)
    """
    from Stage_1.registry import parse as _parse

    return SecondBrainContext(
        db=db,
        config=config,
        services=services,
        parse=lambda path, modality=None, config=None: _parse(path, modality, config, services),
        call_tool=call_tool,
    )
    