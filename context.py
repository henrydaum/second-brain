"""
DataRefinery Context.

The single context object passed to everything: tasks, tools, parsers.
Built once in main.py, shared everywhere.

Tasks get:     db, config, services, parse
Tools get:     db, config, services, parse, call_tool
Parsers get:   config (if they need it)

The services field is a ServiceManager instance. Tasks and tools
access models through it:

    embedder = context.services.get("embedder")
    embedder.encode(chunks)

This avoids duplicate model instances — everyone shares the same
embedder, the same LLM, the same OCR engine.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataRefineryContext:
    """
    What every task and tool receives when it runs.

    db:         Database instance. Read/write access.
    config:     Global settings dict.
    services:   ServiceManager instance. Access shared models and factories.
    parse:      Call a parser directly.
                Usage: context.parse("report.pdf", "text") -> ParseResult
    call_tool:  Invoke another tool by name. Only populated for tools.
                Usage: context.call_tool("keyword_search", query="revenue") -> ToolResult
    """
    db: Any = None
    config: dict = field(default_factory=dict)
    services: Any = None     # ServiceManager instance
    parse: Any = None        # callable(path, modality, config) -> ParseResult
    call_tool: Any = None    # callable(name, **kwargs) -> ToolResult (tools only)