"""
Read File tool.

Gives the LLM agent a simple, direct way to read file contents by path.
No shell commands, no timeouts, no syntax to remember.
"""

from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from paths import ROOT_DIR

MAX_CHARS = 20_000


class ReadFile(BaseTool):
    name = "read_file"
    description = (
        "Read a text file by path. Use this when you need the exact contents of "
        "source code, templates, docs, or sandbox plugins. Paths may be absolute "
        "or relative to the project root."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to read, either absolute or relative to the project root.",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed). Default 1. For .log files this counts from the newest line.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return. Output is also capped at ~20k chars regardless.",
            },
        },
        "required": ["path"],
    }
    requires_services = []
    agent_enabled = True
    max_calls = 10

    def run(self, context, **kwargs) -> ToolResult:
        raw_path = kwargs.get("path", "").strip()
        if not raw_path:
            return ToolResult.failed("No path provided.")

        try:
            offset = int(kwargs.get("offset") or 1)
        except (TypeError, ValueError):
            offset = 1
        offset = max(1, offset)

        limit_raw = kwargs.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else None
        except (TypeError, ValueError):
            limit = None
        if limit is not None:
            limit = max(1, limit)

        target = Path(raw_path)
        if not target.is_absolute():
            target = ROOT_DIR / target

        if not target.exists():
            return ToolResult.failed(f"File not found: {target}")
        if not target.is_file():
            return ToolResult.failed(f"Not a file: {target}")

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.failed(f"Cannot read as text (binary file?): {target}")
        except Exception as e:
            return ToolResult.failed(f"Read error: {e}")

        lines = content.splitlines()
        if target.suffix == ".log":
            # Logs are read newest-first so the latest messages are always visible.
            lines = list(reversed(lines))

        total_lines = len(lines)
        start = min(offset - 1, total_lines)
        end = total_lines if limit is None else min(start + limit, total_lines)
        window = lines[start:end]
        content = "\n".join(window)

        char_truncated = False
        if len(content) > MAX_CHARS:
            nl = content.rfind("\n", 0, MAX_CHARS)
            content = content[:nl] if nl != -1 else content[:MAX_CHARS]
            char_truncated = True

        notes = []
        if start > 0:
            notes.append(f"showing lines {start + 1}-{end} of {total_lines}")
        elif end < total_lines:
            notes.append(f"showing lines 1-{end} of {total_lines}")
        if char_truncated:
            notes.append(f"output capped at {MAX_CHARS} chars — pass offset/limit to page further")
        if notes:
            content += "\n\n... (" + "; ".join(notes) + ")"

        return ToolResult(llm_summary=content)
