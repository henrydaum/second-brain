"""
Read File tool.

Gives the LLM agent a simple, direct way to read file contents by path.
No shell commands, no timeouts, no syntax to remember.
"""

from pathlib import Path

from Stage_3.BaseTool import BaseTool, ToolResult
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

        if len(content) > MAX_CHARS:
            total = len(content)
            if target.suffix == ".log":
                # Log files want the most recent lines, not the oldest.
                tail = content[-MAX_CHARS:]
                nl = tail.find("\n")
                if nl != -1:
                    tail = tail[nl + 1:]
                content = f"... (earlier lines truncated, {total} chars total)\n\n" + tail
            else:
                content = content[:MAX_CHARS] + f"\n\n... truncated ({total} chars total)"

        return ToolResult(llm_summary=content)
