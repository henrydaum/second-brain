"""
Read File tool.

Gives the LLM agent a simple, direct way to read file contents by path.
No shell commands, no timeouts, no syntax to remember.
"""

from pathlib import Path

from Stage_3.BaseTool import BaseTool, ToolResult
from paths import ROOT_DIR

MAX_CHARS = 20_000


def _number_lines(text: str, start_line: int = 1) -> str:
    """Prefix each line with its line number, cat -n style (tab-separated)."""
    lines = text.split("\n")
    # Trailing newline produces an empty final element; drop it so we don't number a phantom line.
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return "\n".join(f"{i + start_line:>6}\t{line}" for i, line in enumerate(lines))


class ReadFile(BaseTool):
    name = "read_file"
    description = (
        "Read a text file by path. Use this when you need the exact contents of "
        "source code, templates, docs, or sandbox plugins. Paths may be absolute "
        "or relative to the project root. "
        "Output is prefixed with line numbers (cat -n style, tab-separated); "
        "the numbers are not part of the file content."
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
                split_at = total - MAX_CHARS
                nl = content.find("\n", split_at)
                if nl != -1:
                    dropped = content[: nl + 1]
                    tail = content[nl + 1 :]
                else:
                    dropped = content[:split_at]
                    tail = content[split_at:]
                start_line = dropped.count("\n") + 1
                numbered = _number_lines(tail, start_line)
                content = f"... (earlier lines truncated, {total} chars total)\n\n" + numbered
            else:
                content = _number_lines(content[:MAX_CHARS]) + f"\n\n... truncated ({total} chars total)"
        else:
            content = _number_lines(content)

        return ToolResult(llm_summary=content)
