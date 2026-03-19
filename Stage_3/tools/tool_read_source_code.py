"""
Read source code tool.

Lets the LLM read Second Brain's own source code by module name.
Used when the agent needs to reference existing implementations to help
design new tools or tasks.
"""

from pathlib import Path

from Stage_3.BaseTool import BaseTool, ToolResult

# Build module-name → file-path map at import time
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SKIP = {"__pycache__", ".git", ".venv", "venv"}

_MODULE_MAP: dict[str, Path] = {}
for p in _PROJECT_ROOT.rglob("*"):
    if p.suffix not in (".py", ".pyw"):
        continue
    if any(part in _SKIP for part in p.parts):
        continue
    _MODULE_MAP[p.stem] = p


class ReadSourceCode(BaseTool):
    name = "read_source_code"
    description = (
        "Read the source code of a Second Brain module by name. "
        "Use this to see how existing tasks, tools, services, or parsers are implemented. "
        "Pass just the module name without extension (e.g. 'BaseTask', 'agent', 'tool_lexical_search')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "module": {
                "type": "string",
                "description": "Module name without extension (e.g. 'BaseTask', 'tool_sql_query').",
            }
        },
        "required": ["module"],
    }
    requires_services = []

    def run(self, context, **kwargs):
        module = kwargs.get("module", "").strip()
        if not module:
            return ToolResult.failed("No module name provided.")

        path = _MODULE_MAP.get(module)
        if path is None:
            available = ", ".join(sorted(_MODULE_MAP.keys()))
            return ToolResult.failed(
                f"Unknown module: '{module}'. Available: {available}"
            )

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult.failed(f"Failed to read {path}: {e}")

        return ToolResult(data=content, llm_summary=content, gui_display_paths=[str(path)])
