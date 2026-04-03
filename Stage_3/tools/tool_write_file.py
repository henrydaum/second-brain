"""
Write File tool.

Writes a markdown (.md) file into a configured sync directory.
Used by the agent to leave durable notes, summaries, and research
that persist for future sessions, are readable by humans, and are
automatically indexed by the pipeline.
"""

from pathlib import Path

from Stage_3.BaseTool import BaseTool, ToolResult


class WriteFile(BaseTool):
    name = "write_file"
    description = (
        "Write a markdown (.md) file to the sync directory. "
        "Use this to save notes, summaries, deep-dive results, or research "
        "that should persist across sessions. The file will be indexed and "
        "searchable. Path must be relative (e.g. '_summaries/2026-04-02.md')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": (
                    "Relative path within the sync directory "
                    "(e.g. '_summaries/notes.md'). Must end in .md."
                ),
            },
            "content": {
                "type": "string",
                "description": "Markdown content to write.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Overwrite if file already exists. Defaults to false.",
            },
        },
        "required": ["relative_path", "content"],
    }
    requires_services = []
    agent_enabled = True
    max_calls = 5

    def run(self, context, **kwargs) -> ToolResult:
        relative_path = kwargs.get("relative_path", "").strip()
        content = kwargs.get("content", "")
        overwrite = kwargs.get("overwrite", False)

        if not relative_path:
            return ToolResult.failed("No path provided.")

        if not relative_path.endswith(".md"):
            return ToolResult.failed("Only .md files are allowed.")

        target = Path(relative_path)
        if target.is_absolute():
            return ToolResult.failed("Path must be relative, not absolute.")

        sync_dirs = context.config.get("sync_directories", [])
        if not sync_dirs:
            return ToolResult.failed("No sync directories configured.")

        base = Path(sync_dirs[0]).resolve()
        resolved = (base / target).resolve()

        # Security: path must stay within the sync directory
        try:
            resolved.relative_to(base)
        except ValueError:
            return ToolResult.failed("Path escapes the sync directory.")

        if resolved.exists() and not overwrite:
            return ToolResult.failed(
                f"File already exists: {relative_path}. Pass overwrite=true to replace it."
            )

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except Exception as e:
            return ToolResult.failed(f"Write error: {e}")

        return ToolResult(
            llm_summary=f"Written: {resolved}",
            data={"path": str(resolved)},
        )
