"""Display files to the user in the chat."""

from pathlib import Path

from Stage_3.BaseTool import BaseTool, ToolResult


class RenderFiles(BaseTool):
    name = "render_files"
    description = (
        "Display one or more files to the user in the chat. "
        "Accepts any file path — images, text, audio, video, tabular data, etc. "
        "Use this to show the user specific files. "
        'Example: {"paths": ["C:/Users/user/Documents/report.pdf", "C:/Users/user/Pictures/photo.jpg"]}'
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of absolute file paths to display.",
            }
        },
        "required": ["paths"],
    }
    requires_services = []
    max_calls = 5

    def run(self, context, **kwargs) -> ToolResult:
        paths = kwargs.get("paths", [])
        if not paths:
            return ToolResult.failed("No file paths provided.")

        valid = []
        missing = []
        for p in paths:
            if Path(p).exists():
                valid.append(str(Path(p)))
            else:
                missing.append(p)

        if not valid:
            return ToolResult.failed(f"None of the provided paths exist: {missing}")

        summary = "Successfully rendered file paths to the user."
        if missing:
            summary += f" (Missing: {missing})"

        return ToolResult(
            llm_summary=summary,
            gui_display_paths=valid,
        )
