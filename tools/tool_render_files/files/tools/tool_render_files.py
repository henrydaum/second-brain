"""Display files to the user in the chat."""

from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult


class RenderFiles(BaseTool):
    """Render files."""
    name = "render_files"
    description = (
        "Display one or more local files to the user in chat alongside an optional caption. Always use this for images, audio, and video — a description is not a substitute. Use it for documents the user asked to find or open, and for files referenced in your reply that the user will likely want to inspect. Skip it when your text reply fully covers the content (e.g. you quoted the three relevant lines)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to display. Maximum 10 per call.",
            },
            "caption": {
                "type": "string",
                "description": "Optional short text shown alongside the rendered files in the same chat turn (e.g. 'Here are the three invoices that match.'). Use this instead of sending a separate reply when the text is about the files.",
            },
        },
        "required": ["paths"],
    }
    requires_services = []
    max_calls = 5

    def run(self, context, **kwargs) -> ToolResult:
        """Run render files."""
        paths = kwargs.get("paths", [])
        caption = (kwargs.get("caption") or "").strip()
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
            return ToolResult.failed(
                f"None of the provided paths exist: {missing}. "
                f"If you guessed the paths, try hybrid_search first to find real ones."
            )

        truncated_extra = max(0, len(valid) - 10)
        if truncated_extra:
            valid = valid[:10]

        names = ", ".join(Path(p).name for p in valid)
        notes = []
        if truncated_extra:
            notes.append(f"Skipped {truncated_extra} extra path(s) — 10-file limit per call.")
        if missing:
            notes.append(f"Missing: {missing}")

        # llm_summary is shown to the user alongside the attachments AND echoed
        # back to the LLM. When a caption is given, lead with it so the user sees
        # it as the message accompanying the files.
        if caption:
            summary = caption
            if notes:
                summary += "\n\n(" + " ".join(notes) + ")"
        else:
            summary = f"Rendered {len(valid)} file(s) to the user: {names}."
            if notes:
                summary += " " + " ".join(notes)

        return ToolResult(
            data={"caption": caption} if caption else None,
            llm_summary=summary,
            attachment_paths=valid,
        )
