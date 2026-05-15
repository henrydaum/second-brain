"""Display files to the user in the chat."""

from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class RenderFiles(BaseTool):
    """Render files."""
    name = "render_files"
    description = (
        "Display exactly one local image in the web demo's large showcase pane. Use this only for image files you already created or found. It will not render documents, audio, video, or arbitrary files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Image path to display. Only the first valid image is used.",
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
            path = Path(p)
            if path.exists() and path.suffix.lower() in IMAGE_EXTS:
                valid.append(str(path))
            else:
                missing.append(p)

        if not valid:
            return ToolResult.failed(
                f"No renderable image path was provided. Missing or non-image paths: {missing}."
            )

        valid = valid[:1]
        names = Path(valid[0]).name
        notes = []
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
            summary = f"Displayed image in the showcase pane: {names}."
            if notes:
                summary += " " + " ".join(notes)

        return ToolResult(
            data={"caption": caption, "display": "hero_image", "path": valid[0]},
            llm_summary=summary,
            attachment_paths=valid,
        )
