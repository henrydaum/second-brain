"""Show files to the user in the chat.

The residual outbound case. A tool that *produces* a file should hand it back
on its own result — ``sdk.ok(..., attachments=[path])`` — which is what the
search tools do, and it costs nothing extra. This tool exists for the file no
tool in this turn produced: one the agent found by path and wants the user to
see. That is a small job done infrequently, which is why this is a small tool.

It was called ``render_files``, and the name was doing real damage: "render" is
a word about drawing, so the tool read as being *for images*, and the agent
mostly reached for it only there — describing a PDF it could have simply sent.
The description had said "always use this for images, audio, and video" from
the start, which is the sentence a model weighs against a name that agrees with
half of it. Nothing about the mechanism is modality-specific; it hands paths
back, and the frontend decides how to draw each one. The name says that now,
and the description leads with breadth and with the reason the tool exists at
all — the user cannot see a file until it is shown to them.

Sandboxed: the paths ride out on the tool's own result, the same route
``ToolResult.attachment_paths`` always was. Existence is checked with
``fs.list`` pointed at each file, since a box has no ``pathlib``.
"""

dependencies_files = []
dependencies_pip = []
requests = ["fs.list"]

from guest.bases import BaseTool

MAX_FILES = 10


def _exists(sdk, path) -> bool:
    """Whether ``path`` names a file that is there.

    ``fs.list`` pointed at a file answers for that file alone, which is how a
    box asks a question ``Path.exists()`` used to answer.
    """
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed:
        return False
    return any(not entry.get("is_dir") for entry in entries or [])


class ShowFiles(BaseTool):
    """Show files to the user."""
    name = "show_files"
    description = (
        "Show the user any local file(s) in chat, with an optional caption: images, audio, video, PDFs, spreadsheets, documents, code, archives — anything on disk, of any type or size. The user cannot see a file until you show it, so send the file rather than describing it. Always use this for images, audio, and video — a description is not a substitute. Use it for any file the user asked you to find, open, or check, and for files your reply refers to that they will want to look at themselves. Skip it only when your text fully covers the content (e.g. you quoted the three relevant lines). This shows a file to the *user*; to look at one yourself, read_file it."
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
                "description": "Optional short text shown alongside the files in the same chat turn (e.g. 'Here are the three invoices that match.'). Use this instead of sending a separate reply when the text is about the files.",
            },
        },
        "required": ["paths"],
    }
    requires_services = []

    def run(self, sdk, **kwargs):
        """Run show files."""
        paths = kwargs.get("paths") or []
        caption = (kwargs.get("caption") or "").strip()
        if not paths:
            return sdk.fail("No file paths provided.")

        valid = []
        missing = []
        for p in paths:
            (valid if _exists(sdk, p) else missing).append(str(p))

        if not valid:
            return sdk.fail(
                f"None of the provided paths exist: {missing}. "
                f"If you guessed the paths, try hybrid_search first to find real ones."
            )

        truncated_extra = max(0, len(valid) - MAX_FILES)
        if truncated_extra:
            valid = valid[:MAX_FILES]

        names = ", ".join(sdk.path.name(p) for p in valid)
        notes = []
        if truncated_extra:
            notes.append(f"Skipped {truncated_extra} extra path(s) — {MAX_FILES}-file limit per call.")
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
            summary = f"Showed {len(valid)} file(s) to the user: {names}."
            if notes:
                summary += " " + " ".join(notes)

        return sdk.ok(
            {"caption": caption} if caption else None,
            llm_summary=summary,
            attachments=valid,
        )
