"""
Read File tool.

Gives the LLM agent a simple, direct way to read file contents by path.
No shell commands, no timeouts, no syntax to remember.

Sandboxed: the read is an ``fs.read`` Request, so the kernel decides what may
be read (and refuses the config file and the database outright, whatever the
path says). Nothing here asks permission — reads are classified safe, and
always were.
"""

dependencies_files = ['tools/helpers/file_reads.py']
dependencies_pip = []
requests = ["fs.read", "fs.list", "paths.get",
            "session.state_get", "session.state_set"]

from guest.bases import BaseTool

# Flat: the box is one namespace and the declared dependency's directory is on
# its import path, so the helper is a sibling despite shipping in a subfolder.
from .file_reads import record_read

MAX_CHARS = 20_000


class ReadFile(BaseTool):
    """Read file."""
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
            "line_numbers": {
                "type": "boolean",
                "description": "Include 1-indexed line numbers. Defaults to true; pass false when you need raw text for exact replacement.",
            },
        },
        "required": ["path"],
    }
    requires_services = []

    def run(self, sdk, **kwargs):
        """Run read file."""
        raw_path = (kwargs.get("path") or "").strip()
        if not raw_path:
            return sdk.fail("No path provided.")

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

        target = sdk.path.absolute(raw_path, base=sdk.paths.get("project"))

        try:
            content = sdk.fs.read(target)
        except sdk.Denied as refused:
            return sdk.fail(str(refused))
        except sdk.Failed as failed:
            return sdk.fail(f"Could not read {target}: {failed.error}")

        lines = content.splitlines()
        if sdk.path.suffix(target) == ".log":
            # Logs are read newest-first so the latest messages are always visible.
            lines = list(reversed(lines))

        total_lines = len(lines)
        start = min(offset - 1, total_lines)
        end = total_lines if limit is None else min(start + limit, total_lines)
        window = lines[start:end]
        if kwargs.get("line_numbers", True):
            window = [f"{i}: {line}" for i, line in enumerate(window, start + 1)]
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

        # Mark the file as seen for edit_file's read-before-edit gate.
        record_read(sdk, target)
        return sdk.ok(None, llm_summary=content)
