"""
Read File tool.

Gives the LLM agent a simple, direct way to read file contents by path.
No shell commands, no timeouts, no syntax to remember.

Sandboxed: the read is an ``fs.read`` Request, so the kernel decides what may
be read (and refuses the config file and the database outright, whatever the
path says). Nothing here asks permission — reads are classified safe, and
always were.

**One door for every file type**, because the agent should not have to know
what kind of file it is pointing at before it may look:

- image, audio, video — staged as an attachment, so the model *sees* it
- text — read as text, which is the path with offset/limit/line numbers
- a document that is not text at all (pdf, docx, xlsx) — handed to a parser

Only the middle one ever worked. ``fs.read`` on a PDF returned decoded binary
noise, and an image had no route to the model at all.

How the last branch decides was the subtle part, and it was wrong in a way
nothing reported. It could not be the extension's modality, because
``parse_pdf`` registers ``.pdf`` as "text" and so does ``parse_text`` for
``.py`` — one word for a document and for a source file. So it was decided on
the *bytes*: ``fs.read`` decodes with ``errors="replace"``, and a file that
comes back full of replacement characters has said it is not text.

That is right for a PDF and silently wrong for a **pointer** — a file whose
text is not its content. A ``.gdoc`` is a 150-byte JSON stub naming a Drive
document; it decodes perfectly, so it never reached ``parse_gdoc``, and the
agent got ``{"doc_id": ...}`` back as a successful read. The parser could not
have been reached by this route at all: it does ``json.loads(sdk.fs.read(...))``,
so being textual is a precondition of it working, and being textual is what
this branch took as proof no parser was needed.

The registry answers it now. ``sdk.parse.modality(ext, detail=True)`` reports
whether the parser behind an extension is the generic text fallback or a
format specialist, and a specialist owns its format: parse first, no bytes
read. The byte sniff stays as the fallback for extensions nothing has
registered, which is the case neither the registry nor a list can cover.

Reading text is deliberately still ``fs.read`` rather than the parser, even
though ``parse_text`` would answer: it applies ``clean_text`` and a char cap,
and ``edit_file``'s exact-replacement gate needs what is actually on disk.

Nothing here asks whether the model can read a modality. If it cannot, the
kernel substitutes the file's parsed text and failing that a line naming where
the file is — and a check in here could only get it wrong, since a box cannot
see which brain the session resolved to.
"""

dependencies_files = ['tools/helpers/file_reads.py']
dependencies_pip = []
requests = ["fs.read", "fs.list", "paths.get",
            "parse.modality", "parse.file", "session.add_attachment",
            "session.get", "session.state_get", "session.state_set"]

from guest.bases import BaseTool

# Flat: the box is one namespace and the declared dependency's directory is on
# its import path, so the helper is a sibling despite shipping in a subfolder.
from .file_reads import record_read

MAX_CHARS = 20_000

# A staged file's bytes reach the backend over the wire, which caps a message
# at 16 MB. Refusing here names the file and the size; letting it through fails
# inside the model call, where the report blames the provider.
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024

# Modalities the model ingests directly. Everything else is on its way to text.
NATIVE_MODALITIES = ("image", "audio", "video")

# Above this share of undecodable characters, the file is not text. A real
# UTF-8 file has none of these; a latin-1 one has a scattering, which is why
# this is a ratio rather than "any". NUL is decisive on its own — text files
# essentially never carry one and binary container formats nearly always do.
BINARY_RATIO = 0.05

# U+FFFD, named rather than pasted: the character this looks for is exactly
# the one a bad encoding round trip mangles, and a source file is no place to
# depend on that going well.
REPLACEMENT = chr(0xFFFD)


def _looks_binary(content: str) -> bool:
    """Whether ``fs.read`` just handed back a decoded binary file."""
    if not content:
        return False
    if "\x00" in content:
        return True
    return content.count(REPLACEMENT) / len(content) > BINARY_RATIO


def _size(sdk, path) -> int | None:
    """Byte size, or None when it cannot be determined.

    ``fs.list`` pointed at a file answers for that file alone — the same idiom
    ``file_reads`` stats one path with.
    """
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed:
        return None
    for entry in entries or []:
        if not entry.get("is_dir"):
            return entry.get("size")
    return None


class ReadFile(BaseTool):
    """Read file."""
    name = "read_file"
    description = (
        "Read any file by path. Use this when you need the exact contents of "
        "source code, templates, docs, or sandbox plugins. Paths may be absolute "
        "or relative to the project root. Images, audio and video are attached "
        "to your next message so you can look at them directly; documents such "
        "as PDFs and spreadsheets come back as extracted text. The "
        "offset/limit/line_numbers options apply to text files."
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

    # This text reads the permission mode and nothing else, so it goes stale
    # only when the session does — not on every file the agent writes, which is
    # what the default rung would charge for it.
    agent_prompt_refresh = "session"

    def agent_prompt(self, sdk):
        """Point lockdown toward the mediated file-content path."""
        if (sdk.session.get() or {}).get("mode") != "lockdown":
            return ""
        return (
            "## Reading files in lockdown\n"
            "Once you know a path, use read_file for its contents instead of "
            "shell reads. Page large text with offset and limit."
        )

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

        route = sdk.parse.modality(sdk.path.suffix(target), detail=True)
        if route["modality"] in NATIVE_MODALITIES:
            return self._attach(sdk, target, route["modality"])

        # A specialist parser owns its format, so it decides what the file
        # says -- before any bytes are read, which is what makes a pointer
        # like .gdoc work and saves pulling a whole PDF in to discover it is
        # binary. ``generic`` is the kernel's text parser, whose extensions
        # are their own content and must stay on fs.read: it applies
        # clean_text and a char cap, and edit_file's exact-replacement gate
        # needs what is actually on disk.
        if route["known"] and not route["generic"]:
            return self._parse(sdk, target)

        try:
            content = sdk.fs.read(target)
        except sdk.Denied as refused:
            return sdk.fail(str(refused))
        except sdk.Failed as failed:
            return sdk.fail(f"Could not read {target}: {failed.error}")

        if _looks_binary(content):
            return self._parse(sdk, target)

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

    def _attach(self, sdk, target, modality):
        """Put the file in front of the model rather than describing it."""
        size = _size(sdk, target)
        if size is None:
            return sdk.fail(f"Could not read {target}.")
        if size > MAX_ATTACHMENT_BYTES:
            mb = size / (1024 * 1024)
            return sdk.fail(
                f"{sdk.path.name(target)} is {mb:.1f} MB, over the "
                f"{MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB limit for a file "
                "the model can be shown."
            )

        try:
            sdk.session.add_attachment(target)
        except sdk.Failed as failed:
            return sdk.fail(f"Could not attach {target}: {failed.error}")

        return sdk.ok(None, llm_summary=(
            f"Attached {sdk.path.name(target)} ({modality}) to your next "
            "message. Look at it, then continue."
        ))

    def _parse(self, sdk, target):
        """Extract text from a file that turned out not to be text.

        No ``record_read`` here: the read-before-edit gate exists so an edit
        replaces text the agent has actually seen, and extracted text is not
        what is on disk. Editing a PDF by string replacement is not a thing
        the gate should bless.
        """
        name = sdk.path.name(target)
        suffix = sdk.path.suffix(target) or "this type"
        try:
            text = sdk.parse.file(target)
        except sdk.Failed as failed:
            return sdk.fail(
                f"{name} is not a text file, and it could not be parsed into "
                f"text: {failed.error}. Installing a parser package for "
                f"{suffix} may help — see /packages."
            )

        text = str(text or "").strip()
        if not text:
            return sdk.fail(
                f"{name} is not a text file, and parsing it produced no text. "
                "It may be empty, or hold only content the parser for "
                f"{suffix} does not extract."
            )

        note = ""
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            note = f"\n\n... (output capped at {MAX_CHARS} chars)"

        return sdk.ok(None, llm_summary=(
            f"{name} is not text; extracted its contents instead:"
            f"\n\n{text}{note}"
        ))
