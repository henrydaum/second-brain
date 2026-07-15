"""Small text-file CRUD tool for repo-native editing."""

dependencies_files = ['tools/helpers/file_reads.py']
dependencies_pip = []

import re
from difflib import SequenceMatcher
from pathlib import Path

from paths import DATA_DIR, ROOT_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers.plugin_paths import iter_plugin_dirs
from .helpers import file_reads

ROOTS = tuple(p.resolve() for p in (ROOT_DIR, DATA_DIR))
ROOT_WARNING = "WARNING: this edits a source-controlled ROOT file."
PLUGIN_EDIT_REMINDER = " You edited or created a plugin file. Use test_plugin(plugin_path=...) to make sure it is correct."
READ_FIRST = "Read the file with read_file before editing it."
STALE_READ = "File changed on disk since it was last read — re-read it with read_file."

# Self-correcting no-match errors: quote the closest real text so the model
# can fix its old_text in one step instead of re-reading the whole file.
NEAREST_MIN_RATIO = 0.4
NEAREST_MAX_LINES = 10_000   # skip the scan on huge files
NEAREST_QUOTE_CAP = 1200
LINE_PREFIX_RE = re.compile(r"\s*\d+:\s")
LINE_PREFIX_HINT = (
    " Your old_text looks like it includes read_file's 'N:' line-number "
    "prefixes — strip them, or call read_file with line_numbers=false."
)


def _path(raw: str) -> tuple[Path | None, str | None]:
    """Internal helper to handle path."""
    raw = (raw or "").strip()
    if not raw:
        return None, "path is required."
    p = Path(raw)
    p = (p if p.is_absolute() else ROOT_DIR / p).resolve()
    return (p, None) if any(p == r or r in p.parents for r in ROOTS) else (None, f"Path is outside allowed roots: {p}")


class EditFile(BaseTool):
    """Edit file."""
    name = "edit_file"
    description = (
        "Create, overwrite, exact-replace, append to, or delete a UTF-8 text file. "
        "Editing an existing file REQUIRES reading it with read_file first in this "
        "conversation (create is exempt). For replace, old_text must match the raw "
        "file exactly — read with line_numbers=false when copying text to replace. "
        "Paths may be absolute or relative to the project root; edits are limited "
        "to the project root and Second Brain data directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["create", "overwrite", "replace", "append", "delete"], "description": "File operation to perform."},
            "path": {"type": "string", "description": "Target file path."},
            "content": {"type": "string", "description": "Text for create, overwrite, or append."},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring exactly one match."},
            "justification": {"type": "string", "description": "Short plain-English reason for the edit, shown in the approval dialog."},
        },
        "required": ["operation", "path", "justification"],
    }
    requires_services = []
    max_calls = 20
    background_safe = False
    plan_mode_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        """Run edit file."""
        op = (kwargs.get("operation") or "").strip().lower()
        justification = (kwargs.get("justification") or "").strip()
        p, err = _path(kwargs.get("path", ""))
        if err:
            return ToolResult.failed(err)
        if not justification:
            return ToolResult.failed("A justification is required for every edit.")

        def approve(extra: str = "") -> ToolResult | None:
            """Approve edit file."""
            if context.approve_command is None:
                return ToolResult.failed("File editing is not available — no approval handler is configured.")
            try:
                warning = f"\n\n{ROOT_WARNING}" if _is_root_file(p) else ""
                ok = context.approve_command(f"edit_file {op} {p}", f"{justification}{warning}\n\npath: {p}{extra}".strip())
            except Exception as e:
                return ToolResult.failed(f"Approval dialog error: {e}")
            if ok:
                return None
            return ToolResult.failed(
                getattr(context, "approval_denial_reason", "")
                or "Edit denied by user. STOP — do not retry this edit. Ask the user what they would like you to do instead."
            )

        try:
            if op == "delete":
                if not p.exists():
                    return ToolResult.failed(f"File not found: {p}")
                if not p.is_file():
                    return ToolResult.failed(f"Not a file: {p}")
                state = file_reads.check(context, p)
                if state == file_reads.UNREAD:
                    return ToolResult.failed(READ_FIRST)
                if state == file_reads.STALE:
                    return ToolResult.failed(STALE_READ)
                denied = approve()
                if denied:
                    return denied
                p.unlink()
                file_reads.forget(context, p)
                return ToolResult(data={"path": str(p), "operation": op}, llm_summary=f"Deleted {p}.")
            if op in {"create", "overwrite", "append"}:
                text = kwargs.get("content")
                if text is None:
                    return ToolResult.failed("content is required for create, overwrite, and append.")
                if op == "create" and p.exists():
                    return ToolResult.failed(f"File already exists: {p}")
                if op != "create" and p.exists():
                    state = file_reads.check(context, p)
                    if state == file_reads.UNREAD:
                        return ToolResult.failed(READ_FIRST)
                    if state == file_reads.STALE:
                        return ToolResult.failed(STALE_READ)
                denied = approve(f"\nchars: {len(text)}")
                if denied:
                    return denied
                p.parent.mkdir(parents=True, exist_ok=True)
                prior = p.read_text(encoding="utf-8") if op == "append" and p.exists() else ""
                p.write_text((prior + text) if op == "append" else text, encoding="utf-8")
                file_reads.record_read(context, p)
                verb = {"create": "Created", "overwrite": "Overwrote", "append": "Appended to"}[op]
                return ToolResult(data={"path": str(p), "operation": op}, llm_summary=f"{verb} {p}.{_plugin_edit_reminder(p)}")
            if op == "replace":
                old, new = kwargs.get("old_text"), kwargs.get("new_text")
                if old in (None, ""):
                    return ToolResult.failed("old_text is required for replace.")
                if new is None:
                    return ToolResult.failed("new_text is required for replace.")
                if not p.is_file():
                    return ToolResult.failed(f"File not found: {p}")
                state = file_reads.check(context, p)
                if state == file_reads.UNREAD:
                    return ToolResult.failed(READ_FIRST)
                text = p.read_text(encoding="utf-8")
                count = text.count(old)
                if count == 0:
                    return ToolResult.failed(_not_found_error(text, old))
                if state == file_reads.STALE and count != 1:
                    # A single exact match is safe evidence even on a changed
                    # file; anything else needs a fresh read.
                    return ToolResult.failed(STALE_READ)
                if count > 1 and not kwargs.get("replace_all"):
                    return ToolResult.failed(
                        f"old_text appears {count} times (lines {_occurrence_lines(text, old)}); "
                        "pass replace_all=true or make it unique.")
                replacements = count if kwargs.get("replace_all") else 1
                denied = approve(f"\nreplacements: {replacements}")
                if denied:
                    return denied
                p.write_text(text.replace(old, new, -1 if kwargs.get("replace_all") else 1), encoding="utf-8")
                file_reads.record_read(context, p)
                return ToolResult(data={"path": str(p), "operation": op, "replacements": replacements}, llm_summary=f"Replaced text in {p}.{_plugin_edit_reminder(p)}")
            return ToolResult.failed("operation must be create, overwrite, replace, append, or delete.")
        except UnicodeDecodeError:
            return ToolResult.failed(f"Cannot edit binary or non-UTF-8 file: {p}")
        except Exception as e:
            return ToolResult.failed(f"Edit failed: {e}")


def _not_found_error(text: str, old: str) -> str:
    """Build a self-correcting no-match error.

    Quotes the closest actual region of the file (so the model can fix its
    old_text in one step) and flags read_file line-number contamination.
    """
    msg = "old_text was not found."
    old_lines = old.splitlines()
    numbered = sum(1 for l in old_lines if LINE_PREFIX_RE.match(l))
    if numbered >= 2 and numbered * 2 >= len(old_lines):
        msg += LINE_PREFIX_HINT
    lines = text.splitlines()
    n = len(old_lines)
    if not n or len(lines) > NEAREST_MAX_LINES:
        return msg
    # Slide an n-line window over the file; SequenceMatcher's cheap-ratio
    # cascade keeps full ratio() calls to a handful.
    sm = SequenceMatcher(None, "", old, autojunk=False)
    best_ratio, best_at = 0.0, -1
    for i in range(max(1, len(lines) - n + 1)):
        window = "\n".join(lines[i:i + n])
        sm.set_seq1(window)
        if sm.real_quick_ratio() <= best_ratio or sm.quick_ratio() <= best_ratio:
            continue
        ratio = sm.ratio()
        if ratio > best_ratio:
            best_ratio, best_at = ratio, i
    if best_ratio >= NEAREST_MIN_RATIO and best_at >= 0:
        quote = "\n".join(lines[best_at:best_at + n])[:NEAREST_QUOTE_CAP]
        msg += (f" Closest match (lines {best_at + 1}-{min(best_at + n, len(lines))}):\n"
                f"{quote}")
    return msg


def _occurrence_lines(text: str, old: str, cap: int = 10) -> str:
    """Line numbers of the first ``cap`` occurrences, comma-joined."""
    out, start = [], 0
    while len(out) < cap:
        idx = text.find(old, start)
        if idx == -1:
            break
        out.append(str(text.count("\n", 0, idx) + 1))
        start = idx + 1
    return ", ".join(out)


def _is_root_file(p: Path) -> bool:
    """Return whether root file."""
    root = ROOT_DIR.resolve()
    data = DATA_DIR.resolve()
    return (p == root or root in p.parents) and not (p == data or data in p.parents)


def _plugin_edit_reminder(p: Path) -> str:
    """Internal helper to handle plugin edit reminder."""
    if p.suffix != ".py":
        return ""
    parent = p.parent.resolve()
    return PLUGIN_EDIT_REMINDER if any(parent == d.resolve() for _kind, d in iter_plugin_dirs()) else ""
