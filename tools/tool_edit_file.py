"""Small text-file CRUD tool for repo-native editing.

Sandboxed. The interesting part of this migration is what *left*: this tool
used to run its own approval dialog, with its own path exemptions deciding
when to skip it. That is policy, and policy does not belong in a plugin —
every effect here is now a Request, so ``sandbox/policy.py`` is the single
place that decides, and it already knows that scratch and the agent's own
plugin tree are free while everything else asks. The tool's job is the edit;
the kernel's job is whether.

That deletion is the whole point of the exercise: there is no longer a
question of the native path and the sandboxed path disagreeing, because there
is no native path.
"""

dependencies_files = ['tools/helpers/file_reads.py',
                      'tools/helpers/path_repair.py']
dependencies_pip = []
requests = ["fs.read", "fs.write", "fs.delete", "fs.list", "paths.get",
            "session.state_get", "session.state_set"]

import re
from difflib import SequenceMatcher

from guest.bases import BaseTool

# Flat, not ``from .helpers import file_reads``: a box is one namespace, and
# the kernel puts each declared dependency's *directory* on the box's import
# path. The helper is a sibling here even though it ships in a subdirectory.
from . import file_reads, path_repair

PLUGIN_EDIT_REMINDER = " You edited or created a plugin file. Use validate(path=...) to make sure it is correct."
SCRIPT_EDIT_HINT = (" This is a script, not a plugin: run_script checks it the "
                    "same way validate does and reports the same errors, so "
                    "run it rather than validating it first.")

#: The filename prefixes that make the validator apply the plugin contract,
#: wherever the file sits. Mirrors the roots table in the kernel's ``trees``.
FAMILY_PREFIXES = ("tool_", "task_", "service_", "command_", "frontend_",
                   "parse_", "llm_")
READ_FIRST = "Read the file with read_file before editing it."
STALE_READ = "File changed on disk since it was last read — re-read it with read_file."
DENIED_STOP = (" STOP — do not retry this edit. Ask the user what they would "
               "like you to do instead.")

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


def _resolve(sdk, raw: str):
    """Absolutize a path against the project root. Nothing is out of bounds.

    This used to confine edits to the project and data directories and refuse
    anything else, on the reasoning that narrowing what a tool attempts is
    always safe. It is not, when the narrowing is a *refusal in place of a
    question*: a path elsewhere is unsafe, not forbidden, and the kernel is
    what says so. Refusing here meant the dialog never appeared, so the person
    was never asked and — since the approval dialog is now what adds a folder
    to ``fs_writable_dirs`` — could never grant one either. The tool had made
    itself the last word on a decision it does not own.

    Which is the same lesson this file's docstring already tells about the
    approval dialog it used to run, arriving a second time through a helper
    that looked like mere tidiness.
    """
    raw = (raw or "").strip()
    if not raw:
        return None, "path is required."
    # A shell escape that reached a tool which is not a shell. Repaired here
    # for the same reason read_file repairs it: the backslash is never removed
    # again, so the write lands somewhere nobody asked for or fails outright.
    raw, _ = path_repair.unescape_known_roots(sdk, raw)
    return sdk.path.absolute(raw, base=sdk.paths.get("project")), None


def _stat(sdk, path):
    """The file's entry, or None if it is absent or not a file.

    ``fs.list`` on a file answers for that file alone, which is how a
    sandboxed plugin asks "does this exist, and when did it change?" without
    an existence Request of its own.
    """
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed:
        return None
    for entry in entries or []:
        if not entry.get("is_dir"):
            return entry
    return None


class EditFile(BaseTool):
    """Edit file."""
    name = "edit_file"
    description = (
        "Create, overwrite, exact-replace, append to, or delete a UTF-8 text file. "
        "Editing an existing file REQUIRES reading it with read_file first in this "
        "conversation (create is exempt). For replace, old_text must match the raw "
        "file exactly — read with line_numbers=false when copying text to replace. "
        "Paths may be absolute or relative to the project root; edits are limited "
        "to the project root and Second Brain data directory. Missing parent "
        "folders are created for you, so writing out/report.json into an empty "
        "workspace works without making out/ first."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["create", "overwrite", "replace", "append", "delete"], "description": "File operation to perform."},
            "path": {"type": "string", "description": "Target file path. Parent folders are created as needed — you never need to make a directory first."},
            "content": {"type": "string", "description": "Text for create, overwrite, or append."},
            "old_text": {"type": "string", "description": "Exact text to replace."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring exactly one match."},
            "narration": {"type": "string", "description": "A few words on what you are changing and why, shown to the user beside the call. E.g. 'adding the missing null check'."},
        },
        "required": ["path"],
    }
    requires_services = []

    def run(self, sdk, **kwargs):
        """Run edit file.

        No approval branch anywhere below. ``fs.write`` and ``fs.delete`` are
        classified by the kernel on the way out: free under scratch and the
        agent's own plugin tree, a dialog everywhere else. A refusal comes
        back as ``sdk.Denied``, which is the one case worth translating —
        the model needs to be told to stop rather than retry.
        """
        path, err = _resolve(sdk, kwargs.get("path", ""))
        if err:
            return sdk.fail(err)
        op, err = _operation(sdk, path, kwargs)
        if err:
            return sdk.fail(err)

        try:
            if op == "delete":
                return self._delete(sdk, path)
            if op in {"create", "overwrite", "append"}:
                return self._put(sdk, op, path, kwargs)
            if op == "replace":
                return self._replace(sdk, path, kwargs)
            return sdk.fail("operation must be create, overwrite, replace, "
                            "append, or delete.")
        except sdk.Denied as refused:
            return sdk.fail(f"{refused}{DENIED_STOP}")

    # ── the three shapes ───────────────────────────────────────────

    def _delete(self, sdk, path):
        """Remove a file that has been read this conversation."""
        if _stat(sdk, path) is None:
            return sdk.fail(f"File not found: {path}")
        if (stale := self._staleness(sdk, path)) is not None:
            return stale
        sdk.fs.delete(path)
        file_reads.forget(sdk, path)
        return sdk.ok({"path": path, "operation": "delete"},
                      llm_summary=f"Deleted {path}.")

    def _put(self, sdk, op: str, path, kwargs):
        """create / overwrite / append."""
        text = kwargs.get("content")
        if text is None:
            return sdk.fail(
                "content is required for create, overwrite, and append.")
        exists = _stat(sdk, path) is not None
        if op == "create" and exists:
            return sdk.fail(f"File already exists: {path}")
        if op != "create" and exists:
            if (stale := self._staleness(sdk, path)) is not None:
                return stale

        # Append is a write mode rather than read-concat-write: the handler
        # opens the file in append mode, so this neither loads the prior
        # contents nor races anything between the two halves.
        sdk.fs.write(path, text,
                     mode="append" if op == "append" else "overwrite")
        file_reads.record_read(sdk, path)
        verb = {"create": "Created", "overwrite": "Overwrote",
                "append": "Appended to"}[op]
        return sdk.ok(
            {"path": path, "operation": op},
            llm_summary=f"{verb} {path}.{_plugin_edit_reminder(sdk, path)}")

    def _replace(self, sdk, path, kwargs):
        """Exact-match replacement, with a self-correcting miss."""
        old, new = kwargs.get("old_text"), kwargs.get("new_text")
        if old in (None, ""):
            return sdk.fail("old_text is required for replace.")
        if new is None:
            return sdk.fail("new_text is required for replace.")
        if _stat(sdk, path) is None:
            return sdk.fail(f"File not found: {path}")

        state = file_reads.check(sdk, path)
        if state == file_reads.UNREAD:
            return sdk.fail(READ_FIRST)
        try:
            text = sdk.fs.read(path)
        except sdk.Failed as failed:
            return sdk.fail(f"Could not read {path}: {failed.error}")

        count = text.count(old)
        if count == 0:
            return sdk.fail(_not_found_error(text, old))
        if state == file_reads.STALE and count != 1:
            # A single exact match is safe evidence even on a changed file;
            # anything else needs a fresh read.
            return sdk.fail(STALE_READ)
        if count > 1 and not kwargs.get("replace_all"):
            return sdk.fail(
                f"old_text appears {count} times "
                f"(lines {_occurrence_lines(text, old)}); "
                "pass replace_all=true or make it unique.")

        replacements = count if kwargs.get("replace_all") else 1
        sdk.fs.write(path, text.replace(
            old, new, -1 if kwargs.get("replace_all") else 1))
        file_reads.record_read(sdk, path)
        return sdk.ok(
            {"path": path, "operation": "replace",
             "replacements": replacements},
            llm_summary=f"Replaced text in {path}."
                        f"{_plugin_edit_reminder(sdk, path)}")

    @staticmethod
    def _staleness(sdk, path):
        """Refuse an edit to a file the model has not actually looked at."""
        state = file_reads.check(sdk, path)
        if state == file_reads.UNREAD:
            return sdk.fail(READ_FIRST)
        if state == file_reads.STALE:
            return sdk.fail(STALE_READ)
        return None


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


def _operation(sdk, path, kwargs):
    """The declared operation, or the one the other arguments already imply.

    ``operation`` used to be required, and omitting it was the single largest
    source of failed edits — 102 of them across a 636-trial benchmark, two
    thirds of every ``edit_file`` failure. Each one costs a full model
    round-trip to say something the arguments had already said.

    Inference is deliberately partial, because two of the five verbs are not
    recoverable from the arguments and guessing them destroys work:

    - ``append`` is distinguishable from ``overwrite`` by *nothing* in the
      argument set. Both carry ``content`` and nothing else. Inferring
      ``overwrite`` for an append would silently drop the file's contents, so
      a ``content`` write to a file that already exists still has to ask.
    - ``delete`` has no positive signal at all. It is never inferred; a
      destructive operation should be something the caller said out loud.

    What is left is unambiguous. ``old_text`` means ``replace`` — there is no
    other operation that reads it, and replace still demands an exact match and
    still runs the staleness guard, so a wrong guess here cannot write
    anything. ``content`` aimed at a path that does not exist means ``create``
    — overwrite and append of a missing file both reduce to exactly that.

    An explicit operation is always honoured, including an explicit ``create``
    over an existing file: that is a caller telling us it believes the file is
    new, and disagreeing with it is worth a failure.
    """
    declared = (kwargs.get("operation") or "").strip().lower()
    if declared:
        return declared, None

    if kwargs.get("old_text") not in (None, ""):
        return "replace", None

    if kwargs.get("content") is not None:
        if _stat(sdk, path) is None:
            return "create", None
        return "", (
            f"operation is required when the file already exists: pass "
            f"overwrite to replace all of {path}, or append to add to the "
            "end of it.")

    return "", ("operation is required: pass create, overwrite, replace, "
                "append, or delete. (replace is assumed when you pass "
                "old_text, and create when you pass content for a file that "
                "does not exist yet.)")


def _is_script(sdk, path) -> bool:
    """Whether the kernel will treat this file as a script rather than a plugin.

    Mirrors ``sandbox.isolation.is_script``, which asks two questions: is the
    file in a ``scripts/`` root, and is it *directly* in one. Nesting does not
    count — ``scripts/sub/x.py`` is not a script and ``run_script`` refuses it —
    so comparing the parent directory rather than testing containment is the
    whole point of doing it this way.

    The second half is the validator's rule, not isolation's. A file whose name
    carries a family prefix is checked against the plugin contract wherever it
    sits (``sandbox/validator.py``, ``_check_contract``), so ``scripts/tool_x.py``
    is a plugin as far as the checker is concerned and its author does want the
    nudge — run_script's preflight would reject it for declaring no tool class.
    """
    scripts = sdk.paths.get("scripts")
    if not scripts:
        return False
    if sdk.path.normalize(sdk.path.parent(path)) != sdk.path.normalize(scripts):
        return False
    stem = str(path).replace("\\", "/").rsplit("/", 1)[-1][:-3]
    return not any(stem.startswith(p) for p in FAMILY_PREFIXES)


def _plugin_edit_reminder(sdk, path) -> str:
    """Nudge the author toward the check that actually applies to this file.

    Two different files get two different answers, and conflating them was
    expensive. A **plugin** that fails to load does so silently from here: the
    kernel's adapter returns ``None`` and the reason reaches the app log or a
    notification aimed at the user, never the model. ``validate`` is the only
    in-band way to find out, so the nudge stays.

    A **script** is the opposite case. ``run_script`` runs the identical
    ``validate_file`` in its own preflight and hands back the same findings,
    the same line numbers and the same fix strings — so validating first buys
    nothing that running it would not have told you, and it costs a model
    round-trip to learn it. Measured across a 636-trial benchmark this fired on
    1,291 of 2,608 successful writes and pulled 790 validate calls after it,
    1,100 of them on files in ``scripts/``.

    Checks the tree roots rather than enumerating every family directory: the
    old version imported ``iter_plugin_dirs`` from the kernel, which is the one
    import a sandboxed file may never make. ``scripts`` is exempt from that
    because the kernel states it as a first-class path — where a script goes is
    what decides how it runs, so it is a fact rather than a convention.

    Anything not provably a script keeps the plugin nudge; the failure
    direction is the one that costs a round-trip rather than the one that lets
    a broken plugin through silently.
    """
    if sdk.path.suffix(path) != ".py":
        return ""
    in_a_tree = any(sdk.path.within(path, sdk.paths.get(name))
                    for name in ("workspace", "installed", "bundled"))
    if not in_a_tree:
        return ""
    return SCRIPT_EDIT_HINT if _is_script(sdk, path) else PLUGIN_EDIT_REMINDER
