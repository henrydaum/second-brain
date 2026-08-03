"""
Run Command tool.

Every command is asked about. That is a deliberate simplification of what
this tool used to be: five hundred lines decomposing compound command lines at
unquoted ``&&``, ``||``, ``;``, ``|`` and newlines, matching each segment
against a read-only whitelist, and sending redirection, command substitution,
backgrounding and unbalanced quotes down the approval path regardless. It
worked, mostly, and "mostly" is the problem — deciding what an arbitrary
command line *does* is undecidable, so a classifier of that shape is a
whitelist racing against quoting forever, and it loses silently: a wrong
"safe" is invisible, only a wrong "unsafe" ever gets reported.

The classifier is gone from here, and none of it is reimplemented. The kernel
classifies ``proc.run`` and ``proc.start`` as unsafe, so the dialog is the
policy's decision rather than this file's opinion about it. Making that less
onerous is a *policy* change (``_SHELL_RECOGNIZERS`` in ``sandbox/policy.py``,
which is where a read-only recognizer or a remembered "yes" belongs) and not a
change here — which is the point of moving it. Authorization does not live in
the code being authorized.

What is left is the useful part: a persistent working directory, the right
Python interpreter, output that spills to a file when it is large, and
background processes for servers and watchers.
"""

dependencies_files = []
dependencies_pip = []
requests = ["proc.run", "proc.start", "proc.status", "proc.stop", "proc.list",
            "fs.temp", "fs.write", "fs.list", "paths.get",
            "session.state_get", "session.state_set"]

import re

from guest.bases import BaseTool

# Per-stream cap before the full text spills to a file.
MAX_STREAM_CHARS = 4000

# A standalone `cd`, and nothing compound: the character class excludes every
# metacharacter, so `cd x && rm -rf y` is not matched and runs as an ordinary
# command (where its `cd` affects only that subshell, as it always has).
_CD = re.compile(r"cd(?:\s+(?P<target>[^&|;<>`$\r\n]+))?")

_SHELLS = ("default", "powershell", "cmd")


def _unquote(text):
    """Strip one matched pair of surrounding quotes."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _roots(sdk):
    """Where commands may run: the project, and the data directory."""
    return [sdk.paths.get("project"), sdk.paths.get("data")]


def _in_bounds(sdk, path):
    """Whether a directory is one this tool will run a command in.

    Advisory, and knowingly so: the command itself can `cd` anywhere once a
    shell is running, and the real limit on that is the approval dialog. This
    only keeps an *accidental* cwd — a typo, a stale sticky directory — from
    quietly running a build in someone's home folder.
    """
    return any(sdk.path.within(path, root) for root in _roots(sdk))


def _resolve(sdk, raw, base=None):
    """Absolute, bounded directory, or ``(None, why not)``."""
    project = sdk.paths.get("project")
    if not raw:
        return project, None
    target = sdk.path.absolute(_unquote(str(raw)), base=base or project)
    if not _in_bounds(sdk, target):
        return None, (f"{target} is outside the project and data directories, "
                      "which is as far as this tool reaches.")
    return target, None


def _is_dir(sdk, path):
    """Whether a path names a directory.

    One Request. ``fs.list`` fails outright on a path that does not exist,
    and a *file* answers for itself — one entry, its own path — so a single
    self-referring entry is the file case and everything else is a directory
    (including an empty one, which answers with nothing).
    """
    try:
        entries = sdk.fs.list(path)
    except sdk.Failed:
        return False
    if len(entries) == 1:
        return sdk.path.normalize(entries[0]) != sdk.path.normalize(path)
    return True


# Session state is one value per namespace, not a keyed store, so the whole
# bag is read and written at once. Unreachable state is not an error: a tool
# driven from a background session has nowhere to keep this, and the honest
# degradation is "no persistent directory", not a refusal.
_STATE = "run_command"


def _sticky(sdk, value=None):
    """Read or move the working directory that persists across calls."""
    if value is not None:
        try:
            sdk.session.state_set({"cwd": value}, namespace=_STATE)
        except sdk.Failed:
            pass
        return value
    try:
        stored = sdk.session.state_get(namespace=_STATE)
    except sdk.Failed:
        return None
    return stored.get("cwd") if isinstance(stored, dict) else None


def _retarget_python(sdk, command):
    """Point ``python`` and ``pip`` at the interpreter hosting the app.

    Otherwise ``pip install x`` installs into whatever is first on PATH, which
    on a machine with several Pythons is reliably the wrong one — the package
    arrives somewhere Second Brain cannot import it from.
    """
    head, _, rest = command.strip().partition(" ")
    executable = sdk.paths.get("python")
    if head in ("pip", "pip3"):
        return f'"{executable}" -m pip {rest}'.strip()
    if head in ("python", "python3"):
        return f'"{executable}" {rest}'.strip()
    return command


def _spill(sdk, header, stdout, stderr):
    """Write the full output somewhere readable, and say where."""
    try:
        path = sdk.fs.temp(suffix=".log")
        sdk.fs.write(path, f"{header}\n\n=== STDOUT ===\n{stdout}\n\n"
                           f"=== STDERR ===\n{stderr}\n")
        return path
    except sdk.Failed:
        return None


def _clip(text, cap=MAX_STREAM_CHARS):
    """Trim to ``cap`` characters. Answers ``(text, was_trimmed)``."""
    text = text or ""
    if len(text) <= cap:
        return text, False
    return text[:cap] + f"\n... (trimmed at {cap} characters)", True


def _table(headers, rows):
    """A markdown table. Frontends render it by policy; the REPL aligns it."""
    lines = ["", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _describe(entry):
    """One background process, as a line of prose."""
    state = ("running" if entry.get("running")
             else f"exited (code {entry.get('code')})")
    label = f" [{entry['label']}]" if entry.get("label") else ""
    return f"#{entry['id']}{label} — {state}: {entry.get('command', '')}"


class RunCommand(BaseTool):
    """Run command."""
    name = "run_command"
    description = (
        "Run a terminal command in the project. Prefer read_file, edit_file and the "
        "retrieval tools for ordinary file work — reach for this when you need a real "
        "shell: builds, tests, git, package installs, servers. Every command is shown "
        "to the user for approval before it runs, so propose what you actually need "
        "and let them decide. Long-running things (servers, watchers) must use "
        "run_in_background=true; poll them with operation='check' and always "
        "operation='stop' when done."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command line to run. Pipes, redirection and && work.",
            },
            "narration": {
                "type": "string",
                "description": "A few words on what you are doing and why, "
                               "shown to the user beside the call. E.g. "
                               "'checking what changed since the last commit'.",
            },
            "timeout": {
                "type": "integer",
                "description": "Seconds to wait before giving up. Default 60, max 600. Ignored for background runs.",
            },
            "cwd": {
                "type": "string",
                "description": "Directory to run in, absolute or relative to the project root. Defaults to the persistent working directory.",
            },
            "shell": {
                "type": "string",
                "enum": ["default", "powershell", "cmd"],
                "description": "Which shell parses the command. Defaults to the platform's.",
            },
            "run_in_background": {
                "type": "boolean",
                "description": "Start it and return immediately with a process id. For servers and long jobs only.",
            },
            "operation": {
                "type": "string",
                "enum": ["run", "check", "stop", "list"],
                "description": "run (default) executes 'command'. check/stop take process_id; list shows every tracked process.",
            },
            "process_id": {
                "type": "integer",
                "description": "Which background process check/stop refers to.",
            },
        },
        "required": [],
    }
    requires_services = []
    agent_prompt = (
        "## Running shell commands\n"
        "run_command runs a real shell, scoped to the project root and the Second Brain "
        "data directory. **Every command pauses for the user's approval**, so there is no "
        "list of blessed commands to memorise and no reason to phrase a command "
        "defensively — ask for what you actually need, including package installs, and the "
        "user decides. A denial is an answer: stop and ask what they would prefer rather "
        "than retrying a variant.\n"
        "Because each call costs the user a decision, prefer read_file / edit_file / the "
        "search tools for ordinary file work, and batch shell work into one command line "
        "rather than three round trips.\n"
        "**Anything expressible in Python should be a script, not a command** — see the "
        "Scripts section, which explains why and where.\n"
        "`python` and `pip` are rewritten to the interpreter running Second Brain, so "
        "`pip install x` always lands in the right environment.\n"
        "The working directory persists for the conversation: a standalone `cd <dir>` moves "
        "it for every later call and needs no approval (it starts no process); bare `cd` "
        "resets to the project root. A `cd` inside a compound command affects only that "
        "subshell, as always.\n"
        "Large output is trimmed inline and written in full to a temp file whose path is "
        "returned — read_file that path when you need the rest.\n"
        "For servers, watchers and anything that does not end on its own, pass "
        "run_in_background=true: you get a process id back immediately and keep working. "
        "Poll with operation='check', survey with operation='list', and ALWAYS "
        "operation='stop' when the task is finished — stopping needs no approval. The "
        "registry is in memory: if Second Brain restarts, anything still running is "
        "orphaned rather than killed, so leaving servers up is a real cost."
    )

    def run(self, sdk, **kwargs):
        """Run, or speak about, a command."""
        operation = (kwargs.get("operation") or "run").strip().lower()
        if operation == "list":
            return self._list(sdk)
        if operation in ("check", "stop"):
            return self._one(sdk, operation, kwargs.get("process_id"))
        if operation != "run":
            return sdk.fail("operation must be run, check, stop, or list.")
        return self._run(sdk, **kwargs)

    # ── background processes ──────────────────────────────────────

    def _list(self, sdk):
        """Everything still tracked."""
        entries = sdk.proc.list() or []
        if not entries:
            return sdk.ok({"processes": []},
                          llm_summary="No background processes are running.")
        rows = [[str(entry["id"]), "yes" if entry.get("running") else "no",
                 entry.get("label") or "",
                 (entry.get("command") or "")[:60]] for entry in entries]
        return sdk.ok({"processes": entries},
                      llm_summary=_table(["id", "running", "label", "command"],
                                         rows))

    def _one(self, sdk, operation, process_id):
        """Check on, or stop, one background process."""
        if not isinstance(process_id, int):
            return sdk.fail(f"{operation} needs an integer process_id — "
                            "use operation='list' to see them.")
        try:
            if operation == "stop":
                stopped = sdk.proc.stop(process_id)
                return sdk.ok(stopped,
                              llm_summary=f"Stopped process #{process_id}. "
                                          f"Its log is at {stopped.get('log')}.")
            status = sdk.proc.status(process_id)
        except sdk.Failed as failed:
            return sdk.fail(str(failed.error))
        return sdk.ok(status,
                      llm_summary=(f"{_describe(status)}\n"
                                   f"Recent output:\n{status.get('output', '')}\n"
                                   f"(full log: {status.get('log')})"))

    # ── running one ───────────────────────────────────────────────

    def _run(self, sdk, **kwargs):
        """The ordinary case."""
        command = (kwargs.get("command") or "").strip()
        if not command:
            return sdk.fail("No command provided.")
        shell = (kwargs.get("shell") or "default").strip().lower()
        if shell not in _SHELLS:
            return sdk.fail(f"shell must be one of {list(_SHELLS)}.")
        try:
            timeout = min(max(int(kwargs.get("timeout") or 60), 5), 600)
        except (TypeError, ValueError):
            timeout = 60

        # Explicit cwd wins, then the sticky one, then the project root. An
        # explicit cwd also re-pins the sticky directory, so "work over here
        # now" only needs saying once.
        explicit = (kwargs.get("cwd") or "").strip()
        cwd, why_not = _resolve(sdk, explicit or _sticky(sdk))
        if why_not:
            return sdk.fail(why_not)

        # A standalone `cd` moves the persistent directory. No process starts,
        # so there is nothing to approve — and running it in a subshell would
        # discard the effect anyway, which is what makes it worth intercepting.
        moved = _CD.fullmatch(command)
        if moved is not None:
            return self._cd(sdk, (moved.group("target") or "").strip(), cwd)

        if explicit:
            _sticky(sdk, cwd)

        resolved = _retarget_python(sdk, command)
        if kwargs.get("run_in_background"):
            return self._background(sdk, resolved, cwd, shell)
        return self._foreground(sdk, resolved, cwd, shell, timeout)

    def _cd(self, sdk, target, cwd):
        """Move the persistent working directory."""
        if not target:
            moved = sdk.paths.get("project")
        else:
            moved, why_not = _resolve(sdk, target, base=cwd)
            if why_not:
                return sdk.fail(why_not)
            if not _is_dir(sdk, moved):
                return sdk.fail(f"Not a directory: {moved}")
        _sticky(sdk, moved)
        return sdk.ok({"cwd": moved},
                      llm_summary=f"Working directory is now {moved}. "
                                  "It persists for later run_command calls.")

    def _background(self, sdk, command, cwd, shell):
        """Start something and leave it running.

        No ``label``: it used to carry the retired ``justification``, and the
        registry records the command line beside it anyway, so the label only
        ever restated what ``proc.list`` was already showing. ``narration``
        cannot replace it — the kernel strips that before this method runs.
        """
        try:
            started = sdk.proc.start(command, cwd=cwd, shell=shell)
        except sdk.Denied as refused:
            return self._denied(sdk, refused)
        except sdk.Failed as failed:
            return sdk.fail(f"Could not start: {failed.error}")
        return sdk.ok(started,
                      llm_summary=(
                          f"Started process #{started['id']} (pid "
                          f"{started['pid']}): {command}\n"
                          f"Output is being written to {started['log']}. Poll it "
                          f"with operation='check' and process_id="
                          f"{started['id']}, and stop it when you are done."))

    def _foreground(self, sdk, command, cwd, shell, timeout):
        """Run to completion and report what it printed."""
        try:
            done = sdk.proc.run(command, timeout=timeout, cwd=cwd, shell=shell)
        except sdk.Denied as refused:
            return self._denied(sdk, refused)
        except sdk.Failed as failed:
            return sdk.fail(f"Command failed to run: {failed.error}")

        stdout, stderr = done.get("stdout", ""), done.get("stderr", "")
        shown_out, trimmed_out = _clip(stdout)
        shown_err, trimmed_err = _clip(stderr)
        spilled = None
        if trimmed_out or trimmed_err:
            spilled = _spill(sdk, f"$ {command}\n# cwd: {cwd}", stdout, stderr)

        parts = []
        if shown_out:
            parts.append(shown_out)
        if shown_err:
            parts.append(f"STDERR:\n{shown_err}")
        if done.get("code"):
            parts.append(f"(exit code {done['code']})")
        if spilled:
            parts.append(f"(full output written to {spilled})")
        if cwd != sdk.paths.get("project"):
            parts.append(f"(cwd: {cwd})")

        return sdk.ok({"stdout": stdout, "stderr": stderr,
                       "code": done.get("code"), "spill_path": spilled,
                       "cwd": cwd, "shell": shell},
                      llm_summary="\n".join(parts) if parts else "(no output)")

    def _denied(self, sdk, refused):
        """A refusal is an answer, and retrying a variant is not the reply."""
        return sdk.fail(
            f"{refused}\nThe user declined this command. Do not retry it or a "
            "variation of it — ask them what they would like you to do instead.")
