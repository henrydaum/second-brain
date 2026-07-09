"""
Run Command tool.

Read-only commands — including pipelines built entirely from read-only
commands — run directly; anything that can change state requires active user
approval before execution.

The classifier is modeled on the safe-command lists in Claude Code and OpenAI
Codex: a compound command is decomposed at unquoted `&&`, `||`, `;`, `|`, and
newlines, and auto-runs only when *every* segment is read-only. Redirection,
command substitution, backgrounding, and unbalanced quotes always take the
approval path, because the string is handed to a real shell and those forms
can smuggle an unvetted command past the whitelist.

  rg/grep/findstr/find, ls/dir/tree, cat/type, head/tail/wc/sort/...,
  git status/log/diff/show/... , pip list/show/freeze, version checks
                                 — auto-approved, project-scoped
  pipelines/chains of the above  — auto-approved
  pip install/uninstall          — requires user approval
  everything else                — requires user approval
"""

dependencies_files = []
dependencies_pip = []

import logging
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from paths import ROOT_DIR, DATA_DIR

logger = logging.getLogger("RunCommand")

# Per-stream truncation cap before spilling full output to a temp file.
_OUTPUT_CHAR_CAP = 4000


def _truncate_stream(label: str, text: str, cap: int = _OUTPUT_CHAR_CAP) -> tuple[str, bool]:
    """Internal helper to handle truncate stream."""
    if len(text) <= cap:
        return text, False
    head = text[: cap // 2]
    tail = text[-cap // 2 :]
    return f"{head}\n... [{label} truncated, {len(text)} chars total] ...\n{tail}", True


# ── Whitelist configuration ──────────────────────────────────────────

# Read-only inspectors and pure filters (the Codex/Claude Code safelists plus
# Windows equivalents). A compound command auto-runs only when every pipeline
# segment resolves to one of these families.
_READ_ONLY_COMMANDS = {
    # POSIX-ish
    "basename", "cat", "cut", "dirname", "du", "echo", "file", "grep",
    "head", "ls", "nl", "pwd", "realpath", "sort", "stat", "tail", "tr",
    "uniq", "wc", "which", "whoami",
    # Windows
    "dir", "findstr", "tree", "type", "ver", "vol", "where",
    # Guarded below — specific flags force the approval path.
    "find", "rg",
}
_SEARCH_COMMANDS = {"find", "findstr", "grep", "rg", "sed", "select-string", "sls"}

# PowerShell read-only cmdlets + canonical aliases. Only reachable once the
# PowerShell structural guard passes (no (, {, $, ` outside single quotes —
# those introduce subexpressions/scriptblocks that execute code).
_PS_READ_ONLY = {
    "get-childitem", "gci", "get-content", "gc", "get-item", "get-location",
    "gl", "get-command", "gcm", "get-date", "select-string", "sls", "test-path",
}
_PS_STRUCTURAL_RE = re.compile(r"[(){}$`]")

# `find` options that delete, write files, or execute commands.
_UNSAFE_FIND_OPTIONS = {
    "-exec", "-execdir", "-ok", "-okdir", "-delete",
    "-fls", "-fprint", "-fprint0", "-fprintf",
}

# `rg` options that execute a helper command or decompression tool.
_UNSAFE_RG_OPTIONS = {"--pre", "--hostname-bin", "-z", "--search-zip"}

# Git subcommands that only inspect state.
_GIT_READ_ONLY = {
    "blame", "branch", "cat-file", "describe", "diff", "grep", "log",
    "ls-files", "ls-tree", "rev-parse", "shortlog", "show", "show-ref",
    "status",
}

# Options that make an otherwise read-only git subcommand write or execute
# (`git log --output=f`, `git diff --ext-diff`, `git grep -O<pager>`).
_UNSAFE_GIT_OPTIONS = {"--output", "--ext-diff", "--textconv", "--exec", "-O", "--open-files-in-pager"}

# `git branch` is read-only only with listing flags; any other flag or a
# positional argument may create, rename, or delete a branch.
_GIT_BRANCH_LIST_FLAGS = {
    "--list", "-l", "--show-current", "-a", "--all", "-r", "--remotes",
    "-v", "-vv", "--verbose",
}

# Pip subcommands that are read-only (no approval needed)
_PIP_READ_ONLY = {"list", "show", "freeze", "check", "--version"}

# Pip subcommands that modify the environment (need user approval)
_PIP_MODIFYING = {"install", "uninstall"}

# Bare version/help checks for well-known executables.
_VERSION_SAFE_BASES = {"git", "node", "npm", "pip", "pip3", "python", "python3", "rg", "uv"}
_VERSION_FLAGS = {"--version", "-V", "--help", "-h"}  # case-sensitive: python -v opens a REPL

# `sed -n <N[,M]>p [file]` — the one sed form that is purely a range print.
_SED_N_RE = re.compile(r"^\d+(,\d+)?p$")

# Directories the agent is allowed to target
_ALLOWED_ROOTS = {Path(ROOT_DIR).resolve(), Path(DATA_DIR).resolve()}


# ── Command decomposition ────────────────────────────────────────────

# Redirections that discard a stream (to the null device) or duplicate one fd
# onto another (2>&1) write nothing durable, so they don't disqualify a
# segment — strip them before scanning. Any other redirection still forces
# the approval path.
_NULL_REDIRECT_RE = re.compile(
    r"(?:^|(?<=\s))[&\d]?>{1,2}\s*(?:/dev/null|nul)(?=\s|$)|\d?>&\d",
    re.IGNORECASE,
)

def _split_segments(command: str) -> list[str] | None:
    """Split ``command`` into pipeline segments at unquoted ``&&``, ``||``,
    ``;``, ``|``, and newlines.

    Returns None — forcing the approval path — when the command contains a
    construct the per-segment classifier cannot vouch for: redirection
    (``>``, ``<``), command substitution (`` ` ``, ``$(``), backgrounding
    (a lone ``&``), an unbalanced quote (the rest of the string can hide an
    operator), or an empty segment (a dangling operator).
    """
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "`" or ch in ("<", ">"):
            return None
        if ch == "$" and command[i : i + 2] == "$(":
            return None
        if ch == "&":
            if command[i : i + 2] == "&&":
                segments.append("".join(buf))
                buf = []
                i += 2
                continue
            return None  # background / cmd single-& chaining
        if ch == "|":
            step = 2 if command[i : i + 2] == "||" else 1
            segments.append("".join(buf))
            buf = []
            i += step
            continue
        if ch in (";", "\n"):
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if quote:
        return None
    segments.append("".join(buf))
    segments = [s.strip() for s in segments]
    if not segments or any(not s for s in segments):
        return None
    return segments


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_command(command: str) -> tuple[str, list[str]]:
    """Extract (base_command, tokens) from a shell command string."""
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return "", []
    return tokens[0].strip("\"'").lower(), tokens


def _unquote(token: str) -> str:
    """Strip one layer of surrounding quotes (shlex posix=False keeps them)."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]
    return token


def _is_pip_command(tokens: list[str]) -> tuple[bool, str | None]:
    """Check if this is a pip command. Returns (is_pip, subcommand)."""
    if not tokens:
        return False, None

    base = tokens[0].strip("\"'").lower()

    # Direct pip call: pip install ...
    if base in ("pip", "pip3"):
        sub = tokens[1].lower() if len(tokens) > 1 else None
        return True, sub

    # python -m pip install ...
    if base in ("python", "python3") and len(tokens) >= 3:
        if tokens[1] == "-m" and tokens[2].lower() in ("pip", "pip3"):
            sub = tokens[3].lower() if len(tokens) > 3 else None
            return True, sub

    return False, None


def _paths_in_bounds(tokens: list[str]) -> bool:
    """True when no token targets a path outside the allowed roots.

    Absolute paths must resolve under a root; a ``..`` path component can
    climb out of the (in-bounds) cwd, so it also fails the check. Only gates
    auto-approval — the user can still approve the command as-is.
    """
    for token in tokens[1:]:  # skip the command itself
        if token.startswith("-"):
            continue
        raw = _unquote(token)
        looks_pathish = "/" in raw or "\\" in raw or raw == ".."
        if looks_pathish and ".." in raw.replace("\\", "/").split("/"):
            return False
        p = Path(raw)
        if p.is_absolute():
            resolved = p.resolve()
            if not any(resolved == root or root in resolved.parents for root in _ALLOWED_ROOTS):
                return False
    return True


def _is_safe_sed(tokens: list[str]) -> bool:
    """True only for ``sed -n <N[,M]>p [file]`` — a pure range print."""
    if len(tokens) not in (3, 4) or tokens[1] != "-n":
        return False
    return bool(_SED_N_RE.match(_unquote(tokens[2])))


def _git_option_unsafe(arg: str) -> bool:
    """Internal helper: does this git argument write or execute?"""
    if arg in _UNSAFE_GIT_OPTIONS:
        return True
    if arg.startswith(("--output=", "--exec=", "--open-files-in-pager=")):
        return True
    return arg.startswith("-O") and len(arg) > 2  # -O<pager> inline form


def _git_branch_read_only(args: list[str]) -> bool:
    """`git branch` with no args lists; otherwise every arg must be a listing
    flag (Codex's rule — positionals may create/rename/delete)."""
    return all(a in _GIT_BRANCH_LIST_FLAGS or a.startswith("--format=") for a in args)


def _rewrite_for_current_python(command: str) -> str:
    """Rewrite python/pip commands to use the running interpreter.

    Ensures 'pip install foo' becomes '"/path/to/python" -m pip install foo',
    so commands always target the same environment that is hosting the app —
    whether that's a system Python on Windows or a .venv on Mac.
    """
    base, tokens = _parse_command(command)
    if not tokens:
        return command

    py = sys.executable  # always the right interpreter

    # pip ... / pip3 ... → python -m pip ...
    if base in ("pip", "pip3"):
        return f'"{py}" -m pip ' + " ".join(tokens[1:])

    # python -m pip ... / python3 -m pip ...
    if base in ("python", "python3"):
        return f'"{py}" ' + " ".join(tokens[1:])

    return command


def _resolve_cwd(raw: str | None) -> tuple[Path | None, str | None]:
    """Internal helper to resolve cwd."""
    cwd = Path((raw or "").strip()) if raw else Path(ROOT_DIR)
    cwd = (cwd if cwd.is_absolute() else ROOT_DIR / cwd).resolve()
    return (cwd, None) if any(cwd == root or root in cwd.parents for root in _ALLOWED_ROOTS) else (None, f"cwd is outside the allowed roots: {cwd}")


def _classify_single(segment: str) -> tuple[str, bool, str | None]:
    """Classify one pipeline segment. Same contract as _classify."""
    base, tokens = _parse_command(segment)
    if not base:
        return "blocked", False, "Empty command."

    # Version/help checks for well-known executables: `python --version` etc.
    if base in _VERSION_SAFE_BASES and len(tokens) > 1 \
            and all(t in _VERSION_FLAGS for t in tokens[1:]):
        return "version", False, None

    # Pip commands
    is_pip, sub = _is_pip_command(tokens)
    if is_pip:
        if sub is None or sub in _PIP_READ_ONLY:
            return "pip_read", False, None  # bare "pip" prints help
        if sub in _PIP_MODIFYING:
            return "pip_modify", True, None
        return "shell", True, None  # download/config/wheel/... — user decides

    if base == "git":
        if len(tokens) == 1:
            return "git_read", False, None  # bare git prints help
        sub = tokens[1].lower()
        if sub.startswith("-"):
            # Global options (-C, -c, --git-dir, --exec-path, ...) can point
            # git elsewhere or inject config — approval path.
            return "shell", True, None
        args = tokens[2:]
        if sub not in _GIT_READ_ONLY or any(_git_option_unsafe(a) for a in args):
            return "shell", True, None
        if sub == "branch" and not _git_branch_read_only(args):
            return "shell", True, None
        if not _paths_in_bounds(tokens):
            return "shell", True, None
        return "git_read", False, None

    if base == "sed":
        return ("search", False, None) if _is_safe_sed(tokens) else ("shell", True, None)

    if base in _READ_ONLY_COMMANDS or base in _PS_READ_ONLY:
        args = [_unquote(t) for t in tokens[1:]]
        if base == "find" and any(a in _UNSAFE_FIND_OPTIONS for a in args):
            return "shell", True, None
        if base == "rg" and any(
            a in _UNSAFE_RG_OPTIONS or a.startswith(("--pre=", "--hostname-bin="))
            for a in args
        ):
            return "shell", True, None
        if base == "sort" and any(a.startswith(("-o", "--output")) for a in args):
            return "shell", True, None  # sort -o writes a file
        if not _paths_in_bounds(tokens):
            return "shell", True, None
        return ("search" if base in _SEARCH_COMMANDS else "listing"), False, None

    return "shell", True, None


def _classify(command: str, shell_name: str = "default") -> tuple[str, bool, str | None]:
    """Classify a command.

    Returns:
        (category, needs_approval, error_message)
        - category: "pip_modify", "pip_read", "version", "search", "listing",
          "git_read", "compound_read", "shell", or "blocked"
        - needs_approval: whether to prompt the user
        - error_message: if blocked, a helpful message; otherwise None
    """
    stripped = command.strip().rstrip(";").strip()
    if not stripped:
        return "blocked", False, "Empty command."

    # PowerShell evaluates (...) subexpressions, {...} scriptblocks, $ variables
    # and `-escapes anywhere outside single quotes — even as arguments to a
    # read-only cmdlet — so any of those forces the approval path wholesale.
    if shell_name == "powershell" and _PS_STRUCTURAL_RE.search(re.sub(r"'[^']*'", "", stripped)):
        return "shell", True, None

    # Discard-only redirections (2>/dev/null, >NUL, 2>&1) are read-safe;
    # remove them so they neither trip the redirect guard nor look like an
    # out-of-bounds path. Execution still runs the original string.
    segments = _split_segments(_NULL_REDIRECT_RE.sub(" ", stripped))
    if segments is None:
        # Redirection, substitution, backgrounding, or unbalanced quoting —
        # the string goes to a real shell, so it can never be auto-approved.
        return "shell", True, None

    categories = []
    for segment in segments:
        category, needs_approval, error = _classify_single(segment)
        if error:
            return category, False, error
        if needs_approval:
            return category, True, None
        categories.append(category)

    return (categories[0] if len(categories) == 1 else "compound_read"), False, None


class RunCommand(BaseTool):
    """Run command."""
    name = "run_command"
    description = (
        "Run terminal commands from the project root. Prefer read_file/edit_file and "
        "retrieval tools for ordinary file work. Read-only commands — including pipelines "
        "built only from read-only commands — run immediately; package changes and arbitrary "
        "shell commands require active user approval.\n\n"
        "Auto-approved (read-only, project-scoped):\n"
        "- rg / grep / findstr / find — search code (flags that execute or delete still need approval)\n"
        "- dir / ls / tree / cat / type / head / tail / wc / sort / uniq / stat / where / which / pwd\n"
        "- sed -n <N,M>p — print a line range\n"
        "- git status / diff / show / log / blame / branch / ls-files / ls-tree / cat-file / describe / grep\n"
        "- pip list / pip show <pkg> / pip freeze; python --version / pip --version\n"
        "- pipelines and chains (|, &&, ||, ;) where every part is read-only, e.g. `rg foo | head -20`\n\n"
        "Requires approval: pip install/uninstall, any other command, and any redirection "
        "(>, <), command substitution (`, $()), or backgrounding (&) — those run only after "
        "the user approves the exact command. Exception: discard-only redirections "
        "(2>/dev/null, >NUL, 2>&1) are treated as read-safe."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Terminal command to execute. Broad commands require user approval.",
            },
            "justification": {
                "type": "string",
                "description": "Short plain-English reason for running the command.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Maximum seconds to wait. Defaults to 30. "
                    "Use higher values for pip install. Max 600."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory, relative to the project root or absolute under the project/data roots. Defaults to project root.",
            },
            "shell": {
                "type": "string",
                "enum": ["default", "powershell", "cmd"],
                "description": "Shell to use. Defaults to the platform default shell.",
            },
        },
        "required": ["command", "justification"],
    }
    requires_services = []
    max_calls = 10
    background_safe = False
    agent_prompt = (
        "## Running shell commands\n"
        "run_command runs shell commands scoped to the project root and the Second Brain data directory. "
        "Read-only commands run automatically: ls/dir, rg/grep/findstr/find, tree, cat/type, head/tail/wc/sort, "
        "sed -n range prints, git status/log/diff/show/blame/branch/ls-files, pip list/show/freeze, --version "
        "checks — and pipelines or chains (|, &&, ||, ;) composed entirely of such commands, e.g. "
        "`rg foo | head -20`. Everything else — pip install/uninstall, any non-whitelisted command, and any "
        "redirection (>, <; discard-only forms like 2>/dev/null and 2>&1 are fine), command substitution "
        "(`, $()), or backgrounding (&) — pauses for user approval, "
        "so you may freely propose installing a needed package; the user decides whether it runs. "
        "Large output is truncated inline and the full text is written to a temp file whose path is returned "
        "(spill_path) — read_file that path when you need everything."
    )

    def run(self, context, **kwargs) -> ToolResult:
        """Execute `/tool_run_command` for the active session."""
        command = kwargs.get("command", "").strip()
        justification = kwargs.get("justification", "").strip()
        timeout = min(max(int(kwargs.get("timeout", 30)), 5), 600)
        cwd, cwd_err = _resolve_cwd(kwargs.get("cwd"))
        shell_name = (kwargs.get("shell") or "default").strip().lower()

        if not command:
            return ToolResult.failed("No command provided.")
        if not justification:
            return ToolResult.failed("A justification is required for every command.")
        if cwd_err:
            return ToolResult.failed(cwd_err)
        if shell_name not in {"default", "powershell", "cmd"}:
            return ToolResult.failed("shell must be default, powershell, or cmd.")

        # ── Whitelist check ──────────────────────────────────────
        category, needs_approval, error = _classify(command, shell_name)

        if error:
            logger.warning(f"Blocked command: {command} — {category}")
            return ToolResult.failed(error)

        # ── User approval (only for modifying commands) ──────────
        if needs_approval:
            approve_fn = context.approve_command
            if approve_fn is None:
                return ToolResult.failed(
                    "Command execution is not available — no approval handler is configured."
                )
            try:
                approved = approve_fn(command, f"{justification}\n\ncwd: {cwd}\nshell: {shell_name}\ntimeout: {timeout}s")
            except Exception as e:
                logger.error(f"Approval callback failed: {e}")
                return ToolResult.failed(f"Approval dialog error: {e}")

            if not approved:
                return ToolResult.failed(
                    getattr(context, "approval_denial_reason", "")
                    or "Command denied by user. STOP — do not retry this command. "
                    "Ask the user what they would like you to do instead.")

        # ── Execute ──────────────────────────────────────────────
        resolved = _rewrite_for_current_python(command)
        cmd = resolved
        use_shell = True
        if shell_name == "powershell":
            cmd, use_shell = ["powershell", "-NoProfile", "-Command", resolved], False
        elif shell_name == "cmd":
            cmd, use_shell = ["cmd", "/c", resolved], False
        logger.info(f"Running ({category}) in {cwd}: {resolved}")
        try:
            result = subprocess.run(
                cmd,
                shell=use_shell,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(cwd),
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failed(f"Command timed out after {timeout} seconds.")
        except Exception as e:
            return ToolResult.failed(f"Command execution error: {e}")

        # ── Build summary ────────────────────────────────────────
        stdout_view, out_trunc = _truncate_stream("stdout", result.stdout or "")
        stderr_view, err_trunc = _truncate_stream("stderr", result.stderr or "")

        spill_path = None
        if out_trunc or err_trunc:
            try:
                fd, spill_path = tempfile.mkstemp(
                    prefix=f"runcmd-{int(time.time())}-",
                    suffix=".log",
                    dir=str(DATA_DIR),
                )
                with open(fd, "w", encoding="utf-8") as f:
                    f.write(f"$ {resolved}\n# cwd: {cwd}\n\n=== STDOUT ===\n{result.stdout or ''}\n\n=== STDERR ===\n{result.stderr or ''}\n")
            except Exception as e:
                logger.warning(f"Failed to spill full output: {e}")
                spill_path = None

        parts = []
        if stdout_view:
            parts.append(stdout_view)
        if stderr_view:
            parts.append(f"STDERR:\n{stderr_view}")
        if result.returncode != 0:
            parts.append(f"(exit code {result.returncode})")
        if spill_path:
            parts.append(f"(full output written to {spill_path})")

        output = "\n".join(parts) if parts else "(no output)"

        return ToolResult(
            data={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode, "spill_path": spill_path, "cwd": str(cwd), "shell": shell_name, "category": category},
            llm_summary=output,
        )
