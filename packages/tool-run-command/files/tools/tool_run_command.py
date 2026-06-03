"""
Run Command tool.

Small read-only commands run directly; anything broader requires active user
approval before execution.

Allowed commands:
  pip install/uninstall          — requires user approval
  pip list/show/freeze           — auto-approved
  python --version, pip --version — auto-approved
  rg/grep/findstr, dir/ls/tree   — auto-approved, project-scoped
  everything else                — requires user approval
"""

import logging
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

# Commands that are always safe (read-only, no approval needed)
_READ_ONLY_COMMANDS = {"cat", "dir", "findstr", "grep", "ls", "pwd", "rg", "tree", "type"}
_GIT_READ_ONLY = {"branch", "diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}

# Pip subcommands that are read-only (no approval needed)
_PIP_READ_ONLY = {"list", "show", "freeze", "--version"}

# Pip subcommands that modify the environment (need user approval)
_PIP_MODIFYING = {"install", "uninstall"}

# All allowed pip subcommands
_PIP_ALLOWED = _PIP_READ_ONLY | _PIP_MODIFYING

# Version check commands
_VERSION_COMMANDS = {"python --version", "python3 --version", "pip --version"}

# Directories the agent is allowed to target
_ALLOWED_ROOTS = {Path(ROOT_DIR).resolve(), Path(DATA_DIR).resolve()}


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_command(command: str) -> tuple[str, list[str]]:
    """Extract (base_command, tokens) from a shell command string."""
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return "", []
    return tokens[0].lower(), tokens


def _is_pip_command(tokens: list[str]) -> tuple[bool, str | None]:
    """Check if this is a pip command. Returns (is_pip, subcommand)."""
    if not tokens:
        return False, None

    base = tokens[0].lower()

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


def _check_paths_in_bounds(tokens: list[str]) -> str | None:
    """If any token looks like an absolute path, verify it's under an allowed root.
    Returns an error string if out of bounds, None if OK."""
    for token in tokens[1:]:  # skip the command itself
        # Skip flags
        if token.startswith("-"):
            continue
        p = Path(token)
        if p.is_absolute():
            resolved = p.resolve()
            if not any(resolved == root or root in resolved.parents for root in _ALLOWED_ROOTS):
                return (
                    f"Path '{token}' is outside the allowed directories. "
                    f"Commands are scoped to the project root and data directory."
                )
    return None


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


def _classify(command: str) -> tuple[str, bool, str | None]:
    """Classify a command.

    Returns:
        (category, needs_approval, error_message)
        - category: "pip_modify", "pip_read", "version", "search", "listing", "git_read", "shell", or "blocked"
        - needs_approval: whether to prompt the user
        - error_message: if blocked, a helpful message; otherwise None
    """
    stripped = command.strip().lower()

    # Version checks
    if stripped in _VERSION_COMMANDS:
        return "version", False, None

    base, tokens = _parse_command(command)
    if not base:
        return "blocked", False, "Empty command."

    # Pip commands
    is_pip, sub = _is_pip_command(tokens)
    if is_pip:
        if sub is None:
            return "pip_read", False, None  # bare "pip" prints help
        if sub in _PIP_MODIFYING:
            return "pip_modify", True, None
        if sub in _PIP_READ_ONLY:
            return "pip_read", False, None
        return "blocked", False, (
            f"pip {sub} is not allowed. "
            f"Allowed pip subcommands: {', '.join(sorted(_PIP_ALLOWED))}."
        )

    if base == "git":
        sub = tokens[1].lower() if len(tokens) > 1 else "status"
        if sub in _GIT_READ_ONLY:
            path_err = _check_paths_in_bounds(tokens)
            return ("git_read", False, path_err) if path_err else ("git_read", False, None)
        return "shell", True, None

    # Search/list/read commands (project-scoped)
    if base in _READ_ONLY_COMMANDS:
        path_err = _check_paths_in_bounds(tokens)
        if path_err:
            return "blocked", False, path_err
        return "search" if base in ("grep", "findstr", "rg") else "listing", False, None

    return "shell", True, None


class RunCommand(BaseTool):
    """Run command."""
    name = "run_command"
    description = (
        "Run terminal commands from the project root. Prefer read_file/edit_file and "
        "retrieval tools for ordinary file work. Small read-only commands run immediately; package "
        "changes and arbitrary shell commands require active user approval.\n\n"
        "Common commands:\n"
        "- pip install <pkg> / pip uninstall <pkg> — install or remove Python packages (requires user approval)\n"
        "- pip list / pip show <pkg> / pip freeze — check installed packages\n"
        "- python --version / pip --version — check Python environment\n"
        "- rg / grep / findstr — search code within the project directory\n"
        "- dir / ls / tree / cat / type / pwd — inspect project files\n"
        "- git status / diff / show / log / branch / ls-files / grep — inspect git state\n"
        "\n"
        "All other commands are allowed only after the user approves the exact command."
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
        category, needs_approval, error = _classify(command)

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
