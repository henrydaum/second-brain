"""Slash command plugin for `/debug`."""

from pathlib import Path

from paths import DATA_DIR
from plugins.BaseCommand import BaseCommand
from state_machine.debug import format_recent_events, format_state


class DebugCommand(BaseCommand):
    """Slash-command handler for `/debug`.

    Read-only introspection of the active conversation: what the state
    machine currently thinks is happening, plus the tail of recent log
    warnings/errors. Useful when a form, approval, or phase flow gets stuck.
    """
    name = "debug"
    description = "Inspect the live conversation state machine and recent log errors"
    category = "System"

    def run(self, _args, context):
        """Execute `/debug` for the active session."""
        sections = [
            "Conversation state:",
            _state_section(context),
            "",
            "Recent log warnings/errors:",
            *_log_lines(DATA_DIR / "app.log"),
        ]
        return "\n".join(sections)


def _state_section(context) -> str:
    """Return the active session's state-machine snapshot, indented."""
    runtime = getattr(context, "runtime", None)
    session_key = getattr(context, "session_key", None)
    session = (getattr(runtime, "sessions", {}) or {}).get(session_key) if runtime and session_key else None
    cs = getattr(session, "cs", None) if session else None
    if cs is None:
        return "  (no active session)"

    parts = [format_state(cs)]
    flags = [
        flag
        for svc in (getattr(context, "services", None) or {}).values()
        for flag in (svc.debug_flags(session) if callable(getattr(svc, "debug_flags", None)) else [])
    ]
    if flags:
        parts.append("Session: " + ", ".join(f for f in flags if f))
    if getattr(session, "busy", False):
        parts.append("Session: agent turn in progress")
    parts.append(format_recent_events(cs))

    return "\n".join(f"  {line}" for block in parts for line in block.splitlines())


def _log_lines(path: Path, limit: int = 10) -> list[str]:
    """Return recent warning/error/critical log lines, indented."""
    if not path.exists():
        return [f"  No log file found at {path}."]
    hits = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if " | WARNING | " in line or " | ERROR | " in line or " | CRITICAL | " in line
    ]
    return [f"  {line}" for line in hits[-limit:]] or ["  No warnings or errors in this run."]
