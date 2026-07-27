"""Slash command plugin for `/debug`."""

from guest.bases import BaseCommand


class DebugCommand(BaseCommand):
    """Read-only inspection of the active session and recent log errors."""

    name = "debug"
    description = "Inspect the live conversation state machine and recent log errors"
    category = "System"
    requests = ["session.get", "paths.get", "fs.list", "fs.read"]

    def run(self, sdk, args):
        """Execute `/debug` for the active session."""
        data_dir = sdk.paths.get("data")
        log_path = _join(data_dir, "app.log")
        return (
            "**Conversation state**\n```\n"
            + _state_section(sdk)
            + "\n```\n\n"
            + "**Recent log warnings/errors**\n```\n"
            + "\n".join(_log_lines(sdk, data_dir, log_path))
            + "\n```"
        )


def _state_section(sdk):
    """Return the active session's state-machine snapshot."""
    session = sdk.session.get(details=True)
    if session is None or session.get("debug") is None:
        return "(no active session)"

    debug = session["debug"]
    parts = [debug["state"]]
    flags = debug["service_flags"]
    if flags:
        parts.append("Session: " + ", ".join(flag for flag in flags if flag))
    if session["busy"]:
        parts.append("Session: agent turn in progress")
    parts.append(debug["recent_events"])
    return "\n".join(
        line for block in parts for line in block.splitlines())


def _join(root, name):
    """Join an application path without consulting the guest environment."""
    separator = "\\" if "\\" in root else "/"
    return root.rstrip("/\\") + separator + name


def _log_lines(sdk, data_dir, path, limit=10):
    """Return recent warning/error/critical log lines."""
    if path not in sdk.fs.list(data_dir, pattern="app.log"):
        return [f"No log file found at {path}."]
    hits = [
        line.strip()
        for line in sdk.fs.read(path).splitlines()
        if (
            " | WARNING | " in line
            or " | ERROR | " in line
            or " | CRITICAL | " in line
        )
    ]
    return hits[-limit:] or ["No warnings or errors in this run."]
