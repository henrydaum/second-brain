"""Slash command plugin for `/debug`."""

from guest.bases import BaseCommand


class DebugCommand(BaseCommand):
    """Read-only inspection of the active session and recent log errors."""

    name = "debug"
    description = "Inspect the live session, model, and recent log errors"
    category = "System"
    requests = ["session.get", "paths.get", "fs.list", "fs.read", "llm.list",
                "config.read"]

    def run(self, sdk, args):
        """Execute `/debug` for the active session."""
        data_dir = sdk.paths.get("data")
        log_path = _join(data_dir, "app.log")
        return (
            "**Session**\n```\n"
            + _state_section(sdk)
            + "\n```\n\n"
            + "**Recent log warnings/errors**\n```\n"
            + "\n".join(_log_lines(sdk, data_dir, log_path))
            + "\n```"
        )


def _state_section(sdk):
    """What is actually driving this session right now.

    ``Phase`` and ``Attended`` used to head this list and neither carried any
    information: ``/debug`` is a command, so the phase is always
    ``calling_command`` and somebody is always present to have typed it. What
    a person opening this actually wants is which model and which profile are
    behind the next turn — neither of which was here.
    """
    session = sdk.session.get(details=True)
    if session is None or session.get("debug") is None:
        return "(no active session)"

    parts = [
        f"Conversation: {session.get('conversation_id')}",
        f"Frontend: {session.get('frontend') or 'unknown'}",
        f"User: {session.get('user_id')}",
        f"Agent profile: {session.get('agent_profile') or 'default'}",
        f"Model: {_model(sdk)}",
    ]
    flags = session["debug"]["service_flags"]
    if flags:
        parts.append("Services: " + ", ".join(flag for flag in flags if flag))
    if session["busy"]:
        parts.append("Turn: agent turn in progress")
    return "\n".join(
        line for block in parts for line in block.splitlines())


def _model(sdk):
    """The default profile, its backend, and whether its pool is open."""
    try:
        registry = sdk.llm.list() or {}
    except sdk.Failed:
        return "unavailable"
    name = registry.get("default") or ""
    if not name:
        return "none configured — run /setup"
    row = next((entry for entry in registry.get("profiles") or []
                if entry.get("model_name") == name), {})
    backend = (registry.get("aliases") or {}).get(
        row.get("class", ""), row.get("class", ""))
    label = next((entry.get("display_name") for entry in
                  registry.get("backends") or []
                  if entry.get("name") == backend), backend)
    state = "loaded" if row.get("loaded") else "not loaded"
    return f"{name} via {label or 'no backend'} ({state})"


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
