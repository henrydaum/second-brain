"""Slash command plugin for `/doctor`."""

from pathlib import Path

from paths import DATA_DIR
from plugins.BaseCommand import BaseCommand


class DoctorCommand(BaseCommand):
    """Slash-command handler for `/doctor`."""
    name = "doctor"
    description = "Summarize runtime health, queues, and recent log errors"
    category = "System"

    def run(self, _args, context):
        """Execute `/doctor` for the active session."""
        services, config = getattr(context, "services", {}) or {}, getattr(context, "config", {}) or {}
        orch, db, registry = getattr(context, "orchestrator", None), getattr(context, "db", None), getattr(context, "tool_registry", None)
        stats = db.get_system_stats() if db and hasattr(db, "get_system_stats") else {"files": {}, "tasks": {}}
        run_stats = db.get_run_stats() if db and hasattr(db, "get_run_stats") else {}
        tasks = getattr(orch, "tasks", {}) or {}
        checks = _checks(config, services)
        plan_lines = _plan_lines(context)
        lines = [
            "Doctor:",
            "",
            "Status:",
            f"  Checks: {_count(checks, 'ok')} ok, {_count(checks, 'warn')} warning(s)",
            f"  Services: {_loaded(services)}/{len(services)} loaded",
            f"  Tasks: {len(tasks)} registered, {len(getattr(orch, 'paused', set()) if orch else set())} paused",
            f"  Tools: {len(getattr(registry, 'tools', {}) or {})}",
            f"  Files indexed: {sum((stats.get('files') or {}).values())}",
            *plan_lines,
            "",
            "Findings:",
            *[f"  {_label(k)} {msg}" for k, msg in checks],
            "",
            "Queues:",
            *_task_lines({**(stats.get("tasks") or {}), **run_stats}),
            "",
            "LLM:",
            *_llm_lines(services.get("llm")),
            "",
            "Recent log warnings/errors:",
            *_log_lines(DATA_DIR / "app.log"),
        ]
        return "\n".join(lines)


def _checks(config, services):
    """Return health findings."""
    out = []
    sync_dirs = config.get("sync_directories") or []
    autoload = config.get("autoload_services") or []
    frontends = config.get("enabled_frontends") or []
    missing = [name for name in autoload if name not in services]
    unloaded = [name for name in autoload if name in services and not getattr(services[name], "loaded", False)]
    bad_dirs = [p for p in sync_dirs if not Path(p).exists()]
    out.append(("ok" if sync_dirs and not bad_dirs else "warn", f"sync_directories: {len(sync_dirs)} configured"))
    out += [("warn", f"missing sync dir: {p}") for p in bad_dirs[:3]]
    out.append(("ok" if not missing else "warn", f"autoload services known: {len(autoload) - len(missing)}/{len(autoload)}"))
    out += [("warn", f"autoload service not discovered: {name}") for name in missing]
    out += [("warn", f"autoload service not loaded: {name}") for name in unloaded]
    out.append(("ok" if frontends else "warn", f"enabled_frontends: {', '.join(frontends) if frontends else '(none)'}"))
    return out


def _plan_lines(context):
    """Return plan-mode status lines for the active session."""
    runtime = getattr(context, "runtime", None)
    session_key = getattr(context, "session_key", None)
    session = (getattr(runtime, "sessions", {}) or {}).get(session_key) if runtime and session_key else None
    if session is None:
        return ["  Plan mode: no active session"]
    parts = ["on" if getattr(session, "plan_mode", False) else "off"]
    if getattr(session, "full_permissions_this_turn", False):
        parts.append("full permissions this turn")
    return [f"  Plan mode: {', '.join(parts)}"]


def _task_lines(counts):
    """Return task queue findings."""
    if not counts:
        return ["  No task queue entries."]
    lines = []
    for name, c in sorted(counts.items()):
        c = {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0, **(c or {})}
        if c["PENDING"] or c["PROCESSING"] or c["FAILED"]:
            lines.append(f"  {name}: {c['PENDING']} pending, {c['PROCESSING']} running, {c['FAILED']} failed")
    return lines or ["  No pending, running, or failed work."]


def _llm_lines(llm):
    """Return LLM/cache findings."""
    active = getattr(llm, "active", None) or llm
    if not active:
        return ["  LLM is not configured."]
    cached, total = getattr(active, "last_cached_prompt_tokens", None), getattr(active, "last_prompt_tokens", None)
    lines = [f"  Model: {getattr(active, 'model_name', 'unknown')} ({'loaded' if getattr(active, 'loaded', False) else 'unloaded'})"]
    if total is None:
        return lines + ["  Prompt cache: no usage data yet."]
    pct = round((cached or 0) * 100 / total) if total else 0
    return lines + [f"  Prompt cache: {cached or 0}/{total} input tokens cached on last call ({pct}%)."]


def _log_lines(path, limit=8):
    """Return recent warning/error log lines."""
    if not path.exists():
        return [f"  No log file found at {path}."]
    hits = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if " | WARNING | " in line or " | ERROR | " in line or " | CRITICAL | " in line]
    return [f"  {line}" for line in hits[-limit:]] or ["  No warnings or errors in this run."]


def _loaded(services):
    """Count loaded services."""
    return sum(1 for svc in services.values() if getattr(svc, "loaded", False))


def _count(checks, kind):
    """Count findings by kind."""
    return sum(1 for k, _ in checks if k == kind)


def _label(kind):
    """Return finding label."""
    return "OK" if kind == "ok" else "WARN"
