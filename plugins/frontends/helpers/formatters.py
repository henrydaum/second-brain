"""
Plain-text formatters for command output.

Used by the Telegram frontend and the terminal REPL.
Compact mode is used by the Telegram frontend for mobile-friendly output.
"""

import json


# ── Canonical status labels ─────────────────────────────────────────
# Use these everywhere so wording stays consistent across frontends.

def status_badge(loaded: bool) -> str:
    return "Loaded" if loaded else "Unloaded"


def enabled_badge(enabled: bool) -> str:
    return "Enabled" if enabled else "Disabled"


def paused_suffix(paused: bool) -> str:
    return "  (paused)" if paused else ""


TASK_STATE_LABELS = {
    "PENDING": "Pending", "PROCESSING": "Running",
    "DONE": "Done", "FAILED": "Failed",
}
TASK_STATE_ABBR = {
    "PENDING": "P", "PROCESSING": "R",
    "DONE": "D", "FAILED": "F",
}


def truncate_cell(text: str, max_len: int = 60) -> str:
    """Truncate *text* to *max_len*, appending '...' if clipped."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _format_subagent_result(data: dict) -> str | None:
    if "final_answer" not in data or ("run_id" not in data and "conversation_id" not in data):
        return None
    title = str(data.get("title") or "").strip()
    answer = str(data.get("final_answer") or "").strip()
    if not answer:
        return None
    return f"{title}\n\n{answer}" if title else answer


def format_tool_result(result) -> str:
    """Format a ToolResult for monospace display.

    Tabular data (columns + rows) is rendered as aligned columns;
    everything else falls back to pretty-printed JSON.
    """
    if not result.success:
        return f"Error: {result.error}"
    data = result.data
    if isinstance(data, dict):
        subagent = _format_subagent_result(data)
        if subagent:
            return subagent
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        columns = data["columns"]
        rows = data["rows"]
        if not rows:
            return "(no results)"
        col_widths = [len(c) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(truncate_cell(str(val))))
        header = "  ".join(c.ljust(w) for c, w in zip(columns, col_widths))
        separator = "  ".join("-" * w for w in col_widths)
        lines = [header, separator]
        for row in rows:
            line = "  ".join(truncate_cell(str(val)).ljust(w) for val, w in zip(row, col_widths))
            lines.append(line)
        if data.get("truncated"):
            lines.append("  ... (results capped at 100 rows)")
        return "\n".join(lines)
    if data is None:
        return result.llm_summary or "(no output)"
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)


def format_services(services: list[dict], compact: bool = False) -> str:
    """Format the service list showing name, loaded/unloaded status, and model."""
    if not services:
        return "No services registered."

    if compact:
        lines = []
        for s in services:
            model = f" ({s['model_name']})" if s["model_name"] else ""
            lines.append(f"{s['name']}: {status_badge(s['loaded'])}{model}")
        return "Services:\n" + "\n".join(lines)

    loaded = [s for s in services if s["loaded"]]
    unloaded = [s for s in services if not s["loaded"]]
    lines = ["Services:"]
    if loaded:
        lines.append("  Loaded:")
        for s in loaded:
            model = s['model_name'] or ""
            lines.append(f"    {s['name']:<20} {model}")
    if unloaded:
        if loaded:
            lines.append("")
        lines.append("  Unloaded:")
        for s in unloaded:
            model = s['model_name'] or ""
            lines.append(f"    {s['name']:<20} {model}")
    return "\n".join(lines)


def _task_sections(tasks) -> list[tuple[str, list[dict]]]:
    """Group task records into path-driven and event-driven sections."""
    empty_counts = {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0}
    normalized = []

    if isinstance(tasks, dict):
        for name, counts in tasks.items():
            normalized.append({
                "name": name,
                "trigger": "path",
                "counts": {**empty_counts, **counts},
                "paused": bool(counts.get("paused")),
                "requires_services": [],
                "trigger_channels": [],
            })
    else:
        for task in tasks or []:
            normalized.append({
                "name": task["name"],
                "trigger": task.get("trigger", "path"),
                "counts": {**empty_counts, **task.get("counts", {})},
                "paused": bool(task.get("paused")),
                "requires_services": task.get("requires_services", []),
                "trigger_channels": task.get("trigger_channels", []),
            })

    normalized.sort(key=lambda task: task["name"])

    path_tasks = [task for task in normalized if task["trigger"] == "path"]
    event_tasks = [task for task in normalized if task["trigger"] == "event"]
    other_tasks = [
        task for task in normalized
        if task["trigger"] not in {"path", "event"}
    ]

    sections = [
        ("Path-driven tasks", path_tasks),
        ("Event-driven tasks", event_tasks),
    ]
    if other_tasks:
        sections.append(("Other tasks", other_tasks))
    return sections


def _task_detail_lines(task: dict) -> list[str]:
    """Return extra metadata lines for a task listing."""
    details = []
    channels = task.get("trigger_channels") or []
    if channels:
        details.append(f"channels: {', '.join(channels)}")
    services = task.get("requires_services") or []
    if services:
        details.append(f"needs: {services}")
    return details


def format_tasks(tasks: list[dict], compact: bool = False) -> str:
    """Format task list with separate path-driven and event-driven sections."""
    if not tasks:
        return "No tasks registered."
    sections = _task_sections(tasks)
    lines = ["Tasks:"]
    if compact:
        for title, section in sections:
            lines.append("")
            lines.append(f"{title}:")
            if not section:
                lines.append("  (none)")
                continue
            for task in section:
                counts = task["counts"]
                lines.append(f"{task['name']}{paused_suffix(task['paused'])}")
                lines.append(
                    f"  {TASK_STATE_ABBR['PENDING']}:{counts['PENDING']} "
                    f"{TASK_STATE_ABBR['PROCESSING']}:{counts['PROCESSING']} "
                    f"{TASK_STATE_ABBR['DONE']}:{counts['DONE']} "
                    f"{TASK_STATE_ABBR['FAILED']}:{counts['FAILED']}"
                )
                for detail in _task_detail_lines(task):
                    lines.append(f"  {detail}")
        return "\n".join(lines)

    for title, section in sections:
        lines.append("")
        lines.append(f"{title}:")
        if not section:
            lines.append("  (none)")
            continue
        for task in section:
            counts = task["counts"]
            lines.append(
                f"  {task['name']:<22} "
                f"{TASK_STATE_LABELS['PENDING']}: {counts['PENDING']:<6} "
                f"{TASK_STATE_LABELS['PROCESSING']}: {counts['PROCESSING']:<6} "
                f"{TASK_STATE_LABELS['DONE']}: {counts['DONE']:<6} "
                f"{TASK_STATE_LABELS['FAILED']}: {counts['FAILED']:<6}"
                f"{paused_suffix(task['paused'])}"
            )
            for detail in _task_detail_lines(task):
                lines.append(f"    {detail}")
    return "\n".join(lines)


def format_tools(tools: list[dict], compact: bool = False) -> str:
    """Format tool list with descriptions and parameters."""
    if not tools:
        return "No tools registered."
    if compact:
        lines = []
        for t in tools:
            desc = t["description"]
            if len(desc) > 100:
                desc = desc[:97] + "..."
            lines.append(f"{t['name']}\n  {desc}")
        return "Tools:\n" + "\n".join(lines)
    lines = []
    for t in tools:
        svc = f"  needs: {t['requires_services']}" if t["requires_services"] else ""
        lines.append(f"  {t['name']}{svc}")
        desc = t["description"]
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"    {desc}")
        params = t["parameters"].get("properties", {})
        required = set(t["parameters"].get("required", []))
        if params:
            parts = [f"{p}{'*' if p in required else ''}" for p in params]
            lines.append(f"    args: {', '.join(parts)}")
        lines.append("")
    return "Tools:\n" + "\n".join(lines)


def format_locations(data: dict) -> str:
    """Format the locations data as a readable file tree."""
    lines = []

    root_path = data.get("root_path", "")
    data_path = data.get("data_path", "")
    root_tree = data.get("root_tree", [])
    data_tree = data.get("data_tree", [])

    lines.append(f"Project root: {root_path}")
    lines.append("")

    if root_tree:
        for f in root_tree:
            lines.append(f"  {f}")
    else:
        lines.append("  (empty)")

    lines.append("")
    lines.append(f"Data directory: {data_path}")
    lines.append("")

    if data_tree:
        for f in data_tree:
            lines.append(f"  {f}")
    else:
        lines.append("  (empty)")

    return "\n".join(lines)


# ── Scheduled jobs ───────────────────────────────────────────────────

def _format_schedule_summary(job: dict, timekeeper=None) -> str:
    """Return a human-readable schedule description for a job."""
    if job.get("one_time"):
        run_at = job.get("run_at") or "?"
        return f"once at {run_at}"
    cron = job.get("cron") or ""
    if timekeeper is not None:
        try:
            return timekeeper.cron_to_text(cron).lower()
        except Exception:
            pass
    return cron


def format_scheduled_jobs(jobs: dict, timekeeper=None) -> str:
    """Format the scheduled-jobs dict as an aligned status list."""
    if not jobs:
        return "No scheduled jobs."

    rows = []
    for name, job in sorted(jobs.items()):
        enabled = job.get("enabled", True)
        schedule = _format_schedule_summary(job, timekeeper)
        title = (job.get("payload", {}).get("title") or "").strip()
        rows.append((name, enabled_badge(enabled), schedule, title))

    name_w = max((len(r[0]) for r in rows), default=4)
    badge_w = max((len(r[1]) for r in rows), default=8)
    sched_w = max((len(r[2]) for r in rows), default=16)

    lines = ["Scheduled jobs:"]
    for name, badge, schedule, title in rows:
        suffix = f'  "{title}"' if title else ""
        lines.append(
            f"  {name:<{name_w}}  {badge:<{badge_w}}  {schedule:<{sched_w}}{suffix}"
        )
    return "\n".join(lines)
