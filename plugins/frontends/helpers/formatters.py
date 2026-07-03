"""
Plain-text formatters for command output.

Used by the Telegram frontend and the terminal REPL.
Compact mode is used by the Telegram frontend for mobile-friendly output.
"""

import json
import re


# ── Markdown tables ─────────────────────────────────────────────────
# Commands emit GitHub-style markdown tables (the single source of truth);
# each frontend renders them natively: rich frontends (Telegram Rich
# Messages) parse the markdown into real tables, while monospace surfaces
# (REPL, Telegram's <pre> fallback) run text through align_md_tables().

def md_table(headers: list, rows: list) -> str:
    """Build a GitHub-style markdown table from headers and row tuples."""
    def cell(value) -> str:
        return str("" if value is None else value).replace("\n", " ").replace("|", "\\|")
    lines = ["| " + " | ".join(cell(h) for h in headers) + " |",
             "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_plain(text: str) -> str:
    """Render markdown-ish output for a monospace terminal: align tables
    and drop code-fence markers (the content already reads as plain text)."""
    aligned = align_md_tables(text)
    return "\n".join(line for line in aligned.split("\n") if not re.fullmatch(r"\s*```\w*\s*", line))


def detail_card(title: str, pairs: list[tuple]) -> str:
    """A titled key/value block: a two-column markdown table whose header
    row carries the title, so describe-style prompts render as a card."""
    return md_table([title, ""], pairs)


_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")


def _split_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip().replace("\\|", "|") for c in cells]


def align_md_tables(text: str) -> str:
    """Render markdown tables in *text* as padded monospace columns.

    Non-table lines pass through untouched, so the same message body works
    on rich and plain surfaces alike.
    """
    lines = (text or "").split("\n")
    out, i = [], 0
    while i < len(lines):
        if (_TABLE_ROW.match(lines[i]) and i + 1 < len(lines)
                and _TABLE_SEPARATOR.match(lines[i + 1])):
            block = [lines[i]]
            j = i + 2
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                block.append(lines[j])
                j += 1
            rows = [_split_row(line) for line in block]
            n = max(len(r) for r in rows)
            rows = [r + [""] * (n - len(r)) for r in rows]
            widths = [max(len(r[c]) for r in rows) for c in range(n)]
            def fmt(row):
                return "  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip()
            out.append(fmt(rows[0]))
            out.append("  ".join("-" * w for w in widths))
            out.extend(fmt(r) for r in rows[1:])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


# ── Canonical status labels ─────────────────────────────────────────
# Use these everywhere so wording stays consistent across frontends.

def status_badge(loaded: bool) -> str:
    """Handle status badge."""
    return "Loaded" if loaded else "Unloaded"


def enabled_badge(enabled: bool) -> str:
    """Handle enabled badge."""
    return "Enabled" if enabled else "Disabled"


def paused_suffix(paused: bool) -> str:
    """Handle paused suffix."""
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


def format_tool_result(result) -> str:
    """Format a ToolResult for monospace display.

    Tabular data (columns + rows) is rendered as aligned columns;
    everything else falls back to pretty-printed JSON.
    """
    if not result.success:
        return f"Failed: {result.error or result.llm_summary or '(no details)'}"
    data = result.data
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        columns = data["columns"]
        rows = data["rows"]
        if not rows:
            return "(no results)"
        table = md_table(columns, [[truncate_cell(str(val)) for val in row] for row in rows])
        if data.get("truncated"):
            table += "\n... (results capped at 100 rows)"
        return table
    if data is None:
        return result.llm_summary or "(no output)"
    if result.llm_summary:
        text = f"Done: {result.llm_summary.strip()}"
        final = data.get("final_text") if isinstance(data, dict) else None
        return f"{text}\n\n{str(final).strip()}" if final else text
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
            status = "Extension" if s.get("lifecycle") == "extension" else status_badge(s['loaded'])
            lines.append(f"{s['name']}: {status}{model}")
        return "Services:\n" + "\n".join(lines)

    rows = [(
        s["name"],
        "Extension" if s.get("lifecycle") == "extension" else status_badge(s["loaded"]),
        s["model_name"] or "",
    ) for s in services]
    return "Services:\n\n" + md_table(["Service", "Status", "Model"], rows)


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
        details.append(f"listens on: {', '.join(channels)}")
    services = task.get("requires_services") or []
    if services:
        details.append(f"needs: {services}")
    details.extend(task.get("schedules") or [])
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
        lines.append(f"**{title}**")
        lines.append("")  # tables must start their own block or parsers fold them inline
        if not section:
            lines.append("(none)")
            continue
        rows = []
        for task in section:
            counts = task["counts"]
            notes = "; ".join((["paused"] if task["paused"] else []) + _task_detail_lines(task))
            rows.append((task["name"], counts["PENDING"], counts["PROCESSING"],
                         counts["DONE"], counts["FAILED"], notes))
        lines.append(md_table(["Task", "Pending", "Running", "Done", "Failed", "Notes"], rows))
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
    rows = []
    for t in tools:
        params = t["parameters"].get("properties", {})
        required = set(t["parameters"].get("required", []))
        args = ", ".join(f"{p}{'*' if p in required else ''}" for p in params)
        desc = truncate_cell(t["description"].split("\n")[0], 100)
        if t["requires_services"]:
            desc += f" (needs: {', '.join(t['requires_services'])})"
        rows.append((t["name"], args, desc))
    return "Tools:\n\n" + md_table(["Tool", "Args", "Description"], rows)


def format_locations(data: dict) -> str:
    """Format the locations data as fenced file trees (rich renderers
    collapse the single newlines of a bare listing)."""
    def section(label: str, path: str, tree: list[str]) -> str:
        listing = "\n".join(tree) if tree else "(empty)"
        return f"**{label}**\n`{path}`\n```\n{listing}\n```"

    return "\n\n".join([
        section("Project root", data.get("root_path", ""), data.get("root_tree", [])),
        section("Data directory", data.get("data_path", ""), data.get("data_tree", [])),
    ])


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
    return "Scheduled jobs:\n\n" + md_table(["Job", "Status", "Schedule", "Title"], rows)
