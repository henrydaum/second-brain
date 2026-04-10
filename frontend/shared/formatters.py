"""
Plain-text formatters for command output.

Used by both the GUI (app.py) and the terminal REPL (repl.py).
"""

import json


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
        return f"Error: {result.error}"
    data = result.data
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
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)


def format_services(services: list[dict]) -> str:
    """Format the service list showing name, loaded/unloaded status, and model."""
    if not services:
        return "No services registered."
    lines = []
    for s in services:
        status = "LOADED" if s["loaded"] else "unloaded"
        lines.append(f"  {s['name']:<20} {status:<10} {s['model_name']}")
    return "Services:\n" + "\n".join(lines)


def format_tasks(tasks: list[dict]) -> str:
    """Format task list with queue counts (Pending/Processing/Done/Failed)."""
    if not tasks:
        return "No tasks registered."
    lines = []
    for t in tasks:
        c = t["counts"]
        paused = " [PAUSED]" if t["paused"] else ""
        svc = f"  needs: {t['requires_services']}" if t["requires_services"] else ""
        lines.append(
            f"  {t['name']:<22} "
            f"P:{c['PENDING']:<8} R:{c['PROCESSING']:<8} "
            f"D:{c['DONE']:<8} F:{c['FAILED']:<8}{paused}{svc}"
        )
    return "Tasks:\n" + "\n".join(lines)


def format_stats(stats: dict) -> str:
    """Format system overview: file counts by modality + task queue summaries."""
    lines = ["Files by modality:"]
    files = stats.get("files", {})
    if files:
        for mod, count in sorted(files.items()):
            lines.append(f"  {mod:<12} {count}")
    else:
        lines.append("  (none)")
    lines.append(f"  {'total':<12} {sum(files.values()) if files else 0}")
    lines.append("")
    lines.append("Task queue:")
    tasks = stats.get("tasks", {})
    if tasks:
        for name, counts in sorted(tasks.items()):
            paused = " [PAUSED]" if counts.get("paused") else ""
            lines.append(
                f"  {name:<22} "
                f"P:{counts['PENDING']:<8} R:{counts['PROCESSING']:<8} "
                f"D:{counts['DONE']:<8} F:{counts['FAILED']:<8}{paused}"
            )
    else:
        lines.append("  (empty)")
    return "\n".join(lines)


def format_tools(tools: list[dict]) -> str:
    """Format tool list with enabled/disabled status, descriptions, and parameters."""
    if not tools:
        return "No tools registered."
    lines = []
    for t in tools:
        status = "" if t["agent_enabled"] else " [DISABLED]"
        svc = f"  needs: {t['requires_services']}" if t["requires_services"] else ""
        lines.append(f"  {t['name']}{status}{svc}")
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


def format_help(commands: list[dict]) -> str:
    """Format the help output as an aligned two-column list."""
    return "Commands:\n" + "\n".join(
        f"  {c['command']:<25} {c['description']}" for c in commands
    )


def format_locations(data: dict) -> str:
    """Format the locations data as a readable file tree."""
    lines = []

    root_path = data.get("root_path", "")
    data_path = data.get("data_path", "")
    root_tree = data.get("root_tree", [])
    data_tree = data.get("data_tree", [])

    lines.append(f"ROOT: {root_path}")
    lines.append("")

    if root_tree:
        for f in root_tree:
            lines.append(f"  {f}")
    else:
        lines.append("  (empty)")

    lines.append("")
    lines.append(f"DATA_DIR: {data_path}")
    lines.append("")

    if data_tree:
        for f in data_tree:
            lines.append(f"  {f}")
    else:
        lines.append("  (empty)")

    return "\n".join(lines)

