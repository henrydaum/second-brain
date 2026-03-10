"""
Controller.

The command layer between user input and the system. Exposes every
control action as a plain method. The terminal REPL calls these,
and the GUI will call the same methods later.

The controller never prints — it returns strings or dicts.
The caller decides how to display them.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("Controller")


def _truncate_cell(text: str, max_len: int = 60) -> str:
    """Shorten long cell values for tabular display."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _format_tool_result(result) -> str:
    """Format a ToolResult for human-readable REPL output."""
    if not result.success:
        return f"Error: {result.error}"

    data = result.data

    # Special handling for sql_query — render as a table
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        columns = data["columns"]
        rows = data["rows"]

        if not rows:
            return "(no results)"

        col_widths = [len(c) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(_truncate_cell(str(val))))

        header = "  ".join(c.ljust(w) for c, w in zip(columns, col_widths))
        separator = "  ".join("-" * w for w in col_widths)
        lines = [header, separator]
        for row in rows:
            line = "  ".join(
                _truncate_cell(str(val)).ljust(w)
                for val, w in zip(row, col_widths)
            )
            lines.append(line)

        if data.get("truncated"):
            lines.append("  ... (results capped at 100 rows)")

        return "\n".join(lines)

    # Default: pretty-print as JSON
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)


class Controller:
    def __init__(self, orchestrator, db, services: dict, config: dict, tool_registry=None):
        self.orchestrator = orchestrator
        self.db = db
        self.services = services
        self.config = config
        self.tool_registry = tool_registry

    # =================================================================
    # SERVICES
    # =================================================================

    def list_services(self) -> str:
        """List all services and their status."""
        if not self.services:
            return "No services registered."

        lines = []
        for name, svc in self.services.items():
            loaded = getattr(svc, 'loaded', False)
            model = getattr(svc, 'model_name', '')
            status = "LOADED" if loaded else "unloaded"
            lines.append(f"  {name:<20} {status:<10} {model}")

        return "Services:\n" + "\n".join(lines)

    def load_service(self, name: str) -> str:
        """Load a service and re-check blocked tasks."""
        svc = self.services.get(name)
        if svc is None:
            return f"Unknown service: '{name}'. Use 'services' to see available."

        if getattr(svc, 'loaded', False):
            return f"Service '{name}' is already loaded."

        logger.info(f"Loading service '{name}'...")
        try:
            success = svc.load()
        except Exception as e:
            return f"Failed to load '{name}': {e}"

        if not success:
            return f"Service '{name}' failed to load."

        # Clear skip log so orchestrator re-checks waiting tasks
        self.orchestrator._skip_logged.discard(name)
        # Clear all task skip flags — services changed, recheck everything
        self.orchestrator._skip_logged.clear()

        return f"Service '{name}' loaded."

    def unload_service(self, name: str) -> str:
        """Unload a service to free resources."""
        svc = self.services.get(name)
        if svc is None:
            return f"Unknown service: '{name}'."

        if not getattr(svc, 'loaded', False):
            return f"Service '{name}' is already unloaded."

        try:
            svc.unload()
        except Exception as e:
            return f"Error unloading '{name}': {e}"

        return f"Service '{name}' unloaded."

    # =================================================================
    # TASKS
    # =================================================================

    def list_tasks(self) -> str:
        """List all tasks with status counts and paused state."""
        stats = self.db.get_system_stats()
        task_stats = stats.get("tasks", {})

        if not self.orchestrator.tasks:
            return "No tasks registered."

        lines = []
        for name, task in self.orchestrator.tasks.items():
            counts = task_stats.get(name, {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0})
            paused = " [PAUSED]" if name in self.orchestrator.paused else ""
            svc_info = f"  needs: {task.requires_services}" if task.requires_services else ""

            lines.append(
                f"  {name:<22} "
                f"P:{counts['PENDING']:<4} "
                f"R:{counts['PROCESSING']:<4} "
                f"D:{counts['DONE']:<4} "
                f"F:{counts['FAILED']:<4}"
                f"{paused}{svc_info}"
            )

        return "Tasks:\n" + "\n".join(lines)

    def pause_task(self, name: str) -> str:
        """Pause a task. It stays PENDING but won't be dispatched."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'. Use 'tasks' to see available."

        if name in self.orchestrator.paused:
            return f"Task '{name}' is already paused."

        self.orchestrator.paused.add(name)
        return f"Task '{name}' paused."

    def unpause_task(self, name: str) -> str:
        """Unpause a task. Pending work will resume on next dispatch cycle."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'."

        if name not in self.orchestrator.paused:
            return f"Task '{name}' is not paused."

        self.orchestrator.paused.discard(name)
        return f"Task '{name}' unpaused."

    def reset_task(self, name: str) -> str:
        """Reset ALL entries for a task back to PENDING, including downstream tasks."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'."

        self.db.reset_task(name)
        downstream = self.orchestrator._get_all_downstream(name)
        if downstream:
            self.db.invalidate_tasks_bulk(downstream)
        return f"Task '{name}' reset — all entries back to PENDING (+ {len(downstream)} downstream)."

    def retry_task(self, name: str) -> str:
        """Retry only FAILED entries for a task, invalidating their downstream tasks."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'."

        failed_paths = self.db.get_paths_for_task_status(name, "FAILED")
        self.db.reset_failed_tasks(name)
        downstream = self.orchestrator._get_all_downstream(name)
        if downstream and failed_paths:
            self.db.invalidate_tasks_for_paths(downstream, failed_paths)
        return f"Task '{name}' — failed entries reset to PENDING."

    def retry_all(self) -> str:
        """Retry all FAILED entries across all tasks, invalidating downstream."""
        # Cascade before resetting so we can detect which paths were FAILED
        for name in self.orchestrator.tasks:
            failed_paths = self.db.get_paths_for_task_status(name, "FAILED")
            if failed_paths:
                downstream = self.orchestrator._get_all_downstream(name)
                if downstream:
                    self.db.invalidate_tasks_for_paths(downstream, failed_paths)
        self.db.reset_failed_tasks()
        return "All failed tasks reset to PENDING."

    # =================================================================
    # STATS
    # =================================================================

    def stats(self) -> str:
        """System overview."""
        s = self.db.get_system_stats()

        lines = ["Files by modality:"]
        file_stats = s.get("files", {})
        if file_stats:
            for mod, count in sorted(file_stats.items()):
                lines.append(f"  {mod:<12} {count}")
        else:
            lines.append("  (none)")

        total = sum(file_stats.values()) if file_stats else 0
        lines.append(f"  {'total':<12} {total}")

        lines.append("")
        lines.append("Task queue:")
        task_stats = s.get("tasks", {})
        if task_stats:
            for name, counts in sorted(task_stats.items()):
                paused = " [PAUSED]" if name in self.orchestrator.paused else ""
                lines.append(
                    f"  {name:<22} "
                    f"P:{counts['PENDING']:<4} "
                    f"R:{counts['PROCESSING']:<4} "
                    f"D:{counts['DONE']:<4} "
                    f"F:{counts['FAILED']:<4}"
                    f"{paused}"
                )
        else:
            lines.append("  (empty)")

        return "\n".join(lines)

    # =================================================================
    # TOOLS
    # =================================================================

    def list_tools(self) -> str:
        """List all registered tools with descriptions and required services."""
        if self.tool_registry is None or not self.tool_registry.tools:
            return "No tools registered."

        lines = []
        for name, tool in self.tool_registry.tools.items():
            svc_info = f"  needs: {tool.requires_services}" if tool.requires_services else ""
            lines.append(f"  {name}{svc_info}")

            # Description (first line only, truncated)
            desc = (tool.description or "").split("\n")[0]
            if len(desc) > 200:
                desc = desc[:197] + "..."
            lines.append(f"    {desc}")

            # Parameters
            params = tool.parameters.get("properties", {})
            required = set(tool.parameters.get("required", []))
            if params:
                param_parts = []
                for pname, pschema in params.items():
                    req = "*" if pname in required else ""
                    param_parts.append(f"{pname}{req}")
                lines.append(f"    args: {', '.join(param_parts)}")

            lines.append("")

        return "Tools:\n" + "\n".join(lines)

    def call_tool(self, name: str, kwargs: dict) -> str:
        """Call a tool by name and return formatted results."""
        if self.tool_registry is None:
            return "No tool registry available."

        result = self.tool_registry.call(name, **kwargs)
        return _format_tool_result(result)

    # =================================================================
    # HELP
    # =================================================================

    def help(self) -> str:
        return """Commands:
  services                  List services and status
  load <n>                  Load a service
  unload <n>                Unload a service

  tasks                     List tasks with status counts
  pause <n>                 Pause a task
  unpause <n>               Unpause a task
  reset <n>                 Reset all entries for a task to PENDING
  retry <n>                 Retry failed entries for a task
  retry all                 Retry all failed across all tasks

  tools                     List registered tools
  call <tool> <json_args>   Call a tool (e.g. call sql_query {"sql": "SELECT ..."})

  chat                      Enter agent chat mode (requires llm service)

  stats                     System overview
  quit / exit               Shutdown"""