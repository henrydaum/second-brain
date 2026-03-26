"""
Controller.

The command layer between user input and the system. Exposes every
control action as a plain method. The terminal REPL calls these,
and the GUI will call the same methods later.

The controller never prints — it returns structured data or status strings.
The caller decides how to display them.
"""

import logging
from pathlib import Path

logger = logging.getLogger("Controller")


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

    def list_services(self) -> list[dict]:
        """List all services and their status."""
        return [
            {"name": name, "loaded": getattr(svc, 'loaded', False),
             "model_name": getattr(svc, 'model_name', '')}
            for name, svc in self.services.items()
        ]

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
        self.orchestrator.skip_cache.discard(name)
        # Clear all task skip flags — services changed, recheck everything
        self.orchestrator.skip_cache.clear()

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

    def list_tasks(self) -> list[dict]:
        """List all tasks with status counts and paused state."""
        stats = self.db.get_system_stats()
        task_stats = stats.get("tasks", {})
        return [
            {"name": name,
             "counts": task_stats.get(name, {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0}),
             "paused": name in self.orchestrator.paused,
             "requires_services": task.requires_services}
            for name, task in self.orchestrator.tasks.items()
        ]

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
        downstream = self.orchestrator.get_all_downstream(name)
        if downstream:
            self.db.invalidate_tasks_bulk(downstream)
        return f"Task '{name}' reset — all entries back to PENDING (+ {len(downstream)} downstream)."

    def retry_task(self, name: str) -> str:
        """Retry only FAILED entries for a task, invalidating their downstream tasks."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'."

        failed_paths = self.db.get_paths_for_task_status(name, "FAILED")
        self.db.reset_failed_tasks(name)
        downstream = self.orchestrator.get_all_downstream(name)
        if downstream and failed_paths:
            self.db.invalidate_tasks_for_paths(downstream, failed_paths)
        return f"Task '{name}' — failed entries reset to PENDING."

    def retry_all(self) -> str:
        """Retry all FAILED entries across all tasks, invalidating downstream."""
        for name in self.orchestrator.tasks:
            failed_paths = self.db.get_paths_for_task_status(name, "FAILED")
            if failed_paths:
                downstream = self.orchestrator.get_all_downstream(name)
                if downstream:
                    self.db.invalidate_tasks_for_paths(downstream, failed_paths)
        self.db.reset_failed_tasks()
        return "All failed tasks reset to PENDING."

    # =================================================================
    # STATS
    # =================================================================

    def stats(self) -> dict:
        """System overview as structured data."""
        s = self.db.get_system_stats()
        return {
            "files": s.get("files", {}),
            "tasks": {
                name: {**counts, "paused": name in self.orchestrator.paused}
                for name, counts in s.get("tasks", {}).items()
            },
        }

    # =================================================================
    # TOOLS
    # =================================================================

    def enable_tool(self, name: str) -> str:
        """Enable a tool for agent use."""
        if self.tool_registry is None:
            return "No tool registry available."
        tool = self.tool_registry.tools.get(name)
        if tool is None:
            return f"Unknown tool: '{name}'. Use 'tools' to see available."
        if tool.agent_enabled:
            return f"Tool '{name}' is already enabled."
        tool.agent_enabled = True
        return f"Tool '{name}' enabled for agent use."

    def disable_tool(self, name: str) -> str:
        """Disable a tool from agent use (still callable via 'call')."""
        if self.tool_registry is None:
            return "No tool registry available."
        tool = self.tool_registry.tools.get(name)
        if tool is None:
            return f"Unknown tool: '{name}'. Use 'tools' to see available."
        if not tool.agent_enabled:
            return f"Tool '{name}' is already disabled."
        tool.agent_enabled = False
        return f"Tool '{name}' disabled for agent use."

    def list_tools(self) -> list[dict]:
        """List all registered tools with descriptions and required services."""
        if self.tool_registry is None:
            return []
        return [
            {"name": name,
             "description": (tool.description or "").split("\n")[0],
             "agent_enabled": tool.agent_enabled,
             "max_calls": tool.max_calls,
             "requires_services": tool.requires_services,
             "parameters": tool.parameters}
            for name, tool in self.tool_registry.tools.items()
        ]

    def call_tool(self, name: str, kwargs: dict):
        """Call a tool by name and return the ToolResult."""
        if self.tool_registry is None:
            from Stage_3.BaseTool import ToolResult
            return ToolResult.failed("No tool registry available.")
        return self.tool_registry.call(name, **kwargs)

    # =================================================================
    # PLUGINS
    # =================================================================

    def reload_plugins(self, root_dir: Path) -> str:
        """Re-discover tasks and tools from all plugin directories."""
        from plugin_discovery import discover_tasks, discover_tools
        discover_tasks(root_dir, self.orchestrator, self.config, reload=True)
        discover_tools(root_dir, self.tool_registry, self.config, reload=True)
        return "Plugins reloaded."

    # =================================================================
    # HELP
    # =================================================================

    def help(self) -> list[dict]:
        """Command list for the REPL. The GUI generates its own help from the command registry."""
        return [
            {"command": "services", "description": "List services and status"},
            {"command": "load <n>", "description": "Load a service"},
            {"command": "unload <n>", "description": "Unload a service"},
            {"command": "", "description": ""},
            {"command": "tasks", "description": "List tasks with status counts"},
            {"command": "pipeline", "description": "Show task dependency graph"},
            {"command": "pause <n>", "description": "Pause a task"},
            {"command": "unpause <n>", "description": "Unpause a task"},
            {"command": "reset <n>", "description": "Reset all entries for a task to PENDING"},
            {"command": "retry <n>", "description": "Retry failed entries for a task"},
            {"command": "retry all", "description": "Retry all failed across all tasks"},
            {"command": "", "description": ""},
            {"command": "tools", "description": "List registered tools"},
            {"command": "enable <n>", "description": "Enable a tool for agent use"},
            {"command": "disable <n>", "description": "Disable a tool from agent use"},
            {"command": "call <tool> <json>", "description": "Call a tool directly"},
            {"command": "", "description": ""},
            {"command": "reload", "description": "Hot-reload tasks and tools"},
            {"command": "", "description": ""},
            {"command": "stats", "description": "System overview"},
            {"command": "quit / exit", "description": "Shutdown"},
        ]
