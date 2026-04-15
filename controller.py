"""
Controller.

The command layer between user input and the system. Exposes every
control action as a plain method. The terminal REPL calls these, as well as other frontends.

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
            return f"Failed to load '{name}': {e}. Check service-related config settings with /config."

        if not success:
            return f"Service '{name}' failed to load. Check service-related config settings with /config."

        from event_bus import bus
        from event_channels import SERVICE_LOADED
        bus.emit(SERVICE_LOADED, {"name": name, "loaded": True})

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

        from event_bus import bus
        from event_channels import SERVICE_LOADED
        bus.emit(SERVICE_LOADED, {"name": name, "loaded": False})

        return f"Service '{name}' unloaded."

    def reload_services_for_settings(self, changed_keys: set, root_dir: Path) -> list[str]:
        """Rebuild only the services whose config_settings include a changed key.

        Groups affected services by source module so that build_services()
        is called once per module (it may return multiple services).

        Returns a list of human-readable feedback strings.
        """
        from plugin_discovery import get_setting_service_map, discover_services

        svc_map = get_setting_service_map()
        affected: set[str] = set()
        for key in changed_keys:
            if key in svc_map:
                affected.update(svc_map[key])

        if not affected:
            return []

        feedback = []

        # Group affected services by source module path.
        # Baked-in services share a module path derived from their class.
        module_groups: dict[str, list[str]] = {}
        for svc_name in affected:
            svc = self.services.get(svc_name)
            if svc is None:
                continue
            src = getattr(svc, '_source_path', None) or svc.__class__.__module__
            module_groups.setdefault(src, []).append(svc_name)

        # Pause orchestrator to prevent dispatches during the swap
        saved_pauses = set(self.orchestrator.paused)
        self.orchestrator.paused.update(self.orchestrator.tasks.keys())

        try:
            # Unload all affected services first
            previously_loaded: list[str] = []
            for svc_names in module_groups.values():
                for n in svc_names:
                    if getattr(self.services.get(n), 'loaded', False):
                        previously_loaded.append(n)
                        try:
                            self.services[n].unload()
                        except Exception as ex:
                            logger.warning(f"Failed to unload '{n}': {ex}")

            # Single rediscovery — cherry-pick only the affected services
            new_services = discover_services(root_dir, self.config)
            for n in affected:
                if n in new_services:
                    self.services[n] = new_services[n]

            # Reload services that were previously loaded
            for n in previously_loaded:
                svc = self.services.get(n)
                if svc:
                    try:
                        svc.load()
                        feedback.append(f"'{n}' reloaded.")
                    except Exception as ex:
                        feedback.append(f"'{n}' failed to reload: {ex}")
        finally:
            # Restore orchestrator pauses + notify listeners
            self.orchestrator.paused.clear()
            self.orchestrator.paused.update(saved_pauses)
            from event_bus import bus
            from event_channels import SERVICE_LOADED
            bus.emit(SERVICE_LOADED, {"name": None, "loaded": True})

        return feedback

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
                "requires_services": getattr(tool, 'requires_services', []),
                "parameters": getattr(tool, 'parameters', {}),
                "_source_path": getattr(tool, '_source_path', None),
                "_mutable": getattr(tool, '_mutable', False),
            }
            for name, tool in self.tool_registry.tools.items()
        ]

    def list_locations(self, filter_type: str | None = None) -> dict:
        """Walk ROOT_DIR and DATA_DIR and return their file trees.

        Args:
            filter_type: Optional — one of 'tools', 'tasks', or 'services'.
                         When set, only the directories relevant to that
                         plugin type are listed.

        Returns a dict with:
            root_path  — absolute path to ROOT_DIR
            data_path  — absolute path to DATA_DIR
            root_tree  — list of relative-path strings under ROOT_DIR
            data_tree  — list of absolute-path strings under DATA_DIR
        """
        from paths import ROOT_DIR, DATA_DIR, SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES
        import plugin_discovery as _pd

        # Which subdirectories matter for each plugin type.
        _type_dirs = {
            "tools":    {
                "root": [_pd._TOOL_CONFIG["baked_in_dir"]],
                "data": [SANDBOX_TOOLS],
            },
            "tasks":    {
                "root": [_pd._TASK_CONFIG["baked_in_dir"]],
                "data": [SANDBOX_TASKS],
            },
            "services": {
                "root": [_pd._SERVICE_CONFIG["baked_in_dir"]],
                "data": [SANDBOX_SERVICES],
            },
        }

        def _walk_tree(base: Path, root_dirs: list[Path] | None = None) -> list[Path]:
            """Return sorted file paths under *base*.

            If *root_dirs* is given, only files that live under one of
            those directories are included.
            """
            if not base.exists():
                return []
            paths: list[Path] = []
            for p in sorted(base.rglob("*")):
                if p.is_dir():
                    continue
                # Skip hidden / cache directories
                parts = p.relative_to(base).parts
                if any(part.startswith(".") or part == "__pycache__" for part in parts):
                    continue
                if root_dirs:
                    if not any(p.is_relative_to(rd) for rd in root_dirs):
                        continue
                paths.append(p)
            return paths

        if filter_type:
            dirs = _type_dirs.get(filter_type)
            if dirs is None:
                return {"root_path": str(ROOT_DIR), "data_path": str(DATA_DIR),
                        "root_tree": [], "data_tree": []}
            root_files = _walk_tree(ROOT_DIR, dirs["root"])
            data_files = _walk_tree(DATA_DIR, dirs["data"])
        else:
            root_files = _walk_tree(ROOT_DIR)
            data_files = _walk_tree(DATA_DIR)

        root_tree = [str(p.relative_to(ROOT_DIR)) for p in root_files]
        data_tree = [str(p) for p in data_files]

        return {
            "root_path": str(ROOT_DIR),
            "data_path": str(DATA_DIR),
            "root_tree": root_tree,
            "data_tree": data_tree,
        }

    def call_tool(self, name: str, kwargs: dict):
        """Call a tool by name and return the ToolResult.
        Approval prompts for destructive tools flow through the event bus
        (APPROVAL_REQUESTED), not a callback threaded through here."""
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
