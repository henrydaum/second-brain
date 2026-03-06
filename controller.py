"""
Controller.

The command layer between user input and the system. Exposes every
control action as a plain method. The terminal REPL calls these,
and the GUI will call the same methods later.

The controller never prints — it returns strings or dicts.
The caller decides how to display them.
"""

import logging
from pathlib import Path

logger = logging.getLogger("Controller")


class Controller:
    def __init__(self, orchestrator, db, services: dict, config: dict):
        self.orchestrator = orchestrator
        self.db = db
        self.services = services
        self.config = config

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

        header = "  {'Task':<22} {'Pending':<6} {'Running':<6} {'Done':<6} {'Failed':<6}"
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
        """Reset ALL entries for a task back to PENDING."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'."

        self.db.reset_task(name)
        return f"Task '{name}' reset — all entries back to PENDING."

    def retry_task(self, name: str) -> str:
        """Retry only FAILED entries for a task."""
        if name not in self.orchestrator.tasks:
            return f"Unknown task: '{name}'."

        self.db.reset_failed_tasks(name)
        return f"Task '{name}' — failed entries reset to PENDING."

    def retry_all(self) -> str:
        """Retry all FAILED entries across all tasks."""
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
    # HELP
    # =================================================================

    def help(self) -> str:
        return """Commands:
		services                  List services and status
		load <name>               Load a service
		unload <name>             Unload a service

		tasks                     List tasks with status counts
		pause <name>              Pause a task
		unpause <name>            Unpause a task
		reset <name>              Reset all entries for a task to PENDING
		retry <name>              Retry failed entries for a task
		retry all                 Retry all failed across all tasks

		stats                     System overview

		quit / exit               Shutdown"""