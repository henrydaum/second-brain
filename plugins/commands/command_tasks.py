from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_tasks


class TasksCommand(BaseCommand):
    name = "tasks"
    description = "List registered tasks"
    category = "System"

    def run(self, _args, context):
        orch, db = getattr(context, "orchestrator", None), getattr(context, "db", None)
        counts = (db.get_system_stats().get("tasks", {}) if db else {}) | (db.get_run_stats() if db and hasattr(db, "get_run_stats") else {})
        return format_tasks([{
            "name": name,
            "trigger": getattr(task, "trigger", "path"),
            "counts": counts.get(name, {}),
            "paused": name in getattr(orch, "paused", set()),
            "requires_services": getattr(task, "requires_services", []),
            "trigger_channels": getattr(task, "trigger_channels", []),
        } for name, task in sorted((getattr(orch, "tasks", {}) or {}).items())])
