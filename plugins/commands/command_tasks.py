from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_tasks
from state_machine.conversationClass import FormStep


ACTIONS = ["pause", "unpause"]


class TasksCommand(BaseCommand):
    name = "tasks"
    description = "Select a task, then pause or unpause it"
    category = "System"

    def form(self, args, context):
        tasks = sorted(getattr(getattr(context, "orchestrator", None), "tasks", {}))
        steps = [FormStep("task_name", "Task", True, enum=tasks)]
        if args.get("task_name"):
            steps.append(FormStep("action", _describe(context, args["task_name"]), True, enum=ACTIONS))
        return steps

    def run(self, args, context):
        action, name = args.get("action"), args.get("task_name")
        if not name:
            return _show(context)
        orch = getattr(context, "orchestrator", None)
        if not orch or name not in getattr(orch, "tasks", {}):
            return "Unknown task."
        if action == "pause":
            orch.paused.add(name)
            return f"Paused task: {name}"
        if action == "unpause":
            orch.paused.discard(name)
            orch.clear_skip_cache(name)
            return f"Unpaused task: {name}"
        return f"Unknown action: {action}"


def _show(context):
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


def _describe(context, task_name):
    orch = getattr(context, "orchestrator", None)
    if not orch or task_name not in getattr(orch, "tasks", {}):
        return "Action"
    db = getattr(context, "db", None)
    counts = (db.get_system_stats().get("tasks", {}) if db else {}) | (db.get_run_stats() if db and hasattr(db, "get_run_stats") else {})
    c = {"PENDING": 0, "PROCESSING": 0, "DONE": 0, "FAILED": 0} | counts.get(task_name, {})
    return f"{task_name}\nPending: {c['PENDING']}      Running: {c['PROCESSING']}      Done: {c['DONE']}      Failed: {c['FAILED']}"
