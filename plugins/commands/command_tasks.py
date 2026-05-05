import json
from uuid import uuid4

from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_tasks
from state_machine.conversationClass import FormStep
from state_machine.forms import schema_to_form_steps


PATH_ACTIONS = ["pause", "unpause", "reset", "retry"]
EVENT_ACTIONS = ["pause", "unpause", "trigger"]
PIPELINE = "pipeline"


class TasksCommand(BaseCommand):
    name = "tasks"
    description = "Select a task, then pause, unpause, reset, retry, or trigger it"
    category = "System"

    def form(self, args, context):
        tasks = sorted(getattr(getattr(context, "orchestrator", None), "tasks", {}))
        steps = [FormStep("task_name", "Task", True, enum=[*tasks, PIPELINE])]
        if args.get("task_name") == PIPELINE:
            return steps
        task = _task(context, args.get("task_name"))
        if task:
            steps.append(FormStep("action", _describe(context, args["task_name"]), True, enum=EVENT_ACTIONS if getattr(task, "trigger", "path") == "event" else PATH_ACTIONS))
        if task and args.get("action") == "trigger":
            steps += schema_to_form_steps(getattr(task, "event_payload_schema", {}) or {})
        return steps

    def run(self, args, context):
        action, name = args.get("action"), args.get("task_name")
        if not name:
            return _show(context)
        orch = getattr(context, "orchestrator", None)
        if name == PIPELINE:
            return orch.dependency_pipeline_graph() if orch and hasattr(orch, "dependency_pipeline_graph") else "Pipeline unavailable."
        task = _task(context, name)
        if not orch or not task:
            return "Unknown task."
        if action == "pause":
            orch.paused.add(name)
            return f"Paused task: {name}"
        if action == "unpause":
            orch.paused.discard(name)
            orch.clear_skip_cache(name)
            return f"Unpaused task: {name}"
        if action == "reset":
            if getattr(task, "trigger", "path") == "event":
                return "Only path-driven tasks can be reset."
            context.db.reset_task(name)
            orch.clear_skip_cache(name)
            return f"Reset task: {name}"
        if action == "retry":
            if getattr(task, "trigger", "path") == "event":
                return "Only path-driven tasks can be retried."
            context.db.reset_failed_tasks(name)
            orch.clear_skip_cache(name)
            return f"Retried failed entries for task: {name}"
        if action == "trigger":
            if getattr(task, "trigger", "path") != "event":
                return "Only event-driven tasks can be triggered manually."
            return _trigger(context, task, args)
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


def _task(context, name):
    return (getattr(getattr(context, "orchestrator", None), "tasks", {}) or {}).get(name)


def _trigger(context, task, args):
    db = getattr(context, "db", None)
    if db is None or not hasattr(db, "create_run"):
        return "No database is available for task runs."
    payload_keys = (getattr(task, "event_payload_schema", {}) or {}).get("properties", {}).keys()
    payload = {k: args[k] for k in payload_keys if k in args}
    name = getattr(task, "name", args.get("task_name"))
    run_id = f"{name}:{uuid4().hex[:12]}"
    db.create_run(run_id, name, triggered_by="manual", payload_json=json.dumps(payload))
    orch = getattr(context, "orchestrator", None)
    if orch and hasattr(orch, "on_run_enqueued"):
        orch.on_run_enqueued(run_id, name)
    return f"Triggered task: {name} ({run_id})"
