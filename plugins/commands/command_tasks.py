import json
from uuid import uuid4

from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_tasks
from state_machine.conversation import FormStep
from state_machine.forms import schema_to_form_steps


PATH_ACTIONS = ["pause", "unpause", "reset", "retry"]
EVENT_ACTIONS = ["pause", "unpause", "trigger", "schedule", "unschedule"]
PIPELINE = "pipeline"


class TasksCommand(BaseCommand):
    name = "tasks"
    description = "Pick a task — pause, unpause, reset, retry, trigger, or manage its cron schedules"
    category = "System"

    def form(self, args, context):
        tasks = sorted(getattr(getattr(context, "orchestrator", None), "tasks", {}))
        steps = [FormStep("task_name", "Task", True, enum=[*tasks, PIPELINE], columns=2)]
        if args.get("task_name") == PIPELINE:
            return steps
        task = _task(context, args.get("task_name"))
        if task:
            steps.append(FormStep("action", _describe(context, args["task_name"]), True, enum=EVENT_ACTIONS if getattr(task, "trigger", "path") == "event" else PATH_ACTIONS))
        action = args.get("action")
        if task and action == "trigger":
            steps += schema_to_form_steps(getattr(task, "event_payload_schema", {}) or {}, prompt_optional=True)
        elif task and action == "schedule":
            steps += [
                FormStep("job_name", "Job name (unique)", True),
                FormStep("cron", "Cron expression (e.g. '0 9 * * *')", True),
            ]
            steps += schema_to_form_steps(getattr(task, "event_payload_schema", {}) or {}, prompt_optional=True)
        elif task and action == "unschedule":
            jobs = _jobs_for_task(context, task)
            steps.append(FormStep("job_name", "Job to remove", True, enum=jobs or ["(no jobs scheduled)"]))
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
        if action == "schedule":
            return _schedule_create(context, task, args)
        if action == "unschedule":
            return _schedule_remove(context, args)
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
    return f"{task_name}\nPending: {c['PENDING']}      Running: {c['PROCESSING']}      Done: {c['DONE']}      Failed: {c['FAILED']}\n\n{_schedules_context(context, orch.tasks[task_name])}"


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


def _timekeeper(context):
    tk = (getattr(context, "services", None) or {}).get("timekeeper")
    return tk if tk is not None and getattr(tk, "loaded", False) else None


def _task_channels(task) -> list[str]:
    return [c for c in (getattr(task, "trigger_channels", []) or []) if c]


def _jobs_for_task(context, task) -> list[str]:
    tk = _timekeeper(context)
    if tk is None:
        return []
    channels = set(_task_channels(task))
    return sorted(name for name, job in tk.list_jobs().items() if (job.get("channel") or "") in channels)


def _schedules_context(context, task) -> str:
    tk = _timekeeper(context)
    rows = [] if tk is None else [(name, job) for name, job in tk.list_jobs().items() if (job.get("channel") or "") in set(_task_channels(task))]
    if not rows:
        return "Schedules:\n  (none)"
    lines = ["Schedules:"]
    for name, job in sorted(rows):
        cron = job.get("cron", "")
        try:
            desc = tk.cron_to_text(cron)
        except Exception:
            desc = cron or "?"
        nf = tk.get_next_fire_at(name)
        lines.append(f"  • {name} — {desc} — next: {nf.strftime('%Y-%m-%d %H:%M') if nf else '(disabled)'}")
    return "\n".join(lines)


def _schedule_create(context, task, args):
    tk = _timekeeper(context)
    if tk is None:
        return "Timekeeper service is not available."
    channels = _task_channels(task)
    if not channels:
        return f"Task '{task.name}' has no trigger_channels — cannot schedule."
    job_name = (args.get("job_name") or "").strip()
    cron = (args.get("cron") or "").strip()
    if not job_name or not cron:
        return "job_name and cron are required."
    payload_keys = (getattr(task, "event_payload_schema", {}) or {}).get("properties", {}).keys()
    payload = {k: args[k] for k in payload_keys if k in args}
    try:
        tk.create_job(job_name, {"cron": cron, "channel": channels[0], "payload": payload, "enabled": True})
    except Exception as e:
        return f"Failed to create job: {e}"
    try:
        when = tk.cron_to_text(cron)
    except Exception:
        when = cron
    return f"Scheduled job '{job_name}' for task '{task.name}': {when}"


def _schedule_remove(context, args):
    tk = _timekeeper(context)
    if tk is None:
        return "Timekeeper service is not available."
    job_name = (args.get("job_name") or "").strip()
    if not job_name or job_name == "(no jobs scheduled)":
        return "Pick a job to remove."
    return f"Removed job: {job_name}" if tk.remove_job(job_name) else f"No such job: {job_name}"
