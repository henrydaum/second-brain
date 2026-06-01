"""Slash command plugin for `/schedule`."""

import json

from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_scheduled_jobs
from state_machine.conversation import FormStep
from state_machine.forms import _schema_prompt, schema_to_form_steps


ADD = "add"
NONE = "(none)"
ACTIONS = ["edit", "delete", "enable", "disable"]
ACTION_LABELS = ["Edit it", "Delete it", "Enable it", "Disable it"]


class ScheduleCommand(BaseCommand):
    """Slash-command handler for `/schedule`."""
    name = "schedule"
    description = "Manage Timekeeper scheduled jobs"
    category = "Tasks"

    def form(self, args, context):
        """Handle form."""
        tk = _timekeeper(context)
        jobs = sorted(tk.list_jobs()) if tk else []
        steps = [FormStep("job_name", _list_prompt(tk), True, enum=[*jobs, ADD], enum_labels=[*_job_labels(tk, jobs), "Schedule new job"], columns=1)]
        if args.get("job_name") == ADD:
            tasks = _event_tasks(context)
            steps += [
                FormStep("task_name", "Choose the event-driven task to schedule.", True, enum=sorted(tasks) or [NONE], columns=1),
                FormStep("new_job_name", "Enter a unique name for this schedule.", True),
                FormStep("cron", "Enter the cron expression, for example 0 9 * * *.", True),
            ]
            task = tasks.get(args.get("task_name"))
            if task:
                steps += schema_to_form_steps(getattr(task, "event_payload_schema", {}) or {}, prompt_optional=True)
            return steps
        job = tk.get_job(args.get("job_name")) if tk and args.get("job_name") else None
        if job:
            steps.append(FormStep("action", f"What do you want to do with this scheduled job?\n\n{_describe(context, args['job_name'], job)}", True, enum=ACTIONS, enum_labels=ACTION_LABELS, columns=2))
        if job and args.get("action") == "edit":
            steps.append(FormStep("run_at" if job.get("one_time") else "cron", "Enter the new run time." if job.get("one_time") else "Enter the new cron expression.", True, default=job.get("run_at") if job.get("one_time") else job.get("cron")))
            steps += _payload_steps(_task_for_job(context, job), job.get("payload") or {})
        return steps

    def run(self, args, context):
        """Execute `/schedule` for the active session."""
        tk = _timekeeper(context)
        if tk is None:
            return "Timekeeper service is not available."
        if not args.get("job_name"):
            return format_scheduled_jobs(tk.list_jobs(), tk)
        if args.get("job_name") == ADD:
            return _create(context, tk, args)
        name = args.get("job_name")
        job = tk.get_job(name)
        if job is None:
            return f"No such job: {name}"
        action = args.get("action")
        if action == "delete":
            return f"Deleted job: {name}" if tk.remove_job(name) else f"No such job: {name}"
        if action == "enable":
            tk.enable_job(name, True)
            return f"Enabled job: {name}"
        if action == "disable":
            tk.enable_job(name, False)
            return f"Disabled job: {name}"
        if action == "edit":
            return _edit(context, tk, name, job, args)
        return f"Unknown action: {action}"


def _timekeeper(context):
    """Internal helper to handle timekeeper."""
    tk = (getattr(context, "services", None) or {}).get("timekeeper")
    return tk if tk is not None and getattr(tk, "loaded", False) else None


def _event_tasks(context) -> dict:
    """Internal helper to handle event tasks."""
    tasks = getattr(getattr(context, "orchestrator", None), "tasks", {}) or {}
    return {name: task for name, task in tasks.items() if getattr(task, "trigger", "path") == "event" and _task_channels(task)}


def _task_channels(task) -> list[str]:
    """Internal helper to handle task channels."""
    return [c for c in (getattr(task, "trigger_channels", []) or []) if c]


def _task_for_job(context, job):
    """Internal helper to handle task for job."""
    channel = (job or {}).get("channel")
    return next((task for task in _event_tasks(context).values() if channel in _task_channels(task)), None)


def _job_labels(tk, jobs: list[str]) -> list[str]:
    """Internal helper to handle job labels."""
    if not tk:
        return []
    labels = []
    for name in jobs:
        job = tk.get_job(name) or {}
        labels.append(f"{name} (disabled)" if not job.get("enabled", True) else name)
    return labels


def _list_prompt(tk) -> str:
    """Internal helper to list prompt."""
    if tk is None:
        return "Timekeeper service is not available."
    return f"{format_scheduled_jobs(tk.list_jobs(), tk)}\n\nSelect a scheduled job, or add a new one."


def _describe(context, name: str, job: dict) -> str:
    """Internal helper to handle describe."""
    tk = _timekeeper(context)
    task = _task_for_job(context, job)
    next_fire = tk.get_next_fire_at(name) if tk else None
    payload = json.dumps(job.get("payload") or {}, separators=(",", ":"), default=str)
    return "\n".join([
        name,
        f"Status: {'Enabled' if job.get('enabled', True) else 'Disabled'}",
        f"Task: {getattr(task, 'name', '') or '-'}",
        f"Channel: {job.get('channel') or '-'}",
        f"Schedule: {_schedule_text(tk, job)}",
        f"Next: {next_fire.strftime('%Y-%m-%d %H:%M') if next_fire else 'disabled'}",
        f"Payload: {_truncate(payload, 500)}",
    ])


def _schedule_text(tk, job: dict) -> str:
    """Internal helper to handle schedule text."""
    if job.get("one_time"):
        return f"once at {job.get('run_at') or '?'}"
    cron = job.get("cron") or ""
    if tk:
        try:
            return tk.cron_to_text(cron).lower()
        except Exception:
            pass
    return cron or "?"


def _payload_steps(task, payload: dict) -> list[FormStep]:
    """Internal helper to handle payload steps."""
    schema = getattr(task, "event_payload_schema", {}) or {}
    props = schema.get("properties") or {}
    if not props:
        return [FormStep("payload", "Enter the payload as a JSON object.", False, "object", default=payload, prompt_when_missing=True)]
    return [
        FormStep(name, _schema_prompt(name, info), False, info.get("type", "string"), info.get("enum"), default=payload.get(name, info.get("default")), prompt_when_missing=True)
        for name, info in props.items()
    ]


def _create(context, tk, args):
    """Internal helper to create schedule."""
    task = _event_tasks(context).get(args.get("task_name"))
    if task is None:
        return "Choose an event-driven task to schedule."
    name, cron = (args.get("new_job_name") or "").strip(), (args.get("cron") or "").strip()
    if not name or not cron:
        return "Enter both a schedule name and a cron expression."
    payload = _schema_payload(task, args)
    try:
        tk.create_job(name, {"cron": cron, "channel": _task_channels(task)[0], "payload": payload, "enabled": True})
    except Exception as e:
        return f"Failed to create job: {e}"
    return f"Created schedule '{name}' for {task.name}: {_schedule_text(tk, tk.get_job(name) or {'cron': cron})}."


def _edit(context, tk, name: str, job: dict, args):
    """Internal helper to handle edit."""
    patch = {"payload": _edited_payload(_task_for_job(context, job), job.get("payload") or {}, args)}
    patch["run_at" if job.get("one_time") else "cron"] = (args.get("run_at") if job.get("one_time") else args.get("cron")) or (job.get("run_at") if job.get("one_time") else job.get("cron"))
    try:
        tk.update_job(name, patch)
    except Exception as e:
        return f"Failed to update job: {e}"
    return f"Updated job: {name}"


def _schema_payload(task, args) -> dict:
    """Internal helper to handle schema payload."""
    return {k: args[k] for k in ((getattr(task, "event_payload_schema", {}) or {}).get("properties") or {}) if k in args}


def _edited_payload(task, current: dict, args) -> dict:
    """Internal helper to handle edited payload."""
    if "payload" in args:
        return args["payload"] or {}
    out = dict(current or {})
    for key in ((getattr(task, "event_payload_schema", {}) or {}).get("properties") or {}):
        if key in args:
            out[key] = args[key]
    return out


def _truncate(text: str, limit: int) -> str:
    """Internal helper to handle truncate."""
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
