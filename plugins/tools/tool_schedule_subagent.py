import re
from datetime import datetime

from croniter import croniter

from events.event_channels import SPAWN_SUBAGENT
from plugins.BaseTool import BaseTool, ToolResult


SCHEDULED = "Scheduled"
SCHEDULED_ONCE = "Scheduled (one-time)"


class ScheduleSubagent(BaseTool):
    name = "schedule_subagent"
    description = (
        "Schedule a background subagent job with Timekeeper. Use this for reminders, recurring "
        "briefs, check-ins, and other proactive subagent jobs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the scheduled conversation."},
            "prompt": {"type": "string", "description": "What the background agent should do."},
            "cron": {"type": "string", "description": "Cron expression for when the job should run."},
            "one_time": {"type": "boolean", "description": "If true, run once at the next cron match.", "default": False},
            "attachments": {"type": "array", "description": "Optional file paths to attach to each run.", "items": {"type": "string"}, "default": []},
        },
        "required": ["title", "cron", "prompt"],
    }
    requires_services = []
    background_safe = False

    def run(self, context, **kwargs):
        title = (kwargs.get("title") or "").strip()
        prompt = (kwargs.get("prompt") or "").strip()
        cron = (kwargs.get("cron") or "").strip()
        one_time = bool(kwargs.get("one_time"))
        attachments = _attachments_arg(kwargs.get("attachments"))
        if "run_immediately" in kwargs:
            return ToolResult.failed("run_immediately is no longer supported; use a dedicated immediate-run tool instead.")
        if not title:
            return ToolResult.failed("title is required.")
        if not prompt:
            return ToolResult.failed("prompt is required.")
        if not cron:
            return ToolResult.failed("cron expression is required.")
        if attachments is None:
            return ToolResult.failed("attachments must be a string or list of strings.")
        tk = _timekeeper(context)
        if tk is None:
            return ToolResult.failed("Timekeeper service is not available.")
        try:
            schedule = _schedule_def(tk, cron, one_time)
        except Exception as e:
            return ToolResult.failed(str(e))
        if not _approved(context, _approval_text(title, prompt, schedule)):
            return ToolResult(success=False, error="Schedule denied.", llm_summary="The user denied the scheduled subagent.")

        job_name = _unique_job_name(tk, title)
        tk.create_job(job_name, {**schedule, "channel": SPAWN_SUBAGENT, "payload": {"title": title, "prompt": prompt, "attachments": attachments}, "enabled": True})

        summary = _summary(title, job_name)
        return ToolResult(True, data={
            "job_name": job_name,
            "scheduled": True,
            "one_time": bool(one_time),
        }, llm_summary=summary)


def _timekeeper(context):
    tk = (getattr(context, "services", None) or {}).get("timekeeper")
    return tk if tk is not None and getattr(tk, "loaded", False) else None


def _schedule_def(tk, cron: str, one_time: bool) -> dict:
    if one_time:
        run_at = croniter(cron, datetime.now().astimezone()).get_next(datetime)
        return {"run_at": run_at.isoformat(), "one_time": True}
    tk.cron_to_text(cron)
    return {"cron": cron, "one_time": False}


def _attachments_arg(value):
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def _approved(context, text: str) -> bool:
    if getattr(context, "user_initiated", False):
        return True
    approve = getattr(context, "approve_command", None)
    return bool(approve and approve("schedule_subagent", text))


def _approval_text(title: str, prompt: str, schedule: dict) -> str:
    mode = "one-time" if schedule.get("one_time") else "recurring"
    when = schedule.get("run_at") or schedule.get("cron")
    return (
        f"Create {mode} subagent schedule?\n\n"
        f"Title: {title}\n"
        f"When: {when}\n"
        f"Prompt preview:\n{_preview(prompt)}"
    )


def _preview(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _unique_job_name(tk, title: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "subagent"
    name = base
    i = 2
    while tk.get_job(name) is not None:
        name = f"{base}_{i}"
        i += 1
    return name


def _summary(title: str, job_name: str) -> str:
    return f"Scheduled subagent job '{job_name}' on {SPAWN_SUBAGENT}: {title}."
