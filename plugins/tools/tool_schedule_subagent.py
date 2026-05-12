"""Tool plugin for schedule subagent."""

import re
from datetime import datetime

from croniter import croniter

from events.event_channels import SPAWN_SUBAGENT
from plugins.BaseTool import BaseTool, ToolResult


SCHEDULED = "Scheduled"
SCHEDULED_ONCE = "Scheduled (one-time)"


class ScheduleSubagent(BaseTool):
    """Schedule subagent."""
    name = "schedule_subagent"
    description = (
        "List, add, edit, or remove Timekeeper-backed background subagent jobs. Use this for "
        "reminders, recurring briefs, check-ins, and other proactive subagent jobs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "description": "Operation to perform.", "enum": ["list", "add", "edit", "remove"]},
            "title": {"type": "string", "description": "Scheduled subagent title. Required for add, edit, and remove."},
            "prompt": {"type": "string", "description": "What the background agent should do. Required for add; optional for edit."},
            "cron": {"type": "string", "description": "Cron expression. Required for add; optional for edit."},
            "one_time": {"type": "boolean", "description": "If true, run once at the next cron match.", "default": False},
            "attachments": {"type": "array", "description": "Optional file paths to attach to each run.", "items": {"type": "string"}},
        },
        "required": ["operation"],
    }
    requires_services = []
    background_safe = False

    def run(self, context, **kwargs):
        """Run schedule subagent."""
        action = (kwargs.get("operation") or (kwargs.get("action") if kwargs.get("action") != "call" else "") or "").strip().lower()
        title = (kwargs.get("title") or "").strip()
        prompt = (kwargs.get("prompt") or "").strip()
        cron = (kwargs.get("cron") or "").strip()
        one_time = bool(kwargs.get("one_time"))
        if "run_immediately" in kwargs:
            return ToolResult.failed("run_immediately is no longer supported; use a dedicated immediate-run tool instead.")
        if action not in {"list", "add", "edit", "remove"}:
            return ToolResult.failed("action must be one of: list, add, edit, remove.")
        tk = _timekeeper(context)
        if tk is None:
            return ToolResult.failed("Timekeeper service is not available.")
        if action == "list":
            return _list_jobs(tk)
        if not title:
            return ToolResult.failed("title is required.")
        job_name = _find_job_name(tk, title) or _job_name(title)
        if action == "remove":
            return _remove_job(context, tk, title, job_name)
        attachments = _attachments_arg(kwargs.get("attachments")) if "attachments" in kwargs else None
        if attachments is None and "attachments" in kwargs:
            return ToolResult.failed("attachments must be a string or list of strings.")
        if action == "edit":
            return _edit_job(context, tk, title, job_name, kwargs, prompt, cron, attachments)
        return _add_job(context, tk, title, job_name, prompt, cron, one_time, attachments or [])


def _list_jobs(tk):
    """Internal helper to list jobs."""
    rows = []
    for name, job in sorted(tk.list_jobs().items()):
        if job.get("channel") != SPAWN_SUBAGENT:
            continue
        payload = job.get("payload") or {}
        rows.append({
            "title": (payload.get("title") or name).strip(),
            "cron": job.get("cron"),
            "run_at": job.get("run_at"),
            "one_time": bool(job.get("one_time")),
            "enabled": bool(job.get("enabled", True)),
            "attachments": list(payload.get("attachments") or []),
            "conversation_id": payload.get("conversation_id"),
        })
    if not rows:
        return ToolResult(True, data={"jobs": []}, llm_summary="No scheduled subagent jobs.")
    lines = [
        f"- {r['title']}: {'once at ' + r['run_at'] if r['one_time'] else r['cron']} "
        f"({'enabled' if r['enabled'] else 'disabled'})"
        for r in rows
    ]
    return ToolResult(True, data={"jobs": rows}, llm_summary="Scheduled subagent jobs:\n" + "\n".join(lines))


def _add_job(context, tk, title: str, job_name: str, prompt: str, cron: str, one_time: bool, attachments: list[str]):
    """Internal helper to handle add job."""
    if not prompt:
        return ToolResult.failed("prompt is required.")
    if not cron:
        return ToolResult.failed("cron expression is required.")
    if _find_job_name(tk, title) is not None:
        return ToolResult.failed(f"A scheduled subagent named '{title}' already exists. Use edit or remove.")
    try:
        schedule = _schedule_def(tk, cron, one_time)
    except Exception as e:
        return ToolResult.failed(str(e))
    payload = {"title": title, "prompt": prompt, "attachments": attachments}
    if not _approved(context, _approval_text("add", title, payload, schedule)):
        return ToolResult(success=False, error="Schedule denied.", llm_summary="The user denied the scheduled subagent.")
    tk.create_job(job_name, {**schedule, "channel": SPAWN_SUBAGENT, "payload": payload, "enabled": True})
    return ToolResult(True, data={"title": title, "scheduled": True, "one_time": bool(one_time)}, llm_summary=f"Scheduled subagent '{title}'.")


def _edit_job(context, tk, title: str, job_name: str, kwargs: dict, prompt: str, cron: str, attachments):
    """Internal helper to handle edit job."""
    job = tk.get_job(job_name)
    if job is None or job.get("channel") != SPAWN_SUBAGENT:
        return ToolResult.failed(f"No scheduled subagent named '{title}'.")
    if not any(k in kwargs for k in ("prompt", "cron", "one_time", "attachments")):
        return ToolResult.failed("edit requires at least one of: prompt, cron, one_time, attachments.")
    if "one_time" in kwargs and not cron and bool(kwargs.get("one_time")) != bool(job.get("one_time")):
        return ToolResult.failed("cron is required when changing one_time.")
    patch = {}
    payload = dict(job.get("payload") or {})
    if "prompt" in kwargs:
        if not prompt:
            return ToolResult.failed("prompt cannot be empty.")
        payload["prompt"] = prompt
    if attachments is not None:
        payload["attachments"] = attachments
    if "cron" in kwargs or "one_time" in kwargs:
        try:
            schedule = _schedule_def(tk, cron or job.get("cron") or "", bool(kwargs.get("one_time", job.get("one_time"))))
        except Exception as e:
            return ToolResult.failed(str(e))
        patch.update(schedule)
    patch["payload"] = payload
    if not _approved(context, _approval_text("edit", title, payload, {**job, **patch})):
        return ToolResult(success=False, error="Schedule edit denied.", llm_summary="The user denied the scheduled subagent edit.")
    tk.update_job(job_name, patch)
    return ToolResult(True, data={"title": title, "edited": True}, llm_summary=f"Updated scheduled subagent '{title}'.")


def _remove_job(context, tk, title: str, job_name: str):
    """Internal helper to remove job."""
    job = tk.get_job(job_name)
    if job is None or job.get("channel") != SPAWN_SUBAGENT:
        return ToolResult.failed(f"No scheduled subagent named '{title}'.")
    if not _approved(context, f"Remove scheduled subagent?\n\nTitle: {title}"):
        return ToolResult(success=False, error="Schedule removal denied.", llm_summary="The user denied removing the scheduled subagent.")
    tk.remove_job(job_name)
    return ToolResult(True, data={"title": title, "removed": True}, llm_summary=f"Removed scheduled subagent '{title}'.")


def _timekeeper(context):
    """Internal helper to handle timekeeper."""
    tk = (getattr(context, "services", None) or {}).get("timekeeper")
    return tk if tk is not None and getattr(tk, "loaded", False) else None


def _schedule_def(tk, cron: str, one_time: bool) -> dict:
    """Internal helper to handle schedule def."""
    if one_time:
        run_at = croniter(cron, datetime.now().astimezone()).get_next(datetime)
        return {"run_at": run_at.isoformat(), "cron": None, "one_time": True}
    tk.cron_to_text(cron)
    return {"cron": cron, "run_at": None, "one_time": False}


def _attachments_arg(value):
    """Internal helper to handle attachments arg."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def _approved(context, text: str) -> bool:
    """Internal helper to handle approved."""
    if getattr(context, "user_initiated", False):
        return True
    approve = getattr(context, "approve_command", None)
    return bool(approve and approve("schedule_subagent", text))


def _approval_text(action: str, title: str, payload: dict, schedule: dict) -> str:
    """Internal helper to handle approval text."""
    mode = "one-time" if schedule.get("one_time") else "recurring"
    when = schedule.get("run_at") or schedule.get("cron")
    return (
        f"{action.title()} {mode} subagent schedule?\n\n"
        f"Title: {title}\n"
        f"When: {when}\n"
        f"Prompt preview:\n{_preview(payload.get('prompt') or '')}"
    )


def _preview(text: str, limit: int = 700) -> str:
    """Internal helper to handle preview."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."


def _job_name(title: str) -> str:
    """Internal helper to handle job name."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "subagent"


def _find_job_name(tk, title: str) -> str | None:
    """Internal helper to find job name."""
    wanted = (title or "").strip()
    for candidate in (wanted, _job_name(wanted)):
        if candidate and tk.get_job(candidate) is not None:
            return candidate
    folded = wanted.casefold()
    for name, job in tk.list_jobs().items():
        payload_title = ((job.get("payload") or {}).get("title") or "").strip()
        if payload_title == wanted or payload_title.casefold() == folded:
            return name
    return None
