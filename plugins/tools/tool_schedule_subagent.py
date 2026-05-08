import re
from datetime import datetime

from croniter import croniter

from events.event_channels import SPAWN_SUBAGENT
from plugins.BaseTool import BaseTool, ToolResult
from plugins.tasks.task_spawn_subagent import SpawnSubagent
from state_machine.serialization import save_state_marker


SCHEDULED = "Scheduled"
SCHEDULED_ONCE = "Scheduled (one-time)"


class ScheduleSubagent(BaseTool):
    name = "schedule_subagent"
    description = (
        "Create a default-agent background conversation and run it now, schedule it, or both. "
        "Use this for reminders, recurring briefs, check-ins, and other proactive subagent jobs."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short title for the scheduled conversation."},
            "prompt": {"type": "string", "description": "What the background agent should do."},
            "cron": {"type": "string", "description": "Optional cron expression for future runs."},
            "one_time": {"type": "boolean", "description": "If true, run once at the next cron match.", "default": False},
            "run_immediately": {"type": "boolean", "description": "If true, run one turn immediately.", "default": False},
        },
        "required": ["title", "prompt"],
    }
    requires_services = []
    background_safe = False

    def run(self, context, **kwargs):
        title = (kwargs.get("title") or "").strip()
        prompt = (kwargs.get("prompt") or "").strip()
        cron = (kwargs.get("cron") or "").strip()
        one_time = bool(kwargs.get("one_time"))
        run_now = bool(kwargs.get("run_immediately"))
        if not title:
            return ToolResult.failed("title is required.")
        if not prompt:
            return ToolResult.failed("prompt is required.")
        if not cron and not run_now:
            return ToolResult.failed("Choose at least one: enter a cron expression, or set run_immediately to True.")

        db = getattr(context, "db", None)
        runtime = getattr(context, "runtime", None)
        if db is None or runtime is None:
            return ToolResult.failed("Database and ConversationRuntime are required.")
        tk = _timekeeper(context)
        if cron and tk is None:
            return ToolResult.failed("Timekeeper service is not available.")
        try:
            schedule = _schedule_def(tk, cron, one_time) if cron else None
        except Exception as e:
            return ToolResult.failed(str(e))
        if schedule and not _approved(context, _approval_text(title, prompt, schedule, run_now)):
            return ToolResult(success=False, error="Schedule denied.", llm_summary="The user denied the scheduled subagent.")

        category = SCHEDULED_ONCE if (one_time or run_now and not cron) else SCHEDULED
        conv_id = db.create_conversation(title, kind="user", category=category)
        save_state_marker(db, conv_id, {
            "conversation_id": conv_id,
            "active_agent_profile": "default",
            "profile_override": "default",
        })

        job_name = _unique_job_name(tk, title) if schedule else None
        payload = {"title": title, "prompt": prompt, "conversation_id": conv_id}
        if job_name:
            payload["job_name"] = job_name
            tk.create_job(job_name, {**schedule, "channel": SPAWN_SUBAGENT, "payload": payload, "enabled": True})

        ran = None
        if run_now:
            ran = SpawnSubagent().run_event(f"spawn_subagent:immediate:{conv_id}", payload, context)
            if not ran.success:
                return ToolResult(False, ran.error, {"conversation_id": conv_id, "job_name": job_name}, ran.error)

        summary = _summary(title, conv_id, job_name, schedule, ran)
        return ToolResult(True, data={
            "conversation_id": conv_id,
            "job_name": job_name,
            "scheduled": bool(schedule),
            "ran_immediately": bool(run_now),
            "final_text": _final_text(runtime, conv_id) if run_now else "",
            "category": category,
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


def _approved(context, text: str) -> bool:
    if getattr(context, "user_initiated", False):
        return True
    approve = getattr(context, "approve_command", None)
    return bool(approve and approve("schedule_subagent", text))


def _approval_text(title: str, prompt: str, schedule: dict, run_now: bool) -> str:
    mode = "one-time" if schedule.get("one_time") else "recurring"
    when = schedule.get("run_at") or schedule.get("cron")
    return (
        f"Create {mode} subagent schedule?\n\n"
        f"Title: {title}\n"
        f"When: {when}\n"
        f"Run immediately: {bool(run_now)}\n\n"
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


def _summary(title: str, conv_id: int, job_name: str | None, schedule: dict | None, ran) -> str:
    parts = [f"Created default-agent subagent conversation #{conv_id}: {title}."]
    if job_name:
        parts.append(f"Scheduled job '{job_name}' on {SPAWN_SUBAGENT}.")
    elif schedule is None:
        parts.append("No schedule was created because no cron expression was provided.")
    if ran is not None:
        parts.append("Ran one subagent turn immediately.")
    return " ".join(parts)


def _final_text(runtime, conv_id: int) -> str:
    key = f"spawn_subagent:{conv_id}"
    session = runtime.sessions.get(key)
    if not session:
        return ""
    msg = next((m for m in reversed(session.history) if m.get("role") == "assistant"), None)
    return (msg or {}).get("content") or ""
