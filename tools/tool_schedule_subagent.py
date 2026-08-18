"""Tool plugin for schedule subagent.

Sandboxed. Creating the job is one Request now (``agent.schedule``) rather
than a hand-built Timekeeper definition, because the channel and payload shape
belong to the kernel's spawn subscriber and a caller spelling them itself can
spell them wrong. Listing, editing and removing still go through ``sdk.cron``,
filtered to jobs on that channel — this tool manages subagent schedules, not
every job on the system, which is what ``/schedule`` is for.

The approval dance is gone: ``agent.schedule`` is ALWAYS_UNSAFE, so the gate
is the policy's rather than this file's, and removing a schedule goes through
``cron.remove`` which is unsafe for the same reason.
"""

dependencies_files = []
dependencies_pip = ['croniter']
requests = ["agent.schedule", "cron.list", "cron.get", "cron.update",
            "cron.remove"]

import re
from datetime import datetime

from croniter import croniter

from guest.bases import BaseTool

# The kernel's spawn channel. Spelled out because a plugin cannot import
# kernel modules; the kernel's own /schedule command carries the same literal.
SUBAGENT_CHANNEL = "subagent.spawn"


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
    agent_prompt = (
        "## Scheduling and cron jobs\n"
        "schedule_subagent is the user's calendar and background task system: "
        "reminders, recurring checks, follow-ups and delayed autonomous work. "
        "When the user asks about their schedule, reminders, upcoming events or "
        "planned tasks, inspect it with schedule_subagent before answering.\n"
        "Schedule reminders for an hour before the event unless told "
        "otherwise, and ask which it is when the request does not say whether "
        "a job recurs or fires once. The scheduled agent runs with nobody "
        "watching and cannot ask you anything, so its prompt has to carry "
        "unambiguous step-by-step instructions."
    )

    def run(self, sdk, **kwargs):
        """Run schedule subagent."""
        action = (kwargs.get("operation") or "").strip().lower()
        if action not in {"list", "add", "edit", "remove"}:
            return sdk.fail("operation must be one of: list, add, edit, remove.")
        if action == "list":
            return _list_jobs(sdk)

        title = (kwargs.get("title") or "").strip()
        if not title:
            return sdk.fail("title is required.")
        attachments = _attachments(kwargs.get("attachments")) if "attachments" in kwargs else None
        if attachments is None and "attachments" in kwargs:
            return sdk.fail("attachments must be a string or list of strings.")

        job_name = _find_job(sdk, title) or _slug(title)
        if action == "remove":
            return _remove(sdk, title, job_name)
        if action == "edit":
            return _edit(sdk, title, job_name, kwargs, attachments)
        return _add(sdk, title, job_name, kwargs, attachments or [])


def _subagent_jobs(sdk) -> dict:
    """Every Timekeeper job that fires a background agent."""
    return {name: job for name, job in (sdk.cron.list() or {}).items()
            if job.get("channel") == SUBAGENT_CHANNEL}


def _list_jobs(sdk):
    """Internal helper to list jobs."""
    rows = []
    for name, job in sorted(_subagent_jobs(sdk).items()):
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
        return sdk.ok({"jobs": []}, llm_summary="No scheduled subagent jobs.")
    lines = [
        f"- {r['title']}: "
        f"{'once at ' + str(r['run_at']) if r['one_time'] else r['cron']} "
        f"({'enabled' if r['enabled'] else 'disabled'})"
        for r in rows
    ]
    return sdk.ok({"jobs": rows},
                  llm_summary="Scheduled subagent jobs:\n" + "\n".join(lines))


def _add(sdk, title: str, job_name: str, kwargs: dict, attachments: list):
    """Internal helper to handle add job."""
    prompt = (kwargs.get("prompt") or "").strip()
    cron = (kwargs.get("cron") or "").strip()
    if not prompt:
        return sdk.fail("prompt is required.")
    if not cron:
        return sdk.fail("cron expression is required.")
    if _find_job(sdk, title) is not None:
        return sdk.fail(f"A scheduled subagent named '{title}' already exists. "
                        "Use edit or remove.")
    try:
        created = sdk.agent.schedule(
            prompt, cron, title=title, attachments=attachments,
            one_time=bool(kwargs.get("one_time")), name=job_name)
    except sdk.Failed as refused:
        return sdk.fail(str(refused))
    return sdk.ok({"title": title, "scheduled": True,
                   "one_time": bool(created.get("one_time"))},
                  llm_summary=f"Scheduled subagent '{title}'.")


def _edit(sdk, title: str, job_name: str, kwargs: dict, attachments):
    """Internal helper to handle edit job."""
    job = sdk.cron.get(job_name)
    if job is None or job.get("channel") != SUBAGENT_CHANNEL:
        return sdk.fail(f"No scheduled subagent named '{title}'.")
    if not any(k in kwargs for k in ("prompt", "cron", "one_time", "attachments")):
        return sdk.fail("edit requires at least one of: prompt, cron, "
                        "one_time, attachments.")
    cron = (kwargs.get("cron") or "").strip()
    if ("one_time" in kwargs and not cron
            and bool(kwargs.get("one_time")) != bool(job.get("one_time"))):
        return sdk.fail("cron is required when changing one_time.")

    payload = dict(job.get("payload") or {})
    if "prompt" in kwargs:
        prompt = (kwargs.get("prompt") or "").strip()
        if not prompt:
            return sdk.fail("prompt cannot be empty.")
        payload["prompt"] = prompt
    if attachments is not None:
        payload["attachments"] = attachments

    patch = {"payload": payload}
    if "cron" in kwargs or "one_time" in kwargs:
        try:
            patch.update(_schedule_def(
                sdk, cron or job.get("cron") or "",
                bool(kwargs.get("one_time", job.get("one_time")))))
        except Exception as exc:
            return sdk.fail(str(exc))
    try:
        sdk.cron.update(job_name, patch)
    except sdk.Failed as refused:
        return sdk.fail(str(refused))
    return sdk.ok({"title": title, "edited": True},
                  llm_summary=f"Updated scheduled subagent '{title}'.")


def _remove(sdk, title: str, job_name: str):
    """Internal helper to remove job."""
    job = sdk.cron.get(job_name)
    if job is None or job.get("channel") != SUBAGENT_CHANNEL:
        return sdk.fail(f"No scheduled subagent named '{title}'.")
    try:
        sdk.cron.remove(job_name)
    except sdk.Failed as refused:
        return sdk.fail(str(refused))
    return sdk.ok({"title": title, "removed": True},
                  llm_summary=f"Removed scheduled subagent '{title}'.")


def _schedule_def(sdk, cron: str, one_time: bool) -> dict:
    """A Timekeeper schedule from a cron expression.

    Only needed on the edit path — ``agent.schedule`` does this itself when
    creating. A one-time job wants an absolute ``run_at`` rather than a cron,
    so the next match is resolved here.
    """
    if one_time:
        run_at = croniter(cron, datetime.now().astimezone()).get_next(datetime)
        return {"run_at": run_at.isoformat(), "cron": None, "one_time": True}
    croniter(cron)  # raises on a malformed expression
    return {"cron": cron, "run_at": None, "one_time": False}


def _attachments(value):
    """Internal helper to handle attachments arg."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


def _slug(title: str) -> str:
    """Internal helper to handle job name."""
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "subagent"


def _find_job(sdk, title: str):
    """Internal helper to find job name."""
    wanted = (title or "").strip()
    jobs = _subagent_jobs(sdk)
    for candidate in (wanted, _slug(wanted)):
        if candidate and candidate in jobs:
            return candidate
    folded = wanted.casefold()
    for name, job in jobs.items():
        payload_title = ((job.get("payload") or {}).get("title") or "").strip()
        if payload_title == wanted or payload_title.casefold() == folded:
            return name
    return None
