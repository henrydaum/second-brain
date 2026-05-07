import logging
from copy import deepcopy

import config.config_manager as config_manager
from plugins.BaseTool import BaseTool, ToolResult
from events.event_channels import SUBAGENT_RUN
from plugins.tasks.helpers.notifications import (
    DEFAULT_NOTIFICATION_MODE,
    NOTIFICATION_MODES,
    category_for_job,
    load_conversation_suffix,
    notification_mode,
)

logger = logging.getLogger("ScheduleSubagent")


def _coerce_input_paths(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_conversation_id(value):
    """Accept an integer id or the literal string 'active'.

    Empty string is treated as "unspecified" (returns None) so that an
    update call clearing the field also strips it from the payload.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("conversation_id cannot be a boolean.")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "active":
        return "active"
    try:
        return int(text)
    except ValueError:
        raise ValueError("conversation_id must be an integer id or the string 'active'.")


def _ensure_timekeeper_autoload(context):
    services = list(context.config.get("autoload_services", []))
    if "timekeeper" in services:
        return
    services.append("timekeeper")
    context.config["autoload_services"] = services
    config_manager.save(context.config)


def _is_subagent_job(job: dict) -> bool:
    return (job.get("channel") or "").strip() == SUBAGENT_RUN


def _job_payload_summary(job: dict) -> tuple[str, str]:
    payload = job.get("payload") or {}
    prompt = str(payload.get("prompt") or "")
    title = str(payload.get("title") or "")
    return title, prompt


def _get_subagent_job_or_none(svc, job_name: str) -> dict | None:
    job = svc.get_job(job_name)
    if job is None or not _is_subagent_job(job):
        return None
    return job


class ScheduleSubagent(BaseTool):
    name = "schedule_subagent"
    description = (
        "Create and manage scheduled background subagent jobs. Use this for "
        "reminders, recurring briefs, periodic checks, or unattended research. "
        "Each job runs inside a specific conversation — the agent profile is "
        "inherited from that conversation. Multiple jobs can share a conversation "
        "to build up shared context. Mutating actions require user approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "delete", "get", "list", "enable", "disable"],
                "description": "What to do with the scheduled subagent job.",
            },
            "job_name": {
                "type": "string",
                "description": "Stable name for the job.",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt the background subagent should run when the job fires.",
            },
            "cron": {
                "type": "string",
                "description": "Cron schedule for repeating jobs.",
            },
            "run_at": {
                "type": "string",
                "description": "ISO datetime for a one-time job.",
            },
            "one_time": {
                "type": "boolean",
                "description": "Whether the job should run once.",
            },
            "input_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional local file paths for the subagent.",
            },
            "title": {
                "type": "string",
                "description": "Optional short title.",
            },
            "conversation_id": {
                "description": (
                    "DEFAULT: leave this blank. A fresh conversation is auto-created "
                    "for the job and appears in /conversations under 'Scheduled' "
                    "(or 'Scheduled (one-time)') so the user can brief it before the "
                    "first run. This is what you want almost every time. "
                    "Only set this if the user has explicitly asked to reuse a "
                    "specific conversation: pass an integer id of an existing "
                    "conversation. The literal string 'active' is a rarely-used "
                    "special case that pins the job to whatever conversation the "
                    "user happens to have open at fire time — do NOT pick it just "
                    "because the user is currently chatting; only use it if they "
                    "explicitly ask for that 'follow my active chat' behavior."
                ),
            },
            "enabled": {
                "type": "boolean",
                "description": "Whether the job should be enabled.",
            },
            "notifications": {
                "type": "string",
                "enum": list(NOTIFICATION_MODES),
                "description": (
                    "How chatty the subagent should be when the job fires. "
                    "'all' (default): the agent pushes regularly via the notify tool, and if it forgets, the final answer is sent automatically as a single push. "
                    "'important': the agent has the notify tool but is told to use it only when something noteworthy comes up; silence is allowed. "
                    "'off': the agent runs silently with no notify tool and produces no user-visible chat output (final answer is still stored). "
                    "When the chosen conversation is the user's currently active one, the notify tool is suppressed regardless of mode — the user already sees the run."
                ),
                "default": DEFAULT_NOTIFICATION_MODE,
            },
        },
        "required": ["action"],
    }
    requires_services = ["timekeeper"]
    max_calls = 8
    background_safe = False

    def run(self, context, **kwargs):
        action = (kwargs.get("action") or "").strip().lower()
        svc = context.services.get("timekeeper")
        if svc is None or not svc.loaded:
            return ToolResult.failed("timekeeper service is not available.")

        if action == "list":
            return self._list_jobs(svc)
        if action in {"get", "delete", "enable", "disable"}:
            job_name = (kwargs.get("job_name") or "").strip()
            if not job_name:
                return ToolResult.failed("job_name is required for this action.")
            if action == "get":
                return self._get_job(svc, job_name, context=context)
            current = _get_subagent_job_or_none(svc, job_name)
            if current is None:
                return ToolResult.failed(f"Unknown subagent job: '{job_name}'.")
            if action == "delete":
                denied = _require_schedule_approval(
                    context,
                    action="delete",
                    job_name=job_name,
                    job=current,
                    schedule_text=self._describe_candidate_job(svc, current),
                )
                if denied:
                    return denied
                removed = svc.remove_job(job_name)
                if not removed:
                    return ToolResult.failed(f"Unknown subagent job: '{job_name}'.")
                return ToolResult(llm_summary=f"Deleted subagent job '{job_name}'.")
            enabled = action == "enable"
            denied = _require_schedule_approval(
                context,
                action=action,
                job_name=job_name,
                job=current,
                schedule_text=self._describe_candidate_job(svc, current),
            )
            if denied:
                return denied
            try:
                job = svc.enable_job(job_name, enabled=enabled)
            except ValueError as e:
                return ToolResult.failed(str(e))
            state = "enabled" if enabled else "disabled"
            return ToolResult(llm_summary=f"Subagent job '{job_name}' {state}.")

        if action in {"create", "update"}:
            job_name = (kwargs.get("job_name") or "").strip()
            if not job_name:
                return ToolResult.failed("job_name is required.")

            _ensure_timekeeper_autoload(context)

            current_job = None
            if action == "update":
                current_job = _get_subagent_job_or_none(svc, job_name)
                if current_job is None:
                    return ToolResult.failed(f"Unknown subagent job: '{job_name}'.")

            job_def = self._build_job_def(
                kwargs,
                require_prompt=(action == "create"),
                current_job=current_job,
            )

            # Eagerly create a conversation for new jobs that didn't get one
            # specified. This way the user can see and brief the conversation
            # in /conversations before the cron ever fires. The "active"
            # sentinel is left untouched — that's an explicit user choice to
            # track whatever conversation the user has open at fire time.
            existing_conv = (job_def.get("payload") or {}).get("conversation_id")
            if action == "create" and existing_conv is None:
                is_one_time = bool(job_def.get("one_time"))
                category = category_for_job(is_scheduled=True, is_one_time=is_one_time)
                title = (str((job_def.get("payload") or {}).get("title") or "").strip()
                         or job_name or "Scheduled subagent run")
                conv_id = self._create_subagent_conversation(context, title[:200], category)
                if conv_id is not None:
                    job_def["payload"]["conversation_id"] = conv_id
                else:
                    logger.warning(
                        "Eager conversation creation for job '%s' returned None — "
                        "the cron will mint one at fire time.", job_name,
                    )
            denied = _require_schedule_approval(
                context,
                action=action,
                job_name=job_name,
                job=job_def,
                schedule_text=self._describe_candidate_job(svc, job_def),
            )
            if denied:
                return denied
            _ensure_timekeeper_autoload(context)
            try:
                job = svc.create_job(job_name, job_def) if action == "create" else svc.update_job(job_name, job_def)
            except ValueError as e:
                return ToolResult.failed(str(e))
            return self._format_job_result(svc, job_name, job, created=(action == "create"), context=context)

        return ToolResult.failed("action must be one of: create, update, delete, get, list, enable, disable.")

    def _build_job_def(self, kwargs: dict, require_prompt: bool, current_job: dict | None) -> dict:
        existing_payload = deepcopy((current_job or {}).get("payload") or {})
        prompt = kwargs.get("prompt")
        if require_prompt and not str(prompt or "").strip():
            raise ValueError("prompt is required when creating a subagent job.")

        payload = existing_payload
        if prompt is not None:
            payload["prompt"] = str(prompt)

        input_paths = kwargs.get("input_paths")
        if input_paths is not None:
            payload["input_paths"] = _coerce_input_paths(input_paths)

        title = kwargs.get("title")
        if title is not None:
            payload["title"] = str(title)

        if "conversation_id" in kwargs:
            coerced = _coerce_conversation_id(kwargs.get("conversation_id"))
            if coerced is None:
                payload.pop("conversation_id", None)
            else:
                payload["conversation_id"] = coerced

        notifications = kwargs.get("notifications")
        if notifications is not None:
            mode = notification_mode(notifications, default="")
            if mode not in NOTIFICATION_MODES:
                raise ValueError(
                    f"notifications must be one of: {', '.join(NOTIFICATION_MODES)}."
                )
            payload["notifications"] = mode
        elif require_prompt and "notifications" not in payload:
            payload["notifications"] = DEFAULT_NOTIFICATION_MODE

        job_def = deepcopy(current_job) if current_job is not None else {}
        job_def["channel"] = SUBAGENT_RUN
        job_def["payload"] = payload

        for key in ("cron", "run_at"):
            if kwargs.get(key) is not None:
                job_def[key] = kwargs.get(key)
        for key in ("one_time", "enabled"):
            if kwargs.get(key) is not None:
                job_def[key] = bool(kwargs.get(key))
        return job_def

    def _describe_candidate_job(self, svc, job: dict) -> str:
        if job.get("one_time"):
            return f"One-time at {job.get('run_at')}"
        cron = str(job.get("cron") or "").strip()
        if not cron:
            return "No schedule configured."
        try:
            return svc.cron_to_text(cron)
        except Exception:
            return f"Cron: {cron}"

    def _list_jobs(self, svc) -> ToolResult:
        jobs = {
            name: job for name, job in svc.list_jobs().items()
            if _is_subagent_job(job)
        }
        if not jobs:
            return ToolResult(llm_summary="No subagent jobs scheduled.")

        lines = []
        data = []
        for name in sorted(jobs):
            job = jobs[name]
            title, prompt = _job_payload_summary(job)
            schedule = svc.describe_job(name)
            state = "enabled" if job.get("enabled", True) else "disabled"
            try:
                next_fire = svc.get_next_fire_at(name)
            except Exception:
                next_fire = None
            next_fire_str = next_fire.isoformat(timespec="seconds") if next_fire else None
            lines.append(f"{name} [{state}]")
            lines.append(f"  {schedule}")
            if next_fire_str:
                lines.append(f"  next run: {next_fire_str}")
            if title:
                lines.append(f"  title: {title}")
            conv = (job.get("payload") or {}).get("conversation_id")
            if conv is not None:
                lines.append(f"  conversation: {conv}")
            notifications = str(
                (job.get("payload") or {}).get("notifications")
                or DEFAULT_NOTIFICATION_MODE
            ).strip().lower()
            lines.append(f"  notifications: {notifications}")
            if prompt:
                lines.append(f"  prompt: {prompt[:160]}")
            data.append({
                "job_name": name,
                "enabled": job.get("enabled", True),
                "schedule": schedule,
                "next_run_at": next_fire_str,
                "title": title,
                "conversation_id": conv,
                "notifications": notifications,
                "prompt": prompt,
            })
        return ToolResult(data=data, llm_summary="\n".join(lines))

    def _get_job(self, svc, job_name: str, context=None) -> ToolResult:
        job = _get_subagent_job_or_none(svc, job_name)
        if job is None:
            return ToolResult.failed(f"Unknown subagent job: '{job_name}'.")
        return self._format_job_result(svc, job_name, job, created=None, context=context)

    @staticmethod
    def _create_subagent_conversation(context, title: str, category: str | None) -> int | None:
        runtime = getattr(context, "runtime", None)
        if runtime is not None:
            try:
                return runtime.create_conversation(title, kind="subagent", category=category)
            except Exception as e:
                logger.warning("runtime.create_conversation failed (%s); falling back to db.", e)
        db = getattr(context, "db", None)
        if db is None:
            return None
        try:
            return db.create_conversation(title, kind="subagent", category=category)
        except Exception as e:
            logger.warning("db.create_conversation failed: %s", e)
            return None

    def _format_job_result(self, svc, job_name: str, job: dict, created: bool | None, context=None) -> ToolResult:
        title, prompt = _job_payload_summary(job)
        schedule = svc.describe_job(job_name)
        state = "enabled" if job.get("enabled", True) else "disabled"
        prefix = (
            f"Created subagent job '{job_name}'."
            if created is True else
            f"Updated subagent job '{job_name}'."
            if created is False else
            f"Subagent job '{job_name}'."
        )
        lines = [prefix, f"State: {state}", f"Schedule: {schedule}"]
        if title:
            lines.append(f"Title: {title}")
        conv = (job.get("payload") or {}).get("conversation_id")
        if conv is not None:
            lines.append(f"Conversation: {conv}")
            db = getattr(context, "db", None) if context is not None else None
            if isinstance(conv, int) and db is not None:
                suffix = load_conversation_suffix(db, conv).strip()
                if suffix:
                    lines.append(suffix)
        notifications = str(
            (job.get("payload") or {}).get("notifications")
            or DEFAULT_NOTIFICATION_MODE
        ).strip().lower()
        lines.append(f"Notifications: {notifications}")
        if prompt:
            lines.append(f"Prompt: {prompt}")
        input_paths = (job.get("payload") or {}).get("input_paths") or []
        if input_paths:
            lines.append("Inputs: " + ", ".join(str(p) for p in input_paths))
        return ToolResult(
            data={
                "job_name": job_name,
                "job": job,
                "schedule": schedule,
            },
            llm_summary="\n".join(lines),
        )


def _require_schedule_approval(context, action: str, job_name: str, job: dict, schedule_text: str) -> ToolResult | None:
    approve_fn = context.approve_command
    if approve_fn is None:
        return ToolResult.failed(
            "Scheduling changes are not available — no approval handler is configured."
        )

    payload = deepcopy(job.get("payload") or {})
    prompt = str(payload.get("prompt") or "").strip()
    title = str(payload.get("title") or "").strip()
    conversation_id = payload.get("conversation_id")
    input_paths = payload.get("input_paths") or []
    state = "enabled" if job.get("enabled", True) else "disabled"

    lines = [
        f"Action: {action}",
        f"Job: {job_name}",
        f"State: {state}",
        f"Schedule: {schedule_text}",
    ]
    if title:
        lines.append(f"Title: {title}")
    if conversation_id is not None:
        lines.append(f"Conversation: {conversation_id}")
    notifications = str(
        payload.get("notifications") or DEFAULT_NOTIFICATION_MODE
    ).strip().lower()
    lines.append(f"Notifications: {notifications}")
    if prompt:
        lines.append(f"Prompt: {prompt}")
    if input_paths:
        lines.append("Inputs: " + ", ".join(str(p) for p in input_paths))

    try:
        approved = approve_fn(f"Schedule subagent job: {action} {job_name}", "\n".join(lines))
    except Exception as e:
        return ToolResult.failed(f"Approval dialog error: {e}")

    if not approved:
        return ToolResult.failed(
            "Scheduling action denied by user. STOP — do not retry this action. "
            "Ask the user what they would like you to do instead."
        )
    return None
