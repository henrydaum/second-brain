from copy import deepcopy

import config_manager
from Stage_3.BaseTool import BaseTool, ToolResult
from subagent_runtime import SUBAGENT_RUN_CHANNEL


def _coerce_input_paths(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _ensure_timekeeper_autoload(context):
    services = list(context.config.get("autoload_services", []))
    if "timekeeper" in services:
        return
    services.append("timekeeper")
    context.config["autoload_services"] = services
    config_manager.save(context.config)


def _is_subagent_job(job: dict) -> bool:
    return (job.get("channel") or "").strip() == SUBAGENT_RUN_CHANNEL


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
    description = "Schedule a background subagent job. Use this for reminders, daily briefs, or recurring research."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "delete", "get", "list", "enable", "disable"],
                "description": "What to do.",
            },
            "job_name": {
                "type": "string",
                "description": "Job name.",
            },
            "prompt": {
                "type": "string",
                "description": "Prompt the background subagent should run.",
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
            "description": {
                "type": "string",
                "description": "Optional job description.",
            },
            "enabled": {
                "type": "boolean",
                "description": "Whether the job should be enabled.",
            },
        },
        "required": ["action"],
    }
    requires_services = ["timekeeper"]
    agent_enabled = True
    max_calls = 8

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
                return self._get_job(svc, job_name)
            current = _get_subagent_job_or_none(svc, job_name)
            if current is None:
                return ToolResult.failed(f"Unknown subagent job: '{job_name}'.")
            if action == "delete":
                removed = svc.remove_job(job_name)
                if not removed:
                    return ToolResult.failed(f"Unknown subagent job: '{job_name}'.")
                return ToolResult(llm_summary=f"Deleted subagent job '{job_name}'.")
            enabled = action == "enable"
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
                job_name, kwargs,
                require_prompt=(action == "create"),
                current_job=current_job,
            )
            try:
                job = svc.create_job(job_name, job_def) if action == "create" else svc.update_job(job_name, job_def)
            except ValueError as e:
                return ToolResult.failed(str(e))
            return self._format_job_result(svc, job_name, job, created=(action == "create"))

        return ToolResult.failed("action must be one of: create, update, delete, get, list, enable, disable.")

    def _build_job_def(self, job_name: str, kwargs: dict, require_prompt: bool, current_job: dict | None) -> dict:
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
        payload["job_name"] = job_name

        job_def = deepcopy(current_job) if current_job is not None else {}
        job_def["channel"] = SUBAGENT_RUN_CHANNEL
        job_def["payload"] = payload

        for key in ("cron", "run_at", "description"):
            if kwargs.get(key) is not None:
                job_def[key] = kwargs.get(key)
        for key in ("one_time", "enabled"):
            if kwargs.get(key) is not None:
                job_def[key] = bool(kwargs.get(key))
        return job_def

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
            lines.append(f"{name} [{state}]")
            lines.append(f"  {schedule}")
            if title:
                lines.append(f"  title: {title}")
            if prompt:
                lines.append(f"  prompt: {prompt[:160]}")
            data.append({
                "job_name": name,
                "enabled": job.get("enabled", True),
                "schedule": schedule,
                "title": title,
                "prompt": prompt,
            })
        return ToolResult(data=data, llm_summary="\n".join(lines))

    def _get_job(self, svc, job_name: str) -> ToolResult:
        job = _get_subagent_job_or_none(svc, job_name)
        if job is None:
            return ToolResult.failed(f"Unknown subagent job: '{job_name}'.")
        return self._format_job_result(svc, job_name, job, created=None)

    def _format_job_result(self, svc, job_name: str, job: dict, created: bool | None) -> ToolResult:
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
