"""Scheduled event service running on the shared resident poll loop."""

import json
from copy import deepcopy
from datetime import datetime

from cron_descriptor import ExpressionDescriptor
from croniter import croniter

from guest.bases import BaseService


def _now_local() -> datetime:
    """Return an aware timestamp in the host's local timezone."""
    return datetime.now().astimezone()


def _local_tz():
    """Return the host's local timezone."""
    return _now_local().tzinfo


class TimekeeperService(BaseService):
    """Persist schedules and emit their events when they become due."""

    name = "timekeeper"
    description = "Persist schedules and emit their events when due."
    shared = True
    poll_interval = 1.0
    max_poll_failures = 5
    dependencies_pip = ["croniter", "cron-descriptor"]
    requests = ["config.read", "config.write", "event.emit"]
    exports = [
        "list_jobs",
        "get_job",
        "create_job",
        "update_job",
        "remove_job",
        "enable_job",
        "cron_to_text",
        "get_next_fire_at",
        "describe_job",
    ]
    config_settings = [
        (
            "Scheduled Jobs",
            "scheduled_jobs",
            "JSON object keyed by job name describing scheduled event "
            "emissions.",
            {},
            {"type": "text", "hidden": True},
        ),
    ]

    def __init__(self):
        self._jobs = {}
        self._next_fire_at = {}

    def start(self, sdk):
        """Load persisted jobs and discard expired one-time schedules."""
        self._load_jobs_from_config(sdk, purge_expired=True)
        return True

    def stop(self, sdk):
        """No guest thread or external resource needs teardown."""
        return None

    def poll(self, sdk):
        """Run one scheduler tick; the kernel supplies the cadence."""
        self._tick(sdk)
        return False

    def list_jobs(self, sdk) -> dict:
        """Return every normalized job."""
        return deepcopy(self._jobs)

    def get_job(self, sdk, name: str) -> dict | None:
        """Return one job."""
        job = self._jobs.get(name)
        return deepcopy(job) if job is not None else None

    def create_job(self, sdk, name: str, job_def: dict) -> dict:
        """Create and persist a schedule."""
        if name in self._jobs:
            raise ValueError(f"Job '{name}' already exists.")
        normalized = self._normalize_job(name, job_def)
        self._jobs[name] = normalized
        self._next_fire_at[name] = self._compute_next_fire(
            normalized,
            from_time=_now_local(),
        )
        self._persist_jobs(sdk)
        return deepcopy(normalized)

    def update_job(self, sdk, name: str, patch: dict) -> dict:
        """Update and persist a schedule."""
        current = self._jobs.get(name)
        if current is None:
            raise ValueError(f"Unknown job: '{name}'.")
        merged = deepcopy(current)
        merged.update(deepcopy(patch or {}))
        normalized = self._normalize_job(name, merged)
        self._jobs[name] = normalized
        self._next_fire_at[name] = self._compute_next_fire(
            normalized,
            from_time=_now_local(),
        )
        self._persist_jobs(sdk)
        return deepcopy(normalized)

    def remove_job(self, sdk, name: str) -> bool:
        """Remove and persist a schedule."""
        removed = self._jobs.pop(name, None)
        self._next_fire_at.pop(name, None)
        if removed is None:
            return False
        self._persist_jobs(sdk)
        return True

    def enable_job(
        self,
        sdk,
        name: str,
        enabled: bool = True,
    ) -> dict:
        """Enable or disable one schedule."""
        return self.update_job(sdk, name, {"enabled": bool(enabled)})

    def cron_to_text(self, sdk, expr: str) -> str:
        """Describe a cron expression in natural language."""
        return self._cron_to_text(expr)

    def get_next_fire_at(self, sdk, name: str) -> str | None:
        """Return the next fire time as a wire-safe ISO timestamp."""
        job = self._jobs.get(name)
        if job is None or not job.get("enabled", True):
            return None
        cached = self._next_fire_at.get(name)
        if cached is None:
            cached = self._compute_next_fire(job, from_time=_now_local())
        return cached.isoformat() if cached is not None else None

    def describe_job(self, sdk, name: str) -> str:
        """Describe one schedule."""
        job = self._jobs.get(name)
        if job is None:
            raise ValueError(f"Unknown job: '{name}'.")
        if job["one_time"]:
            return f"One-time at {job['run_at']}"
        return self._cron_to_text(job["cron"])

    def _tick(self, sdk):
        """Emit due jobs and advance or remove them."""
        now = _now_local()
        due = []
        for name, job in self._jobs.items():
            if not job.get("enabled", True):
                continue
            next_fire = self._next_fire_at.get(name)
            if next_fire is None:
                next_fire = self._compute_next_fire(job, from_time=now)
                self._next_fire_at[name] = next_fire
            if next_fire is not None and next_fire <= now:
                due.append((name, deepcopy(job), next_fire))

        for name, job, scheduled_for in due:
            self._emit_job(sdk, name, job, scheduled_for)

    def _emit_job(
        self,
        sdk,
        name: str,
        job: dict,
        scheduled_for: datetime,
    ):
        """Emit one job and update its next occurrence."""
        emitted_at = _now_local()
        payload = deepcopy(job.get("payload", {}))
        payload["_timekeeper"] = {
            "job_name": name,
            "scheduled_for": scheduled_for.isoformat(),
            "emitted_at": emitted_at.isoformat(),
            "one_time": job["one_time"],
            "source": "timekeeper",
        }
        sdk.log(
            f"Emitting scheduled event '{job['channel']}' for job '{name}'"
        )
        sdk.events.emit(job["channel"], payload)

        current = self._jobs.get(name)
        if current is None:
            return
        if current["one_time"]:
            self._jobs.pop(name, None)
            self._next_fire_at.pop(name, None)
            self._persist_jobs(sdk)
        else:
            self._next_fire_at[name] = self._compute_next_fire(
                current,
                from_time=scheduled_for,
            )

    def _load_jobs_from_config(self, sdk, purge_expired: bool = False):
        """Load, normalize, and schedule persisted jobs."""
        raw = sdk.config.read("scheduled_jobs") or {}
        if isinstance(raw, str):
            raw = raw.strip()
            raw = json.loads(raw) if raw else {}
        if not isinstance(raw, dict):
            raise ValueError(
                "scheduled_jobs must be a JSON object keyed by job name."
            )

        jobs = {}
        next_fire = {}
        now = _now_local()
        purged = []
        for name, job_def in raw.items():
            if not isinstance(job_def, dict):
                raise ValueError(f"Job '{name}' must be an object.")
            normalized = self._normalize_job(name, job_def)
            if (
                purge_expired
                and normalized["one_time"]
                and self._parse_datetime(normalized["run_at"], name) < now
            ):
                purged.append(name)
                continue
            jobs[name] = normalized
            next_fire[name] = self._compute_next_fire(
                normalized,
                from_time=now,
            )

        self._jobs = jobs
        self._next_fire_at = next_fire
        if purged:
            sdk.log(
                "Purged expired one-time job(s): "
                + ", ".join(sorted(purged))
            )
            self._persist_jobs(sdk)

    def _normalize_job(self, name: str, job_def: dict) -> dict:
        """Validate and normalize one job definition."""
        job = {
            "enabled": bool(job_def.get("enabled", True)),
            "channel": (job_def.get("channel") or "").strip(),
            "cron": job_def.get("cron"),
            "run_at": job_def.get("run_at"),
            "one_time": bool(job_def.get("one_time", False)),
            "payload": deepcopy(job_def.get("payload", {})),
        }
        if not job["channel"]:
            raise ValueError(
                f"Job '{name}' is missing required field 'channel'."
            )
        if not isinstance(job["payload"], dict):
            raise ValueError(
                f"Job '{name}' payload must be a JSON object."
            )
        try:
            json.dumps(job["payload"])
        except TypeError as exc:
            raise ValueError(
                f"Job '{name}' payload must be JSON-serializable: {exc}"
            ) from exc

        if job["one_time"]:
            if not job["run_at"]:
                raise ValueError(
                    f"One-time job '{name}' requires 'run_at'."
                )
            if job["cron"]:
                raise ValueError(
                    f"One-time job '{name}' must not define 'cron'."
                )
            run_at = self._parse_datetime(job["run_at"], name)
            job["run_at"] = run_at.isoformat()
            job["cron"] = None
        else:
            if not job["cron"]:
                raise ValueError(
                    f"Repeating job '{name}' requires 'cron'."
                )
            if job["run_at"]:
                raise ValueError(
                    f"Repeating job '{name}' must not define 'run_at'."
                )
            try:
                croniter(job["cron"], _now_local())
            except Exception as exc:
                raise ValueError(
                    f"Job '{name}' has invalid cron expression: {exc}"
                ) from exc
            job["run_at"] = None
        return job

    def _compute_next_fire(
        self,
        job: dict,
        from_time: datetime,
    ) -> datetime | None:
        """Compute the next eligible fire time."""
        if not job.get("enabled", True):
            return None
        if job["one_time"]:
            run_at = self._parse_datetime(job["run_at"], "one_time job")
            return run_at if run_at >= from_time else None
        return croniter(job["cron"], from_time).get_next(datetime)

    def _persist_jobs(self, sdk):
        """Persist normalized state through the service-owned setting."""
        sdk.config.write(
            "scheduled_jobs",
            deepcopy(self._jobs),
            scope="plugin",
        )

    @staticmethod
    def _cron_to_text(expr: str) -> str:
        try:
            return ExpressionDescriptor(expr).get_description()
        except Exception as exc:
            raise ValueError(f"Invalid cron expression: {exc}") from exc

    @staticmethod
    def _parse_datetime(value: str, job_name: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Job '{job_name}' has invalid run_at datetime: {exc}"
            ) from exc
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=_local_tz())
        return parsed.astimezone()
