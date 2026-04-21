import json
import logging
import threading
from copy import deepcopy
from datetime import datetime

from cron_descriptor import ExpressionDescriptor
from croniter import croniter

import config_manager
from Stage_0.BaseService import BaseService
from event_bus import bus
from event_channels import CHAT_MESSAGE_PUSHED

logger = logging.getLogger("TimekeeperService")


def _now_local() -> datetime:
    return datetime.now().astimezone()


def _local_tz():
    return _now_local().tzinfo


class TimekeeperService(BaseService):
    model_name = "Timekeeper"
    shared = True
    config_settings = [
        (
            "Scheduled Jobs",
            "scheduled_jobs",
            "JSON object keyed by job name describing scheduled event emissions. Managed via /schedule command.",
            {},
            {"type": "text", "hidden": True},
        ),
    ]

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._lock = threading.RLock()
        self._stop = None
        self._thread = None
        self._poll_interval_s = 1.0
        self._jobs: dict[str, dict] = {}
        self._next_fire_at: dict[str, datetime | None] = {}
        self._load_jobs_from_config()

    def _load(self) -> bool:
        with self._lock:
            self._load_jobs_from_config()
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, name="Timekeeper", daemon=True)
            self._thread.start()
            self.loaded = True
        return True

    def unload(self):
        stop = self._stop
        thread = self._thread
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            self._stop = None
            self._thread = None
            self.loaded = False

    def list_jobs(self) -> dict[str, dict]:
        with self._lock:
            return {name: deepcopy(job) for name, job in self._jobs.items()}

    def get_job(self, name: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(name)
            return deepcopy(job) if job is not None else None

    def create_job(self, name: str, job_def: dict) -> dict:
        with self._lock:
            if name in self._jobs:
                raise ValueError(f"Job '{name}' already exists.")
            normalized = self._normalize_job(name, job_def)
            self._jobs[name] = normalized
            self._next_fire_at[name] = self._compute_next_fire(normalized, from_time=_now_local())
            self._persist_jobs()
            return deepcopy(normalized)

    def update_job(self, name: str, patch: dict) -> dict:
        with self._lock:
            current = self._jobs.get(name)
            if current is None:
                raise ValueError(f"Unknown job: '{name}'.")
            merged = deepcopy(current)
            merged.update(deepcopy(patch or {}))
            normalized = self._normalize_job(name, merged)
            self._jobs[name] = normalized
            self._next_fire_at[name] = self._compute_next_fire(normalized, from_time=_now_local())
            self._persist_jobs()
            return deepcopy(normalized)

    def remove_job(self, name: str) -> bool:
        with self._lock:
            removed = self._jobs.pop(name, None)
            self._next_fire_at.pop(name, None)
            if removed is None:
                return False
            self._persist_jobs()
            return True

    def enable_job(self, name: str, enabled: bool = True) -> dict:
        return self.update_job(name, {"enabled": bool(enabled)})

    def cron_to_text(self, expr: str) -> str:
        try:
            return ExpressionDescriptor(expr).get_description()
        except Exception as e:
            raise ValueError(f"Invalid cron expression: {e}")

    def get_next_fire_at(self, name: str) -> datetime | None:
        """Return the next scheduled fire time for a job, or None if disabled/unknown/exhausted."""
        with self._lock:
            job = self._jobs.get(name)
            if job is None or not job.get("enabled", True):
                return None
            cached = self._next_fire_at.get(name)
            if cached is not None:
                return cached
            return self._compute_next_fire(job, from_time=_now_local())

    def describe_job(self, name: str) -> str:
        with self._lock:
            job = self._jobs.get(name)
            if job is None:
                raise ValueError(f"Unknown job: '{name}'.")
            if job["one_time"]:
                return f"One-time at {job['run_at']}"
            return self.cron_to_text(job["cron"])

    def _loop(self):
        while self._stop is not None and not self._stop.wait(self._poll_interval_s):
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Timekeeper tick failed: {e}", exc_info=True)

    def _tick(self):
        now = _now_local()
        due: list[tuple[str, dict, datetime]] = []

        with self._lock:
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
            self._emit_job(name, job, scheduled_for)

    def _emit_job(self, name: str, job: dict, scheduled_for: datetime):
        emitted_at = _now_local()
        payload = deepcopy(job.get("payload", {}))
        payload["_timekeeper"] = {
            "job_name": name,
            "scheduled_for": scheduled_for.isoformat(),
            "emitted_at": emitted_at.isoformat(),
            "one_time": job["one_time"],
            "source": "timekeeper",
        }

        logger.info(f"Emitting scheduled event '{job['channel']}' for job '{name}'")
        bus.emit(job["channel"], payload)

        title = (
            str(job.get("payload", {}).get("title") or "").strip()
            or str(job.get("description") or "").strip()
            or name
        )
        bus.emit(CHAT_MESSAGE_PUSHED, {
            "message": f"🕐 {title}",
            "title": "",
            "kind": "",
            "source": "timekeeper",
            "source_id": name,
        })

        with self._lock:
            current = self._jobs.get(name)
            if current is None:
                return

            if current["one_time"]:
                self._jobs.pop(name, None)
                self._next_fire_at.pop(name, None)
                self._persist_jobs()
            else:
                self._next_fire_at[name] = self._compute_next_fire(current, from_time=scheduled_for)

    def _load_jobs_from_config(self):
        raw = self._config.get("scheduled_jobs", {})
        if isinstance(raw, str):
            raw = raw.strip()
            raw = json.loads(raw) if raw else {}
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("scheduled_jobs must be a JSON object keyed by job name.")

        jobs: dict[str, dict] = {}
        next_fire: dict[str, datetime | None] = {}
        now = _now_local()
        for name, job_def in raw.items():
            if not isinstance(job_def, dict):
                raise ValueError(f"Job '{name}' must be an object.")
            normalized = self._normalize_job(name, job_def)
            jobs[name] = normalized
            next_fire[name] = self._compute_next_fire(normalized, from_time=now)

        self._jobs = jobs
        self._next_fire_at = next_fire

    def _normalize_job(self, name: str, job_def: dict) -> dict:
        job = {
            "enabled": bool(job_def.get("enabled", True)),
            "channel": (job_def.get("channel") or "").strip(),
            "cron": job_def.get("cron"),
            "run_at": job_def.get("run_at"),
            "one_time": bool(job_def.get("one_time", False)),
            "payload": deepcopy(job_def.get("payload", {})),
            "description": job_def.get("description"),
        }

        if not job["channel"]:
            raise ValueError(f"Job '{name}' is missing required field 'channel'.")

        if not isinstance(job["payload"], dict):
            raise ValueError(f"Job '{name}' payload must be a JSON object.")

        try:
            json.dumps(job["payload"])
        except TypeError as e:
            raise ValueError(f"Job '{name}' payload must be JSON-serializable: {e}")

        if job["one_time"]:
            if not job["run_at"]:
                raise ValueError(f"One-time job '{name}' requires 'run_at'.")
            if job["cron"]:
                raise ValueError(f"One-time job '{name}' must not define 'cron'.")
            run_at = self._parse_datetime(job["run_at"], name)
            job["run_at"] = run_at.isoformat()
            job["cron"] = None
        else:
            if not job["cron"]:
                raise ValueError(f"Repeating job '{name}' requires 'cron'.")
            if job["run_at"]:
                raise ValueError(f"Repeating job '{name}' must not define 'run_at'.")
            try:
                croniter(job["cron"], _now_local())
            except Exception as e:
                raise ValueError(f"Job '{name}' has invalid cron expression: {e}")
            job["run_at"] = None

        if job["description"] is not None:
            job["description"] = str(job["description"])

        return job

    def _compute_next_fire(self, job: dict, from_time: datetime) -> datetime | None:
        if not job.get("enabled", True):
            return None
        if job["one_time"]:
            run_at = self._parse_datetime(job["run_at"], "one_time job")
            return run_at if run_at >= from_time else None
        return croniter(job["cron"], from_time).get_next(datetime)

    def _persist_jobs(self):
        plugin_values = config_manager.load_plugin_config()
        plugin_values["scheduled_jobs"] = deepcopy(self._jobs)
        config_manager.save_plugin_config(plugin_values)
        self._config["scheduled_jobs"] = deepcopy(self._jobs)

    @staticmethod
    def _parse_datetime(value: str, job_name: str) -> datetime:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError as e:
            raise ValueError(f"Job '{job_name}' has invalid run_at datetime: {e}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_local_tz())
        else:
            dt = dt.astimezone()
        return dt


def build_services(config: dict) -> dict:
    return {"timekeeper": TimekeeperService(config)}
