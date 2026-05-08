from datetime import datetime
from types import SimpleNamespace

from plugins.commands.command_schedule import ADD, ScheduleCommand
from plugins.commands.command_tasks import TasksCommand


class FakeTask:
    name = "spawn_subagent"
    trigger = "event"
    trigger_channels = ["subagent.spawn"]
    requires_services = []
    event_payload_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Prompt"},
            "title": {"type": "string", "description": "Title"},
        },
        "required": ["prompt"],
    }


class FakeDb:
    def get_system_stats(self):
        return {"tasks": {"spawn_subagent": {"DONE": 3, "FAILED": 1}}}

    def get_run_stats(self):
        return {}


class FakeTimekeeper:
    loaded = True

    def __init__(self):
        self.jobs = {
            "nightly_wisdom": {
                "enabled": True,
                "channel": "subagent.spawn",
                "cron": "0 20 * * *",
                "run_at": None,
                "one_time": False,
                "payload": {"prompt": "reflect", "title": "Nightly Wisdom"},
            }
        }

    def list_jobs(self):
        return {k: dict(v, payload=dict(v.get("payload") or {})) for k, v in self.jobs.items()}

    def get_job(self, name):
        job = self.jobs.get(name)
        return dict(job, payload=dict(job.get("payload") or {})) if job else None

    def create_job(self, name, job_def):
        if name in self.jobs:
            raise ValueError("exists")
        self.jobs[name] = {"enabled": True, "run_at": None, "one_time": False, **job_def, "payload": dict(job_def.get("payload") or {})}
        return self.get_job(name)

    def update_job(self, name, patch):
        self.jobs[name].update(patch)
        if "payload" in patch:
            self.jobs[name]["payload"] = dict(patch["payload"])
        return self.get_job(name)

    def remove_job(self, name):
        return self.jobs.pop(name, None) is not None

    def enable_job(self, name, enabled=True):
        return self.update_job(name, {"enabled": bool(enabled)})

    def cron_to_text(self, _cron):
        return "At 08:00 PM"

    def get_next_fire_at(self, name):
        return datetime(2026, 5, 8, 20, 0) if self.jobs.get(name, {}).get("enabled", True) else None


def context():
    return SimpleNamespace(
        services={"timekeeper": FakeTimekeeper()},
        orchestrator=SimpleNamespace(tasks={"spawn_subagent": FakeTask()}, paused=set()),
        db=FakeDb(),
    )


def test_tasks_event_actions_do_not_include_schedule_buttons():
    step = TasksCommand().form({"task_name": "spawn_subagent"}, context())[1]

    assert step.enum == ["pause", "unpause", "trigger"]
    assert "schedule" not in step.enum
    assert "unschedule" not in step.enum


def test_tasks_event_prompt_uses_short_schedule_hint():
    prompt = TasksCommand().form({"task_name": "spawn_subagent"}, context())[1].prompt

    assert "Scheduled jobs: 1. Use /schedule to manage them." in prompt
    assert "nightly_wisdom:" not in prompt
    assert "Schedules:" not in prompt


def test_schedule_form_lists_jobs_and_add():
    step = ScheduleCommand().form({}, context())[0]

    assert step.enum == ["nightly_wisdom", ADD]
    assert step.enum_labels == ["nightly_wisdom", "Schedule new job"]


def test_schedule_add_creates_event_task_job_with_schema_payload():
    ctx = context()
    out = ScheduleCommand().run({"job_name": ADD, "task_name": "spawn_subagent", "new_job_name": "trash", "cron": "3 17 * * *", "prompt": "take out trash", "title": "Trash"}, ctx)

    assert "Created schedule 'trash'" in out
    assert ctx.services["timekeeper"].jobs["trash"]["channel"] == "subagent.spawn"
    assert ctx.services["timekeeper"].jobs["trash"]["payload"] == {"prompt": "take out trash", "title": "Trash"}


def test_schedule_job_actions_edit_delete_enable_disable():
    ctx = context()
    cmd = ScheduleCommand()

    assert cmd.run({"job_name": "nightly_wisdom", "action": "disable"}, ctx) == "Disabled job: nightly_wisdom"
    assert ctx.services["timekeeper"].jobs["nightly_wisdom"]["enabled"] is False
    assert cmd.run({"job_name": "nightly_wisdom", "action": "enable"}, ctx) == "Enabled job: nightly_wisdom"
    assert cmd.run({"job_name": "nightly_wisdom", "action": "edit", "cron": "5 21 * * *", "prompt": "new", "title": "New"}, ctx) == "Updated job: nightly_wisdom"
    assert ctx.services["timekeeper"].jobs["nightly_wisdom"]["cron"] == "5 21 * * *"
    assert ctx.services["timekeeper"].jobs["nightly_wisdom"]["payload"] == {"prompt": "new", "title": "New"}
    assert cmd.run({"job_name": "nightly_wisdom", "action": "delete"}, ctx) == "Deleted job: nightly_wisdom"
    assert "nightly_wisdom" not in ctx.services["timekeeper"].jobs
