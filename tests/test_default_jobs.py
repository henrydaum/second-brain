"""Default Timekeeper job seeding: BaseTask.default_jobs + tombstones.

Covers the kernel half of "installing a scheduling task should schedule
it": the orchestrator seeds declared default jobs at registration, the
timekeeper tombstones deliberate removals so re-registration (boot,
install, hot-reload) does not resurrect them, and explicit re-creation
revives a tombstoned name.
"""

from types import SimpleNamespace

from pipeline.orchestrator import Orchestrator
from plugins.BaseTask import BaseTask
from plugins.services.service_timekeeper import TimekeeperService
from runtime.runtime_approvals import _sane_enum


class _FakeTimekeeper:
    def __init__(self, existing=(), removed=()):
        self.jobs = {name: {"channel": "x"} for name in existing}
        self.removed = set(removed)
        self.created = {}

    def get_job(self, name):
        return self.jobs.get(name)

    def is_job_removed(self, name):
        return name in self.removed

    def create_job(self, name, job_def):
        self.jobs[name] = dict(job_def)
        self.created[name] = dict(job_def)


class _SeederTask(BaseTask):
    name = "seeder"
    trigger = "event"
    trigger_channels = ["seed.chan"]
    default_jobs = {"seed_job": {"channel": "seed.chan", "cron": "*/15 * * * *", "payload": {}}}


def _orchestrator(tk):
    db = SimpleNamespace(
        ensure_output_table=lambda *a, **k: None,
        register_task=lambda **k: None,
    )
    orch = Orchestrator(db, {"max_workers": 1}, {"timekeeper": tk})
    return orch


def test_register_task_seeds_declared_default_jobs():
    tk = _FakeTimekeeper()
    _orchestrator(tk).register_task(_SeederTask())
    assert tk.created["seed_job"]["cron"] == "*/15 * * * *"
    assert tk.created["seed_job"]["channel"] == "seed.chan"


def test_seeding_skips_existing_jobs():
    tk = _FakeTimekeeper(existing=["seed_job"])
    _orchestrator(tk).register_task(_SeederTask())
    assert tk.created == {}


def test_seeding_respects_tombstones():
    tk = _FakeTimekeeper(removed=["seed_job"])
    _orchestrator(tk).register_task(_SeederTask())
    assert tk.created == {}


def test_task_without_default_jobs_needs_no_timekeeper():
    class _Plain(BaseTask):
        name = "plain"
        trigger = "event"
        trigger_channels = ["plain.chan"]

    db = SimpleNamespace(ensure_output_table=lambda *a, **k: None, register_task=lambda **k: None)
    Orchestrator(db, {"max_workers": 1}, {}).register_task(_Plain())  # must not raise


def test_timekeeper_remove_tombstones_and_create_revives(monkeypatch):
    saved = {}
    monkeypatch.setattr("config.config_manager.load_plugin_config", lambda: {})
    monkeypatch.setattr("config.config_manager.save_plugin_config", lambda values: saved.update(values))
    config = {"scheduled_jobs": {"cron": {"enabled": True, "channel": "t", "payload": {}, "cron": "* * * * *"}}}
    service = TimekeeperService(config)

    assert service.remove_job("cron") is True
    assert service.is_job_removed("cron")
    assert saved["removed_scheduled_jobs"] == ["cron"]
    assert config["removed_scheduled_jobs"] == ["cron"]

    service.create_job("cron", {"channel": "t", "cron": "* * * * *"})
    assert not service.is_job_removed("cron")
    assert saved["removed_scheduled_jobs"] == []


def test_timekeeper_tombstones_load_from_config_string():
    service = TimekeeperService({"scheduled_jobs": {}, "removed_scheduled_jobs": '["dead"]'})
    assert service.is_job_removed("dead")
    assert not service.is_job_removed("alive")


def test_sane_enum_drops_unanswerable_choices():
    # A request whose every choice renders empty would wedge the session —
    # the kernel treats it as free-form input instead.
    assert _sane_enum(["", "  ", ""]) is None
    assert _sane_enum(["a", "", "b"]) == ["a", "b"]
    assert _sane_enum(None) is None
    assert _sane_enum([]) is None
    assert _sane_enum([True, False]) == [True, False]
