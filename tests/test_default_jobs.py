"""Default Timekeeper job lifecycle: BaseTask.default_jobs.

Covers the kernel half of "installing a scheduling task should schedule
it": the orchestrator seeds declared default jobs at registration
(existing jobs — including disabled ones — are left alone) and removes
them at unregistration, so a task's default jobs live exactly as long as
the task does and a reinstall picks up updated declarations.
"""

from types import SimpleNamespace

from pipeline.orchestrator import Orchestrator
from plugins.BaseTask import BaseTask
from plugins.services.service_timekeeper import TimekeeperService
from runtime.runtime_approvals import _sane_enum


class _FakeTimekeeper:
    def __init__(self, existing=()):
        self.jobs = {name: {"channel": "x"} for name in existing}
        self.created = {}
        self.removed = []

    def get_job(self, name):
        return self.jobs.get(name)

    def create_job(self, name, job_def):
        self.jobs[name] = dict(job_def)
        self.created[name] = dict(job_def)

    def remove_job(self, name):
        self.removed.append(name)
        return self.jobs.pop(name, None) is not None


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


def test_unregister_removes_default_jobs():
    tk = _FakeTimekeeper()
    orch = _orchestrator(tk)
    orch.register_task(_SeederTask())
    assert "seed_job" in tk.jobs

    orch.unregister_task("seeder")

    assert tk.removed == ["seed_job"]
    assert "seed_job" not in tk.jobs


def test_reinstall_reseeds_updated_declaration():
    # Uninstall + reinstall with a changed cron: the old job is removed at
    # unregistration, so the new registration seeds the new schedule.
    tk = _FakeTimekeeper()
    orch = _orchestrator(tk)
    orch.register_task(_SeederTask())
    orch.unregister_task("seeder")

    class _Updated(_SeederTask):
        default_jobs = {"seed_job": {"channel": "seed.chan", "cron": "* * * * *", "payload": {}}}

    orch.register_task(_Updated())
    assert tk.jobs["seed_job"]["cron"] == "* * * * *"


def test_task_without_default_jobs_needs_no_timekeeper():
    class _Plain(BaseTask):
        name = "plain"
        trigger = "event"
        trigger_channels = ["plain.chan"]

    db = SimpleNamespace(ensure_output_table=lambda *a, **k: None, register_task=lambda **k: None)
    Orchestrator(db, {"max_workers": 1}, {}).register_task(_Plain())  # must not raise


def test_timekeeper_remove_and_recreate(monkeypatch):
    saved = {}
    monkeypatch.setattr("config.config_manager.load_plugin_config", lambda: {})
    monkeypatch.setattr("config.config_manager.save_plugin_config", lambda values: saved.update(values))
    config = {"scheduled_jobs": {"cron": {"enabled": True, "channel": "t", "payload": {}, "cron": "* * * * *"}}}
    service = TimekeeperService(config)

    assert service.remove_job("cron") is True
    assert service.get_job("cron") is None
    assert saved["scheduled_jobs"] == {}
    assert service.remove_job("cron") is False  # already gone

    service.create_job("cron", {"channel": "t", "cron": "* * * * *"})
    assert service.get_job("cron") is not None


def test_sane_enum_drops_unanswerable_choices():
    # A request whose every choice renders empty would wedge the session —
    # the kernel treats it as free-form input instead.
    assert _sane_enum(["", "  ", ""]) is None
    assert _sane_enum(["a", "", "b"]) == ["a", "b"]
    assert _sane_enum(None) is None
    assert _sane_enum([]) is None
    assert _sane_enum([True, False]) == [True, False]
