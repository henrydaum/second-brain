"""Sandboxed Timekeeper lifecycle, exports, and live polling."""

import threading
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from events.event_bus import bus
from sandbox import Sandbox
from sandbox.bridge import adapt, configure


def _job(**kwargs):
    job = {
        "enabled": True,
        "channel": "test.event",
        "payload": {},
        "one_time": False,
        "cron": "* * * * *",
    }
    job.update(kwargs)
    return job


@pytest.fixture
def timekeeper(monkeypatch):
    """Build the real adapter against an isolated kernel context."""
    sandboxes = []

    def build(config):
        saved = {}
        monkeypatch.setattr(
            "plugins.plugin_discovery.get_setting_plugin_names",
            lambda key: ["timekeeper"] if key == "scheduled_jobs" else [],
        )
        monkeypatch.setattr(
            "config.config_manager.save",
            lambda values: saved.update(values),
        )
        monkeypatch.setattr(
            "config.config_manager.load_plugin_config",
            lambda: {"other": "kept"},
        )
        monkeypatch.setattr(
            "config.config_manager.save_plugin_config",
            lambda values: saved.update(values),
        )
        context = SimpleNamespace(
            config=config,
            runtime=None,
            services={},
            db=None,
            user_id=1,
            session_key=None,
        )
        sandbox = Sandbox(context=context)
        configure(sandbox)
        module = adapt("plugins/services/service_timekeeper.py")
        service = module.build_services(config)["timekeeper"]
        context.services["timekeeper"] = service
        sandboxes.append((sandbox, service))
        return service, saved, context

    yield build

    for sandbox, service in reversed(sandboxes):
        service.unload()
        sandbox.shutdown()
    configure(None)


def test_load_purges_and_persists_expired_one_time_jobs(timekeeper):
    now = datetime.now().astimezone()
    config = {
        "scheduled_jobs": {
            "past": _job(
                one_time=True,
                cron=None,
                run_at=(now - timedelta(days=1)).isoformat(),
            ),
            "future": _job(
                one_time=True,
                cron=None,
                run_at=(now + timedelta(days=1)).isoformat(),
            ),
            "cron": _job(),
        },
    }
    service, saved, context = timekeeper(config)

    assert service.load()

    assert sorted(service.list_jobs()) == ["cron", "future"]
    assert sorted(context.config["scheduled_jobs"]) == ["cron", "future"]
    assert sorted(saved["scheduled_jobs"]) == ["cron", "future"]
    assert saved["other"] == "kept"


def test_exports_accept_positional_calls_and_persist(timekeeper):
    config = {"scheduled_jobs": {"cron": _job(channel="t")}}
    service, saved, _context = timekeeper(config)
    assert service.load()

    assert service.remove_job("cron") is True
    assert service.get_job("cron") is None
    assert saved["scheduled_jobs"] == {}
    assert service.remove_job("cron") is False

    created = service.create_job(
        "cron",
        {"channel": "t", "cron": "* * * * *"},
    )
    assert created["channel"] == "t"
    assert service.get_job("cron") == created
    assert isinstance(service.get_next_fire_at("cron"), str)


def test_kernel_poll_loop_emits_and_removes_one_time_job(timekeeper):
    now = datetime.now().astimezone()
    config = {
        "scheduled_jobs": {
            "soon": _job(
                one_time=True,
                cron=None,
                run_at=(now + timedelta(milliseconds=150)).isoformat(),
                payload={"value": 7},
            ),
        },
    }
    service, _saved, context = timekeeper(config)
    received = []
    fired = threading.Event()
    unsubscribe = bus.subscribe(
        "test.event",
        lambda payload: (received.append(payload), fired.set()),
    )
    try:
        assert service.load()
        assert fired.wait(2.5)
    finally:
        unsubscribe()

    assert received[0]["value"] == 7
    assert received[0]["_timekeeper"]["job_name"] == "soon"
    assert service.get_job("soon") is None
    assert context.config["scheduled_jobs"] == {}


def test_unload_stops_future_polls(timekeeper):
    service, _saved, _context = timekeeper(
        {"scheduled_jobs": {"cron": _job()}}
    )
    assert service.load()
    thread = service._poll_thread
    assert thread is not None and thread.is_alive()

    service.unload()

    assert not thread.is_alive()


# ────────────────────────────────────────────────────────────────────
# Task-declared default jobs (was test_default_jobs.py)
# ────────────────────────────────────────────────────────────────────

from pipeline.orchestrator import Orchestrator
from plugins.BaseTask import BaseTask
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


def test_sane_enum_drops_unanswerable_choices():
    # A request whose every choice renders empty would wedge the session —
    # the kernel treats it as free-form input instead.
    assert _sane_enum(["", "  ", ""]) is None
    assert _sane_enum(["a", "", "b"]) == ["a", "b"]
    assert _sane_enum(None) is None
    assert _sane_enum([]) is None
    assert _sane_enum([True, False]) == [True, False]
