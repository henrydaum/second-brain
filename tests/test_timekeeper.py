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
