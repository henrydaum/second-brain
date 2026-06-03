from datetime import datetime, timedelta

from plugins.services.service_timekeeper import TimekeeperService


def _job(**kwargs):
    data = {"enabled": True, "channel": "test.event", "payload": {}, "one_time": False, "cron": "* * * * *"}
    data.update(kwargs)
    return data


def test_timekeeper_load_purges_expired_one_time_jobs(monkeypatch):
    now = datetime.now().astimezone()
    config = {"scheduled_jobs": {
        "past": _job(one_time=True, cron=None, run_at=(now - timedelta(days=1)).isoformat()),
        "future": _job(one_time=True, cron=None, run_at=(now + timedelta(days=1)).isoformat()),
        "cron": _job(),
    }}
    saved = {}
    monkeypatch.setattr("config.config_manager.load_plugin_config", lambda: {"scheduled_jobs": config["scheduled_jobs"], "other": "kept"})
    monkeypatch.setattr("config.config_manager.save_plugin_config", lambda values: saved.update(values))
    service = TimekeeperService(config)
    service._poll_interval_s = 3600

    service.load()
    service.unload()

    assert sorted(service.list_jobs()) == ["cron", "future"]
    assert sorted(config["scheduled_jobs"]) == ["cron", "future"]
    assert sorted(saved["scheduled_jobs"]) == ["cron", "future"]
    assert saved["other"] == "kept"


def test_timekeeper_constructor_does_not_persist_expired_jobs(monkeypatch):
    now = datetime.now().astimezone()
    config = {"scheduled_jobs": {
        "past": _job(one_time=True, cron=None, run_at=(now - timedelta(days=1)).isoformat()),
    }}
    monkeypatch.setattr("config.config_manager.save_plugin_config", lambda _values: (_ for _ in ()).throw(AssertionError("should not persist")))

    service = TimekeeperService(config)

    assert sorted(service.list_jobs()) == ["past"]
