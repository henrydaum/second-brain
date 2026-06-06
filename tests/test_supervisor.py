"""Tests for the plugin supervisor (``runtime/supervisor.py``).

Covers the single supervised call site: timeout/abandon, the strike-counting
circuit breaker, the built-in exemption (eligible=False never quarantines), and
clean-run / clear resets.
"""

import time

import pytest

from events.event_bus import bus
from events.event_channels import PLUGIN_QUARANTINE_REQUESTED
from runtime.supervisor import run_supervised, supervisor


@pytest.fixture(autouse=True)
def _fresh_supervisor():
    """Give each test a clean, enabled supervisor."""
    supervisor.configure({"plugin_supervisor": True})
    yield
    # Reset internal health so keys don't leak across tests.
    supervisor.health._strikes.clear()
    supervisor.health._quarantined.clear()


def _capture_quarantine():
    """Subscribe and return (events, unsubscribe)."""
    events = []
    unsub = bus.subscribe(PLUGIN_QUARANTINE_REQUESTED, lambda p: events.append(p))
    return events, unsub


def test_success_returns_value():
    res = run_supervised(lambda: 42, timeout=5, plugin_key="k/ok", kind="tool", name="ok")
    assert res.ok and res.value == 42 and not res.timed_out


def test_exception_is_not_a_crash_for_caller_but_records_strike():
    def boom():
        raise ValueError("nope")
    res = run_supervised(boom, timeout=5, plugin_key="k/boom", kind="tool", name="boom")
    assert not res.ok and "nope" in res.error and not res.timed_out


def test_timeout_abandons_and_flags():
    res = run_supervised(lambda: time.sleep(0.4) or 1, timeout=0.1,
                         plugin_key="k/slow", kind="tool", name="slow")
    assert not res.ok and res.timed_out and "timed out" in res.error


def test_circuit_breaker_trips_after_threshold():
    events, unsub = _capture_quarantine()
    try:
        def boom():
            raise RuntimeError("crash")
        # Default threshold = 3: no trip on the first two, trip on the third.
        run_supervised(boom, timeout=5, plugin_key="k/bad", kind="tool",
                       name="bad", eligible=True)
        run_supervised(boom, timeout=5, plugin_key="k/bad", kind="tool",
                       name="bad", eligible=True)
        assert events == []
        run_supervised(boom, timeout=5, plugin_key="k/bad", kind="tool",
                       name="bad", eligible=True)
        assert len(events) == 1
        payload = events[0]
        assert payload["plugin_type"] == "tool"
        assert payload["source_path"] == "k/bad"
        assert payload["name"] == "bad"
    finally:
        unsub()


def test_builtin_is_never_quarantined():
    events, unsub = _capture_quarantine()
    try:
        def boom():
            raise RuntimeError("crash")
        for _ in range(5):
            run_supervised(boom, timeout=5, plugin_key="k/builtin", kind="tool",
                           name="builtin", eligible=False)
        assert events == []  # supervised + logged, but never quarantined
    finally:
        unsub()


def test_clean_run_resets_the_window():
    events, unsub = _capture_quarantine()
    try:
        def boom():
            raise RuntimeError("crash")
        run_supervised(boom, timeout=5, plugin_key="k/mix", kind="tool",
                       name="mix", eligible=True)
        # A clean run clears the strike window…
        run_supervised(lambda: 1, timeout=5, plugin_key="k/mix", kind="tool", name="mix")
        # …so a single subsequent strike must not trip (would need 2 again).
        run_supervised(boom, timeout=5, plugin_key="k/mix", kind="tool",
                       name="mix", eligible=True)
        assert events == []
    finally:
        unsub()


def test_watcher_unloads_on_quarantine_event(monkeypatch):
    """End-to-end: a quarantine request makes the watcher unload + notify."""
    import plugins.services.service_plugin_watcher as watcher_mod
    from events.event_channels import CHAT_MESSAGE_PUSHED, PLUGIN_QUARANTINED

    calls = {}
    monkeypatch.setattr(watcher_mod, "unload_plugin",
                        lambda *a, **k: calls.update(args=a, kwargs=k))

    svc = watcher_mod.PluginWatcherService({})
    notices, done = [], []
    unsub_notice = bus.subscribe(CHAT_MESSAGE_PUSHED, lambda p: notices.append(p))
    unsub_done = bus.subscribe(PLUGIN_QUARANTINED, lambda p: done.append(p))
    try:
        svc._on_quarantine({
            "plugin_type": "tool", "source_path": "k/bad",
            "name": "bad", "reason": "crash x2",
        })
        assert calls["kwargs"]["source_path"] == "k/bad"
        assert calls["args"][0] == "tool"
        assert any("bad" in n["message"] for n in notices)
        assert done and done[0]["name"] == "bad"
    finally:
        unsub_notice()
        unsub_done()


def test_disabled_supervisor_does_not_quarantine():
    supervisor.configure({"plugin_supervisor": False})
    events, unsub = _capture_quarantine()
    try:
        def boom():
            raise RuntimeError("crash")
        for _ in range(5):
            run_supervised(boom, timeout=5, plugin_key="k/off", kind="tool",
                           name="off", eligible=True)
        assert events == []
    finally:
        unsub()
