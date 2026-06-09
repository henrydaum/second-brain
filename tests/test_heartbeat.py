"""Tests for the stall watchdog (``runtime/heartbeat.py``).

Covers both halves: the child-side probe registry + conditional beat
(``_tick`` writes the heartbeat file only while every probe is fresh) and the
launcher-side ``_StallMonitor`` decision logic (boot gating, staleness on an
injected monotonic clock, sleep grace, deleted-file handling).
"""

import time

import pytest

from runtime import heartbeat as hb
from runtime.heartbeat import (
    _Heartbeat, _StallMonitor, clamp_stall_timeout, heartbeat, read_mtime_ns,
)


@pytest.fixture(autouse=True)
def _fresh_heartbeat(tmp_path, monkeypatch):
    """Point the singleton at a temp file and reset its state per test."""
    monkeypatch.setattr(hb, "HEARTBEAT_FILE", tmp_path / "heartbeat")
    heartbeat.stop()
    heartbeat.unregister_all()
    yield
    heartbeat.stop()
    heartbeat.unregister_all()


# ── Probe registry ───────────────────────────────────────────────────

def test_fresh_probe_is_not_stale():
    probe = heartbeat.register("loop")
    assert heartbeat._stale_probes(time.monotonic()) == []
    assert probe.age(time.monotonic()) < 1.0


def test_probe_goes_stale_and_beat_refreshes():
    probe = heartbeat.register("loop", max_staleness=10.0)
    later = time.monotonic() + 11.0
    assert [p.name for p in heartbeat._stale_probes(later)] == ["loop"]
    probe.beat()
    probe._last = later - 1.0  # beaten 1s before "later"
    assert heartbeat._stale_probes(later) == []


def test_unregister_and_unregister_all():
    probe = heartbeat.register("a")
    heartbeat.register("b")
    probe.unregister()
    assert "a" not in heartbeat._probes and "b" in heartbeat._probes
    heartbeat.unregister("missing")  # no-op
    heartbeat.unregister_all()
    assert heartbeat._probes == {}


def test_reregister_replaces_stale_probe():
    old = heartbeat.register("loop", max_staleness=10.0)
    old._last -= 100.0  # very stale
    heartbeat.register("loop", max_staleness=10.0)  # fresh replacement
    assert heartbeat._stale_probes(time.monotonic()) == []


# ── Conditional beat (_tick) ─────────────────────────────────────────

def test_tick_writes_with_zero_probes():
    assert heartbeat._tick() is True
    assert hb.HEARTBEAT_FILE.exists()


def test_tick_writes_when_all_fresh():
    heartbeat.register("a")
    heartbeat.register("b")
    assert heartbeat._tick() is True


def test_tick_withholds_and_warns_when_stale(caplog):
    probe = heartbeat.register("dispatch", max_staleness=10.0)
    probe._last -= 50.0
    with caplog.at_level("WARNING", logger="Heartbeat"):
        assert heartbeat._tick() is False
    assert not hb.HEARTBEAT_FILE.exists()
    assert "dispatch" in caplog.text and "withholding" in caplog.text


def test_tick_recovers_after_beat(caplog):
    probe = heartbeat.register("dispatch", max_staleness=10.0)
    probe._last -= 50.0
    assert heartbeat._tick() is False
    probe.beat()
    with caplog.at_level("INFO", logger="Heartbeat"):
        assert heartbeat._tick() is True
    assert hb.HEARTBEAT_FILE.exists()
    assert "recovered" in caplog.text


def test_tick_recreates_deleted_file():
    assert heartbeat._tick() is True
    first = read_mtime_ns(hb.HEARTBEAT_FILE)
    hb.HEARTBEAT_FILE.unlink()
    assert heartbeat._tick() is True
    assert read_mtime_ns(hb.HEARTBEAT_FILE) is not None
    assert first is not None


# ── _StallMonitor ────────────────────────────────────────────────────

class _Clock:
    """Injected clock pair: monotonic and wall advance together by default."""

    def __init__(self):
        self.mono = 1000.0
        self.wall = 50_000.0

    def advance(self, seconds, wall_seconds=None):
        self.mono += seconds
        self.wall += seconds if wall_seconds is None else wall_seconds


def _monitor(clock, stall_timeout=120.0, baseline=None):
    return _StallMonitor(stall_timeout, baseline,
                         monotonic=lambda: clock.mono, wall=lambda: clock.wall)


def _poll_for(mon, clock, seconds, mtime):
    """Advance in launcher-sized 5s polls (a single big jump would — by
    design — look like a suspend and trigger grace). Returns True if any
    poll reported a stall."""
    elapsed = 0.0
    while elapsed < seconds:
        clock.advance(5)
        elapsed += 5
        if mon.observe(mtime):
            return True
    return False


def test_boot_never_enforced_without_first_beat():
    clock = _Clock()
    mon = _monitor(clock, baseline=111)
    for _ in range(1000):  # ~83 minutes of boot silence
        clock.advance(5)
        assert mon.observe(111) is False  # baseline mtime never changes
    clock.advance(5)
    assert mon.observe(None) is False  # file missing during boot is also fine


def test_stall_after_first_beat_then_silence():
    clock = _Clock()
    mon = _monitor(clock, stall_timeout=120.0, baseline=111)
    clock.advance(5)
    assert mon.observe(222) is False  # first beat arms the monitor
    elapsed = 0.0
    stalled_at = None
    while elapsed <= 130:
        clock.advance(5)
        elapsed += 5
        if mon.observe(222):
            stalled_at = elapsed
            break
    assert stalled_at is not None and stalled_at > 120.0


def test_beat_mid_staleness_resets_progress():
    clock = _Clock()
    mon = _monitor(clock, stall_timeout=120.0, baseline=None)
    assert mon.observe(1) is False  # armed
    assert _poll_for(mon, clock, 115, 1) is False
    assert mon.observe(2) is False  # progress just in time
    assert _poll_for(mon, clock, 115, 2) is False  # only ~115s since progress
    assert _poll_for(mon, clock, 15, 2) is True


def test_deleted_file_after_arming_counts_as_silence():
    clock = _Clock()
    mon = _monitor(clock, stall_timeout=120.0, baseline=None)
    assert mon.observe(1) is False  # armed
    assert _poll_for(mon, clock, 60, None) is False  # deleted; not past timeout
    assert _poll_for(mon, clock, 65, None) is True
    # File reappears with a new mtime => progress again.
    mon2 = _monitor(clock, stall_timeout=120.0, baseline=None)
    mon2.observe(1)
    assert _poll_for(mon2, clock, 100, None) is False
    clock.advance(5)
    assert mon2.observe(7) is False  # recreated file = progress
    assert _poll_for(mon2, clock, 110, 7) is False  # only ~110s since progress


def test_sleep_grace_suppresses_kill_after_wake():
    clock = _Clock()
    mon = _monitor(clock, stall_timeout=120.0, baseline=None)
    assert mon.observe(1) is False  # armed
    clock.advance(7200)  # 2h suspend: monotonic AND wall jump (Windows QPC)
    assert mon.observe(1) is False  # stale mtime, but grace kicks in
    # Still graced until stall_timeout after wake...
    assert _poll_for(mon, clock, 115, 1) is False
    # ...but a child that never beats again after the grace window is a stall.
    assert _poll_for(mon, clock, 130, 1) is True


def test_beat_during_grace_rearms_cleanly():
    clock = _Clock()
    mon = _monitor(clock, stall_timeout=120.0, baseline=None)
    mon.observe(1)
    clock.advance(7200)
    assert mon.observe(1) is False  # grace
    clock.advance(10)
    assert mon.observe(2) is False  # beat during grace
    assert _poll_for(mon, clock, 115, 2) is False  # ~115s since beat
    assert _poll_for(mon, clock, 130, 2) is True


def test_backward_wall_jump_is_inert():
    clock = _Clock()
    mon = _monitor(clock, stall_timeout=120.0, baseline=None)
    mon.observe(1)
    clock.advance(5, wall_seconds=-3600)  # clock set back an hour
    assert mon.observe(2) is False
    assert _poll_for(mon, clock, 130, 2) is True  # stall detection unaffected


# ── Helpers ──────────────────────────────────────────────────────────

def test_clamp_stall_timeout():
    assert clamp_stall_timeout(0) == 0.0
    assert clamp_stall_timeout(-5) == 0.0
    assert clamp_stall_timeout(10) == 30.0   # floored to MIN_STALL_TIMEOUT
    assert clamp_stall_timeout(120) == 120.0
    assert clamp_stall_timeout("garbage") == 120.0
    assert clamp_stall_timeout(None) == 120.0


def test_read_mtime_ns_missing_file(tmp_path):
    assert read_mtime_ns(tmp_path / "nope") is None


# ── Semi-integration: real beat thread + monitor, fast timeouts ──────

def test_thread_beats_flow_and_stall_detected(monkeypatch):
    monkeypatch.setattr(hb, "_BEAT_INTERVAL", 0.05)
    local = _Heartbeat()
    local.start()
    try:
        deadline = time.time() + 2.0
        while read_mtime_ns(hb.HEARTBEAT_FILE) is None and time.time() < deadline:
            time.sleep(0.02)
        assert read_mtime_ns(hb.HEARTBEAT_FILE) is not None

        # Healthy: monitor sees progress, never stalls.
        mon = _StallMonitor(0.5, None)
        for _ in range(6):
            time.sleep(0.1)
            assert mon.observe(read_mtime_ns(hb.HEARTBEAT_FILE)) is False

        # A silent probe stops the beats; the monitor reports a stall.
        local.register("wedged", max_staleness=0.01)
        time.sleep(0.1)  # let the probe age past its budget
        stalled = False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            time.sleep(0.1)
            if mon.observe(read_mtime_ns(hb.HEARTBEAT_FILE)):
                stalled = True
                break
        assert stalled
    finally:
        local.stop()
