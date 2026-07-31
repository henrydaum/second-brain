"""Re-entrancy into a resident box — the shape that froze the whole app.

A resident box serializes under one lock, and ``poll`` holds that lock for its
entire duration. Everything here is one question asked four ways: *can a call
into a box ever wait forever?*

It could, and the bill was steep. The timekeeper published ``subagent.spawn``
from its poll tick; ``EventBus.emit`` runs handlers on the publisher's thread;
``SubagentRegistry`` answered by asking the timekeeper to pin the job's
conversation — straight back into the box whose guest thread was parked on that
very emit. The box wedged until the 600s hard ceiling killed it for good, and
every later ``cron.*`` call parked another of the sixteen sandbox workers on the
same dead lock until the process had none left and went silent.

``sandbox/handlers/kernel.py``'s ``_drive`` already existed for exactly this
shape on the inbound side. These tests pin the outbound side and the two
backstops, so the next path cannot inherit the bug by forgetting to detach.
"""

import threading
import time
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from events.event_bus import bus
from guest.loader import unload_box
from sandbox import Interpreter, Sandbox, events, run_in_process
from sandbox.boxes import open_box
from sandbox.bridge import adapt

COUNTER = Path(__file__).parent / "fixtures" / "service_counter.py"

# A service that publishes from inside a call, exactly as the timekeeper's tick
# does, and exposes something a subscriber can call back into.
EMITTER = '''
"""A migrated service that publishes while holding its own box."""

from guest.bases import BaseService


class Ticker(BaseService):
    """Fires an event, and answers to anyone who calls back."""

    name = "ticker"
    exports = ["fire", "ping", "pings"]
    ISOLATION

    def start(self, sdk):
        """Begin with nobody having called back."""
        self._pings = 0
        return True

    def fire(self, sdk, channel="ticker.fired"):
        """Publish while this box's single call lock is held."""
        sdk.events.emit(channel, {"from": "ticker"})
        return "fired"

    def ping(self, sdk):
        """What a subscriber re-enters this box to reach."""
        self._pings += 1
        return self._pings

    def pings(self, sdk):
        """How many times we were called back into."""
        return self._pings
'''


@pytest.fixture
def box():
    """A sandbox torn down even if a test fails."""
    made = Sandbox()
    yield made
    made.shutdown()


@pytest.fixture
def ticker(tmp_path, box, request):
    """A loaded emitting service, unloaded afterwards."""
    isolation = getattr(request, "param", "")
    source = EMITTER.replace(
        "ISOLATION", f'isolation = "{isolation}"' if isolation else "")
    path = tmp_path / "service_ticker.py"
    path.write_text(source, encoding="utf-8")

    made = adapt(path).build_services({})["ticker"]
    assert made.load() is True
    yield made
    made.unload()
    unload_box("service_ticker")


@pytest.fixture
def counter():
    """A bare resident box, for the tests that hold its lock directly."""
    interp = Interpreter()
    box = open_box(interp, COUNTER, "Counter", name="counter")
    yield box
    try:
        box.stop()
    except Exception:
        pass
    unload_box(box.name)
    interp.shutdown()


# ──────────────────────────────────────────────────────────────────────
# The deadlock itself.
# ──────────────────────────────────────────────────────────────────────

def test_a_subscriber_may_call_back_into_the_box_that_published(ticker):
    """The exact cycle that took the app down, asserted end to end.

    Before ``sandbox.events.publish``, ``fire`` never returned: the guest held
    the box lock waiting for its emit to be answered, and the answer required a
    subscriber that was blocking on that same lock. This test hung forever
    rather than failing, which is the worst way for a suite to report a bug.
    """
    seen = threading.Event()
    calls = []

    def subscriber(payload=None):
        """Re-enter the publisher's box, as SubagentRegistry did."""
        calls.append(ticker.ping())
        seen.set()

    drop = bus.subscribe("ticker.fired", subscriber)
    try:
        # The guest returns without waiting for delivery. That is the fix:
        # answering the Request is what frees the lock the subscriber needs.
        assert ticker.fire() == "fired"

        assert seen.wait(timeout=10.0), "the event was never delivered"
        assert calls == [1]
    finally:
        drop()


def test_publishing_does_not_wait_for_a_slow_subscriber(ticker):
    """A guest is answered immediately; delivery is the kernel's problem.

    Not merely a latency nicety. A publisher that waits is a publisher holding
    its lock, which is the whole precondition of the deadlock above.
    """
    release = threading.Event()
    entered = threading.Event()

    def slow(payload=None):
        """Sit on the delivery thread for a while."""
        entered.set()
        release.wait(timeout=10.0)

    drop = bus.subscribe("ticker.slow", slow)
    try:
        started = time.monotonic()
        ticker.fire(channel="ticker.slow")
        elapsed = time.monotonic() - started

        assert entered.wait(timeout=5.0), "delivery never started"
        # The subscriber is still sitting there, and the publisher is long done.
        assert elapsed < 2.0, f"publishing waited {elapsed:.1f}s on a subscriber"
    finally:
        release.set()
        drop()


def test_delivery_keeps_the_order_it_was_published_in(ticker):
    """One queue and one thread, so a burst cannot arrive shuffled.

    Thread-per-emit would pass every other test in this file and fail here,
    which is the reason to assert it rather than trust it.
    """
    heard = []
    done = threading.Event()

    def subscriber(payload=None):
        """Record arrivals in the order they land."""
        heard.append(payload.get("n"))
        if len(heard) == 20:
            done.set()

    drop = bus.subscribe("ticker.ordered", subscriber)
    try:
        for n in range(20):
            events.publish("ticker.ordered", {"n": n})
        assert done.wait(timeout=10.0), f"only {len(heard)} of 20 arrived"
        assert heard == list(range(20))
    finally:
        drop()


def test_drain_makes_published_and_delivered_the_same_statement():
    """Shutdown and tests both need a point where the queue is empty."""
    heard = []
    drop = bus.subscribe("ticker.drained", lambda payload=None: heard.append(1))
    try:
        for _ in range(5):
            events.publish("ticker.drained", {})
        assert events.drain(timeout=10.0)
        assert len(heard) == 5
    finally:
        drop()


# ──────────────────────────────────────────────────────────────────────
# The backstop: a wedged box must not eat a worker forever.
# ──────────────────────────────────────────────────────────────────────

def test_a_wedged_box_fails_its_caller_instead_of_parking_it(counter,
                                                             monkeypatch):
    """Sixteen workers, and a permanently blocked one is gone for good.

    The ceiling is deliberately the *holder's* ceiling: below it nothing is
    refused, because serializing is how a box is supposed to behave. Patched
    down here because the real value is ten minutes.
    """
    monkeypatch.setattr("sandbox.boxes.HARD_CEILING", 1.0)

    counter._lock.acquire()
    try:
        started = time.monotonic()
        result = counter.call("total")
        elapsed = time.monotonic() - started

        assert not result.ok
        assert result.code == "timeout"
        assert result.retryable
        assert elapsed < 10.0, "the caller was parked rather than failed"
    finally:
        counter._lock.release()


def test_an_ephemeral_run_is_not_charged_for_the_kernel_s_own_delay():
    """A command waiting on the kernel is not a command that has hung.

    The resident path has always measured *running* time, discounting what the
    kernel spent answering Requests, through a helper whose docstring says it
    is shared "so the in-process loop and the watchdog thread cannot drift".
    They had drifted: ``run_in_process`` was a bare wall-clock wait, so every
    ephemeral command — every slash command — died at its declared deadline
    however much of that time was the kernel's own.

    That is what turned a wedged timekeeper into an app-wide outage: ``cron.*``
    calls blocked on the dead box, and the command layer reported "timed out
    after 30.0s (declared None)" as though the plugin were at fault.
    """
    from sandbox.guest.requests import Result
    from sandbox.interpreter import HANDLERS

    def slow_handler(ctx, args):
        """A kernel handler that honestly takes a while."""
        time.sleep(1.5)
        return Result(data="eventually")

    interp = Interpreter()
    original = HANDLERS.get("paths.get")
    HANDLERS["paths.get"] = slow_handler
    try:
        def body(sdk):
            """Ask for one slow thing, then finish immediately."""
            return sdk.paths.get("data")

        # A one-second leash against a 1.5s wait that is entirely the kernel's.
        started = time.monotonic()
        result = run_in_process(interp, body, name="patient", timeout=1.0)
        elapsed = time.monotonic() - started

        assert result.ok, f"charged for the kernel's own delay: {result.error}"
        assert result.data == "eventually"
        assert elapsed >= 1.5
    finally:
        HANDLERS["paths.get"] = original
        interp.shutdown()


def test_an_ephemeral_run_that_spins_still_runs_out_of_time():
    """The discount must not make a runaway immortal.

    Subtracting *blocked* time is not the same as "time since the guest last
    did something", and this is the case that separates them: code spinning on
    Requests is never idle for long, so the second reading would never expire.
    """
    interp = Interpreter()
    try:
        def body(sdk):
            """Never stop asking for things."""
            while True:
                sdk.paths.get("data")

        started = time.monotonic()
        result = run_in_process(interp, body, name="runaway", timeout=1.0)
        elapsed = time.monotonic() - started

        assert not result.ok
        assert "timed out" in result.error
        assert elapsed < 30.0
    finally:
        interp.shutdown()


def test_a_box_that_dies_while_we_wait_fails_us_at_once(counter, monkeypatch):
    """Waiting out the full ceiling for an answer that is never coming is
    just a slower version of the same bug."""
    monkeypatch.setattr("sandbox.boxes.HARD_CEILING", 60.0)

    counter._lock.acquire()
    try:
        def die():
            """Kill the box out from under the waiting caller."""
            time.sleep(0.2)
            counter._alive = False

        threading.Thread(target=die, daemon=True).start()

        started = time.monotonic()
        result = counter.call("total")
        elapsed = time.monotonic() - started

        assert not result.ok
        assert "not running" in result.error
        assert elapsed < 10.0, "waited on a box already known to be dead"
    finally:
        counter._lock.release()
