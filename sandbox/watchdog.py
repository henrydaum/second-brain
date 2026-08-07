"""One thread that notices boxes which have stopped responding.

Two problems, one shape, so one mechanism.

**A deadline was being asked the wrong question.** Every call into a resident
box armed a ``threading.Timer`` for ``call_timeout`` and killed the box when it
fired. But a box that has made a Request is *not* stuck — it is waiting for the
kernel to finish something the kernel itself started, and the kernel is free to
take as long as that thing takes. An escort placing a model call, a service
calling ``sdk.ui.ask``, anything reaching ``proc.run``: all of them sat inside
one ``box.call`` while the clock ran, and all of them died at thirty seconds
having done nothing wrong. The escort mechanism was unusable against a real
model for exactly this reason.

So the deadline measures **guest execution**, not wall clock:
``Execution.running_for`` is elapsed time minus the time the kernel spent
owing this execution an answer. Nothing has to be special-cased per Request
type, and the deadline keeps meaning what it was written to mean: *this code
is stuck*.

The subtraction is of *blocked time*, deliberately, and not of "time since the
guest last did something". The second reading looks equivalent and is not: a
runaway spinning on ``while True: sdk.fs.list(".")`` is never idle for more
than a millisecond, so it would never be overdue and could never be stopped —
which is the exact case a deadline exists for. Subtracting blocked time takes
one ninety-second wait off in full and a thousand short ones off to nearly
nothing, which is the distinction that was wanted.

**And a Timer per call is a thread per call.** A frontend polls every 50 ms, so
the old arrangement created and destroyed twenty timer threads a second for as
long as the app ran. One shared watchdog does the same job for every box at a
fixed cost.

The honest limit is granularity: a box is noticed within :data:`TICK` of going
overdue rather than at the instant. For "something is wedged, end it" that is
the right trade.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("Sandbox")

#: How often to look. Coarse on purpose — this decides when to give up on
#: something already stuck, and a tighter loop would only burn cycles.
TICK = 0.5

#: Wall clock a single call may occupy however it spends it. The discount for
#: blocked time is what makes a long escorted model call legal, but taken
#: alone it leaves one runaway immortal: ``while True: sdk.net.http(...)`` is
#: blocked almost continuously, so its *running* time barely advances and no
#: deadline it was given would ever be reached. Ten minutes is far above any
#: legitimate single call — the longest by design is a box start, which is
#: bounded by the same number — and far below forever.
HARD_CEILING = 600.0


def overdue(execution, started: float, deadline: float,
            now: float | None = None) -> bool:
    """Whether this execution has had its time, by either measure.

    Two questions, because one alone is answerable in the wrong direction.
    *Running* time catches code that is stuck or spinning while discounting
    honest waits; wall clock catches the runaway that hides inside those
    waits. Shared so the in-process loop and the watchdog thread cannot drift.
    """
    now = time.monotonic() if now is None else now
    if now - started > HARD_CEILING:
        return True
    return execution.running_for(started, now) > deadline


def _record(execution, started: float, deadline: float, ticket) -> None:
    """Note the deadline on the execution, tolerating anything that lacks one.

    Test doubles stand in for an ``Execution`` all over the suite and only ever
    needed ``running_for``. Watching one must not start failing because the
    watchdog gained a second thing to say to it.
    """
    note = getattr(execution, "watching", None)
    if note is None:
        return
    try:
        note(started, deadline, ticket)
    except Exception:
        logger.debug("could not record a deadline on %r", execution)


def _forget(execution, ticket) -> None:
    """The counterpart of :func:`_record`, with the same tolerance."""
    drop = getattr(execution, "unwatched", None)
    if drop is None:
        return
    try:
        drop(ticket)
    except Exception:
        logger.debug("could not clear a deadline on %r", execution)


class Watchdog:
    """A registry of deadlines and one thread that enforces them."""

    def __init__(self, tick: float = TICK):
        self._tick = tick
        self._lock = threading.Lock()
        self._watched: dict = {}
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._next = 0

    def watch(self, execution, deadline: float, on_overdue) -> int:
        """Watch one execution; returns the ticket needed to stop watching.

        A ticket rather than the execution itself, so two overlapping calls on
        one box — a stop racing a call, say — cannot cancel each other.
        """
        started = time.monotonic()
        with self._lock:
            self._next += 1
            ticket = self._next
            self._watched[ticket] = (execution, started, float(deadline),
                                     on_overdue)
        # Tell the execution what it is being held to, so ``sdk.budget`` can
        # answer from the same two numbers this loop compares against. Done
        # here rather than at each of the three call sites for the reason
        # ``overdue`` is shared: two copies of one deadline drift, and the
        # drift is invisible until something dies early.
        _record(execution, started, float(deadline), ticket)
        self._ensure_running()
        return ticket

    def release(self, ticket) -> None:
        """Stop watching. Safe to call twice, and with None."""
        if ticket is None:
            return
        with self._lock:
            entry = self._watched.pop(ticket, None)
        if entry is not None:
            _forget(entry[0], ticket)

    def _ensure_running(self) -> None:
        """Start the thread on first use; never start a second."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="sandbox-watchdog")
            self._thread.start()

    def _loop(self) -> None:
        """Scan for overdue executions until told to stop."""
        while not self._stopping.wait(self._tick):
            for ticket, entry in self._snapshot():
                execution, started, deadline, on_overdue = entry
                try:
                    if not overdue(execution, started, deadline):
                        continue
                except Exception:
                    continue
                self.release(ticket)
                try:
                    on_overdue()
                except Exception:
                    logger.exception("watchdog callback raised")

    def _snapshot(self) -> list:
        """The watch list as it stands, so the scan holds no lock."""
        with self._lock:
            return list(self._watched.items())

    def stop(self) -> None:
        """Stop the thread. For teardown and tests."""
        self._stopping.set()
        with self._lock:
            self._watched.clear()
            self._thread = None


#: One per process, like the console — there is one set of boxes to watch.
WATCHDOG = Watchdog()
