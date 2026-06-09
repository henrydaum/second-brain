"""
Stall watchdog — a portable systemd ``WatchdogSec``.

Child side: a daemon beat thread touches ``DATA_DIR/heartbeat`` every
``_BEAT_INTERVAL`` seconds, but ONLY while every registered probe is fresh.
The loops that must prove liveness (the orchestrator dispatch loop, the main
idle loop) register a :class:`Probe` and call ``beat()`` each iteration. If a
probe goes silent past its budget the beat thread withholds the heartbeat and
logs a warning naming the stale probe — so ``app.log`` records *which* loop
wedged before the launcher kills the process.

Launcher side (``main.pyw`` ``_supervise``): :class:`_StallMonitor` watches the
heartbeat file's ``st_mtime_ns`` as an **opaque version number** — progress
means "the value changed"; staleness accrues on the launcher's own
``time.monotonic()``. No wall-clock comparison, so NTP steps, manual clock
changes, and DST are all inert. No change for ``stall_timeout`` seconds after
the first post-spawn beat ⇒ kill the child process tree and restart.

False-positive bias (a spurious kill is worse than a missed stall):
  * no enforcement before the first post-spawn beat — arbitrarily slow boots
    are safe; a boot hang is deliberately not covered,
  * clean shutdown calls ``unregister_all()`` first — zero probes ⇒ the beat
    thread keeps the file fresh while models unload; a genuinely hung shutdown
    is deliberately not covered,
  * a wall-clock gap between polls (suspend/hibernate/clock jump) grants a
    full ``stall_timeout`` of grace after wake,
  * budgets are additive: a probe must be silent past its ``max_staleness``
    *and then* the file silent past ``stall_timeout`` — a 1s-tick loop must be
    wedged ~3 minutes end-to-end before a kill. Native code hogging the GIL
    that long has frozen every loop anyway, so killing is correct.

Known accepted gap: two app instances sharing DATA_DIR write the same file, so
a launcher may miss a stall in its own child (missed-stall direction only).

The beat thread always runs and always beats when healthy — the
``stall_timeout`` knob gates only launcher enforcement, so a config mismatch
between the two processes can never kill a healthy child.

Stdlib + ``paths`` only, so the pre-heavy-imports launcher branch of
``main.pyw`` can import this module while staying a few-MB watchdog.
"""

import logging
import os
import threading
import time

from paths import DATA_DIR

logger = logging.getLogger("Heartbeat")

HEARTBEAT_FILE = DATA_DIR / "heartbeat"

# --- Automatic tuning (no user knobs beyond stall_timeout) ---
_BEAT_INTERVAL = 5.0        # seconds between heartbeat file touches
_DEFAULT_STALENESS = 60.0   # probe budget; probed loops tick at ~1s => 60x headroom
_STALE_RELOG_EVERY = 60.0   # re-warn cadence while a probe stays stale
SLEEP_GAP = 30.0            # wall-clock poll gap that implies suspend/resume
MIN_STALL_TIMEOUT = 30.0    # floor for nonzero stall_timeout (6 beat intervals)
DEFAULT_STALL_TIMEOUT = 120.0


# ── Child side: probes + beat thread ─────────────────────────────────

class Probe:
    """Liveness probe for one long-running loop. ``beat()`` each iteration."""

    def __init__(self, name: str, max_staleness: float, registry: "_Heartbeat"):
        """Initialize the probe (fresh as of now)."""
        self.name = name
        self.max_staleness = max_staleness
        self._registry = registry
        self._last = time.monotonic()

    def beat(self) -> None:
        """Mark this probe fresh."""
        self._last = time.monotonic()

    def unregister(self) -> None:
        """Remove this probe from the registry (loop is stopping cleanly)."""
        self._registry.unregister(self.name)

    def age(self, now: float) -> float:
        """Seconds since the last beat."""
        return now - self._last


class _Heartbeat:
    """Process-wide heartbeat singleton (consistent with ``supervisor``/``bus``)."""

    def __init__(self):
        """Initialize the heartbeat registry."""
        self._probes: dict[str, Probe] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._stale_warned: dict[str, float] = {}  # probe name -> last warn time

    def register(self, name: str, max_staleness: float = _DEFAULT_STALENESS) -> Probe:
        """Register (or replace — idempotent across loop restarts) a probe."""
        probe = Probe(name, max_staleness, self)
        with self._lock:
            self._probes[name] = probe
        return probe

    def unregister(self, name: str) -> None:
        """Remove a probe; a missing name is a no-op."""
        with self._lock:
            self._probes.pop(name, None)
            self._stale_warned.pop(name, None)

    def unregister_all(self) -> None:
        """Release every probe — called first thing in clean shutdown so the
        beat thread keeps the file fresh (zero probes ⇒ beat) while models
        unload. ``/quit`` must never be stall-killed."""
        with self._lock:
            self._probes.clear()
            self._stale_warned.clear()

    def start(self) -> None:
        """Start the beat thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")
        self._thread.start()
        logger.info(f"Heartbeat started ({HEARTBEAT_FILE}, every {_BEAT_INTERVAL:.0f}s).")

    def stop(self) -> None:
        """Signal the beat thread to exit (used by tests)."""
        self._stop_evt.set()

    def _stale_probes(self, now: float) -> list[Probe]:
        """Snapshot of probes whose age exceeds their budget."""
        with self._lock:
            return [p for p in self._probes.values() if p.age(now) > p.max_staleness]

    def _tick(self) -> bool:
        """One beat-loop body. Returns whether the heartbeat was written."""
        now = time.monotonic()
        stale = self._stale_probes(now)
        if not stale:
            recovered = list(self._stale_warned)
            self._stale_warned.clear()
            for name in recovered:
                logger.info(f"Probe '{name}' recovered — resuming heartbeat.")
            self._write_beat()
            return True
        for probe in stale:
            last = self._stale_warned.get(probe.name)
            if last is None or now - last >= _STALE_RELOG_EVERY:
                self._stale_warned[probe.name] = now
                logger.warning(
                    f"Probe '{probe.name}' silent for {probe.age(now):.0f}s — "
                    f"withholding heartbeat; the launcher may restart us.")
        return False

    def _write_beat(self) -> None:
        """Rewrite the heartbeat file (recreates it if the user deleted it).
        Content is for humans only — the launcher reads mtime, never content."""
        try:
            with open(HEARTBEAT_FILE, "w") as f:
                f.write(f"{os.getpid()} {time.time():.0f}\n")
        except Exception as e:
            logger.warning(f"Could not write heartbeat file: {e}")

    def _loop(self) -> None:
        """Internal helper: beat until stopped."""
        while not self._stop_evt.wait(_BEAT_INTERVAL):
            self._tick()


heartbeat = _Heartbeat()


# ── Launcher side: stall decision + helpers ──────────────────────────

class _StallMonitor:
    """Pure stall decision for one child generation; clocks injected for tests.

    The heartbeat file's ``st_mtime_ns`` is an opaque version number: progress
    is "the value changed since last observed". Staleness accrues on
    ``monotonic``. Enforcement only starts after the first post-spawn change
    (``baseline_mtime_ns`` is the pre-spawn value, or None if the file was
    missing), so a previous run's leftover file never arms the monitor.
    """

    def __init__(self, stall_timeout: float, baseline_mtime_ns: int | None,
                 monotonic=time.monotonic, wall=time.time):
        """Initialize the monitor for a freshly spawned child."""
        self.stall_timeout = stall_timeout
        self._monotonic = monotonic
        self._wall = wall
        self._last_mtime = baseline_mtime_ns
        self._seen_first_beat = False
        now = self._monotonic()
        self._last_progress = now
        self._last_wall_poll = self._wall()
        self._grace_until = 0.0

    def observe(self, mtime_ns: int | None) -> bool:
        """One poll. Returns True when the child should be considered stalled."""
        now = self._monotonic()
        w = self._wall()
        if w - self._last_wall_poll > SLEEP_GAP:
            # Suspend/hibernate/forward clock jump: on Windows, monotonic kept
            # counting through the sleep, so grant a full window after wake.
            self._grace_until = now + self.stall_timeout
        self._last_wall_poll = w

        if mtime_ns is not None and mtime_ns != self._last_mtime:
            self._last_mtime = mtime_ns
            self._seen_first_beat = True
            self._last_progress = now
            return False
        if not self._seen_first_beat:
            return False  # still booting — never enforced
        if now < self._grace_until:
            return False
        return now - self._last_progress > self.stall_timeout


def read_mtime_ns(path) -> int | None:
    """The file's st_mtime_ns, or None if it can't be statted."""
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def clamp_stall_timeout(raw) -> float:
    """Normalize a configured stall_timeout: 0 means off; nonzero values are
    floored to MIN_STALL_TIMEOUT so a hand-edited tiny value can never
    out-race the beat interval; garbage means the default."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_STALL_TIMEOUT
    if value <= 0:
        return 0.0
    return max(value, MIN_STALL_TIMEOUT)


def kill_tree(proc) -> None:
    """Kill a child process and its descendants (pip/git may be running).

    Uses psutil (a kernel dependency) for the tree walk, with a plain
    ``proc.kill()`` fallback. Never taskkill / process groups — those would
    break the shared-console Ctrl+C behavior the launcher relies on.
    """
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for p in [parent, *children]:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception:
        pass
    finally:
        try:
            proc.kill()  # belt-and-braces; harmless if already dead
        except Exception:
            pass
