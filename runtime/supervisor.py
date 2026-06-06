"""
Plugin supervisor — one supervised call site for plugin code.

Every place the kernel runs plugin-authored code (``tool.run``, ``task.run`` /
``task.run_event``) flows through :func:`run_supervised`, the same way every
conversation action flows through one ``cs.enact`` site. The supervisor:

  * enforces a wall-clock ``timeout`` (a hung thread is *abandoned* — CPython
    can't kill it, but the worker slot is freed and the breaker bounds how many
    zombies can accumulate),
  * counts **strikes** — a raised exception or a timeout. A *returned* failure
    value (``ToolResult.failed`` / ``TaskResult.failed``) is **not** a strike;
    that is the plugin working correctly, and
  * trips a per-plugin circuit breaker that quarantines a misbehaving
    sandbox/installed plugin by emitting ``PLUGIN_QUARANTINE_REQUESTED``. The
    plugin watcher owns the actual unload (*mechanism*); the supervisor owns the
    *policy*. This mirrors the bus's role: producer and consumer kept apart.

Built-in kernel plugins are still supervised (timeout/abandon/logged) but are
passed ``eligible=False`` by their call sites, so the breaker can **never**
disable the kernel itself.

A single :class:`MemoryWatchdog` thread covers the one failure mode that
call-site wrapping cannot see — process RSS growth — by warning when the process
crosses a fraction of total system RAM. (Phase 2's isolated-service host is the
real fix for service memory leaks.)

**One knob.** The whole subsystem is automatic; the only setting is the
``plugin_supervisor`` on/off toggle (escape hatch for debugging). Everything
else is a sensible hardcoded constant below.

This module imports only stdlib + the event bus, so it never participates in a
plugin import cycle (quarantine is fired as a bus event, not a direct call).
"""

import concurrent.futures
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED, PLUGIN_QUARANTINE_REQUESTED

logger = logging.getLogger("Supervisor")

# --- Automatic tuning (no user knobs) ---
_STRIKE_THRESHOLD = 3       # crashes/timeouts within the window that trip the breaker
_STRIKE_WINDOW = 60.0       # seconds; a clean run resets the window
_MEMORY_POLL_INTERVAL = 30.0  # seconds between RSS samples
_MEMORY_LIMIT_FRACTION = 0.75  # warn when process RSS exceeds this fraction of total RAM


@dataclass
class SupervisedResult:
    """Outcome of a supervised call. ``ok`` is False only on crash/timeout, not
    on a returned failure value."""
    ok: bool
    value: Any = None
    error: str = ""
    timed_out: bool = False


class PluginHealth:
    """Per-plugin sliding-window circuit breaker.

    Keyed by the plugin's resolved source path. ``_STRIKE_THRESHOLD`` strikes
    within ``_STRIKE_WINDOW`` seconds trip the breaker for an *eligible* plugin,
    emitting one quarantine request. A clean run clears the window; every
    (re)load clears the plugin's history via :meth:`clear`.
    """

    def __init__(self, supervisor: "_Supervisor"):
        """Initialize the plugin health registry."""
        self._sup = supervisor
        self._strikes: dict[str, deque] = defaultdict(deque)
        self._quarantined: set[str] = set()
        self._lock = threading.Lock()

    def record_success(self, key: str) -> None:
        """A clean run resets the strike window for this plugin."""
        if not key:
            return
        with self._lock:
            self._strikes.pop(key, None)

    def record_strike(self, key: str, *, kind: str, name: str,
                      eligible: bool, reason: str) -> None:
        """Record a crash/timeout; trip + request quarantine if over threshold."""
        if not self._sup.enabled or not key:
            return
        now = time.time()
        tripped = False
        with self._lock:
            if key in self._quarantined:
                return
            dq = self._strikes[key]
            dq.append(now)
            while dq and now - dq[0] > _STRIKE_WINDOW:
                dq.popleft()
            logger.warning(f"Plugin strike {len(dq)}/{_STRIKE_THRESHOLD} for {kind} '{name}': {reason}")
            if eligible and len(dq) >= _STRIKE_THRESHOLD:
                self._quarantined.add(key)
                self._strikes.pop(key, None)
                tripped = True
        if tripped:
            logger.error(f"Circuit breaker tripped for {kind} '{name}' — requesting quarantine ({reason})")
            bus.emit(PLUGIN_QUARANTINE_REQUESTED, {
                "plugin_type": kind,
                "source_path": key,
                "name": name,
                "reason": reason,
            })

    def clear(self, key: str) -> None:
        """Forget a plugin's history — called whenever it is (re)loaded so a
        fixed-and-resaved plugin gets a fresh strike budget."""
        if not key:
            return
        with self._lock:
            self._strikes.pop(key, None)
            self._quarantined.discard(key)


def run_supervised(fn: Callable[[], Any], *, timeout: float, plugin_key: str,
                   kind: str, name: str = "", eligible: bool = True) -> SupervisedResult:
    """Run ``fn`` under a wall-clock timeout, feeding the circuit breaker.

    The timeout/abandon protection always applies; only strike-counting and
    quarantine are gated by the ``plugin_supervisor`` toggle (inside
    ``record_strike``).
    """
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"sb-{kind}-{name or 'plugin'}")
    try:
        future = executor.submit(fn)
        try:
            value = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            supervisor.health.record_strike(
                plugin_key, kind=kind, name=name, eligible=eligible,
                reason=f"timed out after {timeout}s")
            return SupervisedResult(
                ok=False, timed_out=True,
                error=f"{kind} '{name}' timed out after {timeout}s and was abandoned.")
        except Exception as e:
            supervisor.health.record_strike(
                plugin_key, kind=kind, name=name, eligible=eligible, reason=str(e))
            return SupervisedResult(ok=False, error=str(e))
        supervisor.health.record_success(plugin_key)
        return SupervisedResult(ok=True, value=value)
    finally:
        # Don't wait — a timed-out worker may never finish.
        executor.shutdown(wait=False)


class MemoryWatchdog:
    """Single daemon thread that warns when process RSS gets dangerously large.

    The limit is auto-derived from total system RAM (no knob). Per-process,
    in-shared-memory attribution is impossible, so this only warns — the
    isolated-service host (Phase 2) is the real fix.
    """

    def __init__(self, supervisor: "_Supervisor"):
        """Initialize the memory watchdog."""
        self._sup = supervisor
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._proc = None
        self._limit_mb = 0.0
        self._active = False

    def start(self) -> None:
        """Start the watchdog thread, unless disabled or psutil is absent."""
        if not self._sup.enabled:
            logger.info("Plugin supervisor disabled — memory watchdog not started.")
            return
        try:
            import psutil
            self._proc = psutil.Process()
            self._limit_mb = (psutil.virtual_memory().total / (1024 * 1024)) * _MEMORY_LIMIT_FRACTION
        except Exception:
            logger.warning("psutil not installed — memory watchdog inactive.")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="memory-watchdog")
        self._thread.start()
        logger.info(f"Memory watchdog started (warns above {self._limit_mb:.0f} MB).")

    def stop(self) -> None:
        """Signal the watchdog thread to exit."""
        self._stop.set()

    def _loop(self) -> None:
        """Internal helper: sample RSS until stopped."""
        while not self._stop.wait(_MEMORY_POLL_INTERVAL):
            if not self._sup.enabled:
                continue
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
            except Exception:
                continue
            if rss_mb >= self._limit_mb:
                self._on_breach(rss_mb)
            else:
                self._active = False

    def _on_breach(self, rss_mb: float) -> None:
        """Internal helper: warn once per breach episode."""
        if self._active:
            return
        self._active = True
        logger.error(f"Memory high: {rss_mb:.0f} MB (limit {self._limit_mb:.0f} MB)")
        bus.emit(CHAT_MESSAGE_PUSHED, {
            "message": f"⚠️ Memory high: {rss_mb:.0f} MB. A plugin may be leaking — consider /restart.",
            "kind": "alert", "source": "memory_watchdog",
        })


class _Supervisor:
    """Process-wide supervisor singleton (consistent with the ``bus`` singleton).

    Holds a live reference to the config dict so the single ``plugin_supervisor``
    toggle takes effect without a restart.
    """

    def __init__(self):
        """Initialize the supervisor."""
        self._config: dict = {}
        self.health = PluginHealth(self)
        self.memory = MemoryWatchdog(self)

    def configure(self, config: dict) -> None:
        """Bind the live config dict. Called once at bootstrap."""
        self._config = config or {}

    @property
    def enabled(self) -> bool:
        """The one knob: master on/off for quarantine + memory watchdog."""
        return bool(self._config.get("plugin_supervisor", True))

    def start_memory_watchdog(self) -> None:
        """Start the memory watchdog (no-op if disabled / psutil missing)."""
        self.memory.start()

    def stop_memory_watchdog(self) -> None:
        """Stop the memory watchdog."""
        self.memory.stop()


supervisor = _Supervisor()
