"""
Event bus — minimal pub/sub for distant subsystems.

Use when the producer and consumer are architecturally far apart and have no
natural reason to know about each other (e.g. a runtime session notifying
frontends that user input is needed). For anything on the hot path
(dispatch loop, DB writes, file watcher -> orchestrator), keep the direct
wiring.

Channel names live in event_channels.py. Don't sprinkle ad-hoc strings.

Usage:
    from events.event_bus import bus
    from events.event_channels import TASK_COMPLETED

    bus.subscribe(TASK_COMPLETED, lambda p: print(p["task_name"]))
    bus.emit(TASK_COMPLETED, {"task_name": "embed", "rows_written": 3})

    # Generic sync round-trip is still available:
    reply = bus.request("some.channel", {"question": "ping"}, timeout=5.0)
"""

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger("EventBus")


class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, channel: str, handler: Callable) -> Callable:
        """Register a handler. Returns an unsubscribe function."""
        with self._lock:
            self._subs.setdefault(channel, []).append(handler)

        def unsubscribe():
            with self._lock:
                if channel in self._subs and handler in self._subs[channel]:
                    self._subs[channel].remove(handler)
        return unsubscribe

    def has_subscribers(self, channel: str) -> bool:
        with self._lock:
            return bool(self._subs.get(channel))

    def emit(self, channel: str, payload: Any = None) -> None:
        """Fire-and-forget. Handlers run on caller's thread. Exceptions are logged, not raised."""
        with self._lock:
            handlers = list(self._subs.get(channel, []))
        for h in handlers:
            try:
                h(payload)
            except Exception as e:
                logger.warning(f"Handler on '{channel}' raised: {e}")

    def request(self, channel: str, payload: dict, timeout: float = 120.0) -> Any:
        """
        Synchronous round-trip. Emits the payload with a reply Event and result
        slot attached, then blocks until a subscriber calls reply.set() (after
        writing result[0]), or the timeout elapses.

        Returns result[0] on success, None on timeout or no subscribers.
        """
        if not self.has_subscribers(channel):
            return None
        reply = threading.Event()
        result = [None]
        enriched = {**payload, "reply": reply, "result": result}
        self.emit(channel, enriched)
        if not reply.wait(timeout=timeout):
            logger.warning(f"request('{channel}') timed out after {timeout}s")
            return None
        return result[0]


bus = EventBus()
