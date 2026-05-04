import threading
import uuid
from typing import Any


class InteractiveRequest:
    """
    Small reusable request/response object for UI-mediated interactions.

    This is intentionally narrower than a general event payload. Use it only
    for flows where a producer needs a frontend or human to resolve a request.
    Fire-and-forget bus events should stay plain dict payloads.
    """

    kind: str = "interactive_request"

    def __init__(self, title: str = "", body: str = "", metadata: dict | None = None):
        self.id = uuid.uuid4().hex
        self.title = title
        self.body = body
        self.metadata = dict(metadata or {})
        self.response: Any = None
        self._event = threading.Event()

    @property
    def is_resolved(self) -> bool:
        return self._event.is_set()

    def resolve(self, response: Any = None):
        if self._event.is_set():
            return
        self.response = response
        self._event.set()
        self.on_resolved()

    def on_resolved(self):
        """Subclass hook for emitting follow-up events or cleanup signals."""
        pass

    def wait(self, timeout: float = 300.0) -> bool:
        return self._event.wait(timeout=timeout)
