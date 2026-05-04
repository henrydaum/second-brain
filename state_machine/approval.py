from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateMachineApprovalRequest:
    title: str
    body: str
    pending_action: dict[str, Any]
    id: str = field(default_factory=lambda: f"approve_{uuid.uuid4().hex}")
    approved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def is_resolved(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def resolve(self, approved: bool) -> None:
        if not self.is_resolved:
            self.approved = bool(approved)
            self._event.set()

    def to_event(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "body": self.body, "pending_action": self.pending_action, "metadata": self.metadata}
