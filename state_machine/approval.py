"""State-machine support for approval."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StateMachineApprovalRequest:
    """A user-facing request for typed input.

    This carries any typed value, not just a boolean approval. `type` controls
    how the answering action coerces the user's response (boolean, string, integer,
    number, array, object). `value` holds the resolved answer.

    For boolean ("approve / deny") flows, `approved` is a convenience
    accessor that mirrors `value`.
    """

    title: str
    body: str
    pending_action: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: f"approve_{uuid.uuid4().hex}")
    type: str = "boolean"
    enum: list[Any] | None = None
    default: Any = None
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def approved(self) -> bool:
        """Handle approved."""
        return bool(self.value)

    @approved.setter
    def approved(self, val: bool) -> None:
        """Handle approved."""
        self.value = bool(val)

    @property
    def is_resolved(self) -> bool:
        """Return whether resolved."""
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Handle wait."""
        return self._event.wait(timeout)

    def resolve(self, value: Any) -> None:
        """Resolve state machine approval request."""
        if not self.is_resolved:
            self.value = value
            self._event.set()

    def to_event(self) -> dict[str, Any]:
        """Handle to event."""
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "type": self.type,
            "enum": self.enum,
            "default": self.default,
            "pending_action": self.pending_action,
            "metadata": self.metadata,
        }
