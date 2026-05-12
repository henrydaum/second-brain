"""State-machine support for errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_INVALID_ACTION = "invalid_action"
ERROR_WRONG_TURN = "wrong_turn"
ERROR_WRONG_ACTOR_TYPE = "wrong_actor_type"
ERROR_UNKNOWN_COMMAND = "unknown_command"
ERROR_UNKNOWN_TOOL = "unknown_tool"
ERROR_MISSING_INPUT = "missing_input"
ERROR_INVALID_INPUT = "invalid_input"
ERROR_EXECUTION_FAILED = "execution_failed"
ERROR_ATTACHMENT_NOT_ALLOWED = "attachment_not_allowed"
ERROR_APPROVAL_REQUIRED = "approval_required"


@dataclass
class ActionError(Exception):
    """Action error."""
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    retry_phase: str | None = None

    def __str__(self) -> str:
        """Internal helper to handle str."""
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Handle to dict."""
        return {"code": self.code, "message": self.message, "details": self.details, "retry_phase": self.retry_phase}


@dataclass
class ActionResult:
    """Action result."""
    ok: bool
    action: str
    message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: ActionError | None = None

    @classmethod
    def success(cls, action: str, message: str | None = None, **data: Any) -> "ActionResult":
        """Handle success."""
        return cls(True, action, message, data)

    @classmethod
    def fail(cls, action: str, error: ActionError | str, code: str = ERROR_INVALID_ACTION, **details: Any) -> "ActionResult":
        """Handle fail."""
        err = error if isinstance(error, ActionError) else ActionError(code, error, details)
        return cls(False, action, err.message, error=err)

    def to_dict(self) -> dict[str, Any]:
        """Handle to dict."""
        return {
            "ok": self.ok,
            "action": self.action,
            "message": self.message,
            "data": self.data,
            "events": self.events,
            "error": self.error.to_dict() if self.error else None,
        }
