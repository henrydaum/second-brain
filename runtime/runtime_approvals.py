from __future__ import annotations

"""Programmatic approval / typed-input requests.

A *request* is the runtime's way of pausing a conversation to ask the user
for a value (boolean approval, an enum pick, a free-form string). It comes
in two flavours:

- **Tool-initiated**: a tool calls ``runtime.request_input(...)`` from
  inside its own thread and the call blocks until the user answers.
- **Action-initiated**: a callable with ``require_approval=True`` pushes
  an approval phase frame from inside ``_CallableAction._approval``.

Both paths land on the same phase frame shape (``PHASE_APPROVING_REQUEST``)
and the same ``answer_approval`` action resolves them, so this module's
job is just to set up the frame, register the in-memory request object,
and reconcile when the answer comes back.

The ``form`` rendering of the frame (the dict the frontend renders) lives
in ``runtime_dispatch``.
"""

from typing import Any

from state_machine.approval import StateMachineApprovalRequest
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST
from state_machine.conversation import PhaseFrame
from state_machine.errors import ActionResult
from runtime.persistence import get_or_create_session, persist_marker
from runtime.session import RuntimeSession


def request_input(
    runtime,
    session_key: str,
    title: str,
    prompt: str,
    *,
    type: str = "boolean",
    enum: list | None = None,
    default: Any = None,
    required: bool = True,
    pending_action: dict[str, Any] | None = None,
) -> StateMachineApprovalRequest:
    """Push an approval phase and emit an ``approval_requested`` event.

    The phase frame stores everything needed to rebuild the in-memory
    :class:`StateMachineApprovalRequest` after a restart (see
    ``restore_pending_requests`` in ``runtime_persistence``).
    """
    session = get_or_create_session(runtime, session_key)
    with session.lock:
        req = StateMachineApprovalRequest(
            title=title, body=prompt, pending_action=pending_action,
            type=type, enum=enum, default=default,
        )
        req.metadata.update({"session_key": session_key, "conversation_id": session.conversation_id})
        runtime._approval_requests[req.id] = req
        session.cs.push_phase(PhaseFrame(
            PHASE_APPROVING_REQUEST, "answer_approval", "user", title,
            {
                "request_id": req.id,
                "type": type,
                "enum": enum,
                "default": default,
                "required": required,
                "title": title,
                "prompt": prompt,
                "pending": pending_action,
                "previous_priority": session.cs.turn_priority,
            },
        ))
        session.cs.set_priority("user")
        if runtime.emit_event:
            runtime.emit_event("approval_requested", req)
        persist_marker(runtime, session)
        return req


def request_approval(
    runtime,
    session_key: str,
    title: str,
    body: str,
    pending_action: dict[str, Any],
) -> StateMachineApprovalRequest:
    """Boolean-approval gate. Thin wrapper around ``request_input``."""
    return request_input(runtime, session_key, title, body, type="boolean", pending_action=pending_action)


def answer_request(runtime, session_key: str, request_id: str, value):
    """Resolve a pending request by submitting an ``answer_approval`` action."""
    return runtime.handle_action(session_key, "answer_approval", {"value": value, "request_id": request_id})


# ──────────────────────────────────────────────────────────────────────
# Used by the dispatcher to thread request_id through enact()
# ──────────────────────────────────────────────────────────────────────

def current_request_id(session: RuntimeSession, action_type: str) -> str | None:
    """Return the request_id of the top approval frame, if the action that
    is about to be enacted could resolve it."""
    frame = session.cs.frame
    if action_type not in {"answer_approval", "send_text", "cancel"} or not frame or frame.phase != PHASE_APPROVING_REQUEST:
        return None
    return (getattr(frame, "data", {}) or {}).get("request_id")


def resolve_answered_request(runtime, request_id: str | None, result: ActionResult) -> None:
    """If the just-enacted action resolved an approval request, fulfill the
    in-memory request object so any blocked tool call can return."""
    if not request_id or not result.ok:
        return
    req = runtime._approval_requests.pop(request_id, None)
    if req and not req.is_resolved:
        data = result.data or {}
        if result.action == "cancel":
            req.metadata["cancelled"] = True
            req.resolve(None)
            return
        req.resolve(data.get("value", True))
