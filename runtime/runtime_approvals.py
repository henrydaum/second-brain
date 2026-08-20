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

from __future__ import annotations


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
    enum_labels: list | None = None,
    default: Any = None,
    required: bool = True,
    pending_action: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
) -> StateMachineApprovalRequest:
    """Push an approval phase and emit an ``approval_requested`` event.

    The phase frame stores everything needed to rebuild the in-memory
    :class:`StateMachineApprovalRequest` after a restart (see
    ``restore_pending_requests`` in ``runtime_persistence``).

    ``enum_labels`` pairs with ``enum`` by index — the value answers, the label
    reads. Both are filtered together; see :func:`_sane_enum`.

    ``detail`` is the machine-readable half of the question — plain data a
    frontend can match on instead of parsing ``prompt`` (see
    ``sandbox.approval.detail_for``, its one producer). It rides the metadata
    and the frame so a rebuilt dialog keeps it, and crosses to sandboxed
    frontends via ``sandbox.frontends.project_approval``.
    """
    enum, enum_labels = _sane_enum(enum, enum_labels)
    session = get_or_create_session(runtime, session_key)
    with session.lock:
        req = StateMachineApprovalRequest(
            title=title, body=prompt, pending_action=pending_action,
            type=type, enum=enum, enum_labels=enum_labels, default=default,
        )
        req.metadata.update({"session_key": session_key, "conversation_id": session.conversation_id})
        if detail:
            req.metadata["detail"] = detail
        runtime._approval_requests[req.id] = req
        session.cs.push_phase(PhaseFrame(
            PHASE_APPROVING_REQUEST, "answer_approval", "user", title,
            {
                "request_id": req.id,
                "type": type,
                "enum": enum,
                "enum_labels": enum_labels,
                "default": default,
                "required": required,
                "title": title,
                "prompt": prompt,
                "pending": pending_action,
                "detail": detail,
                "previous_priority": session.cs.turn_priority,
            },
        ))
        session.cs.set_priority("user")
        if runtime.emit_event:
            runtime.emit_event("approval_requested", req)
        persist_marker(runtime, session)
        return req


def _sane_enum(enum: list | None,
               labels: list | None = None) -> tuple[list | None, list | None]:
    """Drop unanswerable choices from a caller-supplied enum.

    A choice whose string form is empty can't be rendered as a button or
    typed back, so a bad caller (an LLM tool call, typically) would wedge
    the session on a question with no valid answer. Filter those out; if
    nothing survives, treat the request as free-form input.

    **Labels are filtered in the same pass, and that is the whole reason this
    returns a pair.** ``FormStep.match_enum`` and ``form_display`` both pair
    values with labels *by index*, so dropping one value from the middle of a
    list and leaving the labels alone silently hands every later choice its
    neighbour's text — a dialog that reads correctly and answers wrongly.
    Mismatched lengths are treated as no labels at all rather than zipped
    short, since a partly-labelled list is the same bug wearing a smaller hat.
    """
    if not isinstance(enum, list):
        return None, None
    if not isinstance(labels, list) or len(labels) != len(enum):
        labels = None
    pairs = [(value, None if labels is None else labels[index])
             for index, value in enumerate(enum) if str(value).strip()]
    if not pairs:
        return None, None
    kept = [value for value, _ in pairs]
    kept_labels = None if labels is None else [str(label) for _, label in pairs]
    return kept, kept_labels


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


def resolve_answered_request(runtime, session_key: str, request_id: str | None,
                             result: ActionResult) -> None:
    """If the just-enacted action resolved an approval request, fulfill the
    in-memory request object so any blocked tool call can return, and announce
    that the question stopped waiting.

    **The announcement is the point of doing both here.** This runs after the
    action has been enacted and only when it succeeded, which is exactly when
    the phase frame is gone — so ``approval_settled`` means "there is nothing
    left to answer" rather than "somebody called resolve". Every way a question
    ends funnels through this one call: an answer, a cancel, and the approver's
    300s timeout, which denies by name through ``answer_request``.

    It fires for a *callable* approval too, where there is no live request
    object to fulfil (those are rebuilt from the phase frame rather than held
    in ``_approval_requests``). The id is what a frontend is holding, so the id
    is what it needs to hear about.
    """
    if not request_id or not result.ok:
        return
    req = runtime._approval_requests.pop(request_id, None)
    cancelled = result.action == "cancel"
    if req and not req.is_resolved:
        data = result.data or {}
        if cancelled:
            req.metadata["cancelled"] = True
            req.resolve(None)
        else:
            req.resolve(data.get("value", True))
    if runtime.emit_event:
        runtime.emit_event("approval_settled", {
            "session_key": session_key,
            "request_id": request_id,
            "reason": "cancelled" if cancelled else "answered",
        })
