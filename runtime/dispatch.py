"""Per-action dispatch helpers.

The runtime's ``_dispatch`` is the single labeled enact() site for the
user-side state machine. To keep the dispatch loop readable, the small
per-action concerns (selecting the right ``content`` shape, mirroring
state-machine results into the ``RuntimeResult``, emitting
phase/turn-changed events, decorating forms for frontend rendering) are
factored out here as plain functions.

Each helper does one thing and is named after the question it answers, so
the dispatch loop reads top-to-bottom like a checklist.
"""

from __future__ import annotations


from typing import Any

from events.event_bus import bus
from events.event_channels import (
    SESSION_MESSAGE,
    SESSION_PHASE_CHANGED,
    SESSION_TURN_CHANGED,
)
from state_machine.errors import ActionResult
from state_machine.form_display import form_step_display
from state_machine.serialization import save_history_message
from runtime.session import RuntimeResult, RuntimeSession


# ──────────────────────────────────────────────────────────────────────
# Payload shape helpers
# ──────────────────────────────────────────────────────────────────────

def text_of(payload: dict | str | None) -> str:
    """Extract plain text from an inbound frontend payload."""
    return payload if isinstance(payload, str) else str((payload or {}).get("text") or "")


def attachments_of(payload: dict | str | None):
    """Pull a list of attachment dicts/dataclasses out of an inbound payload.

    Used by ``iterate_agent_turn`` so callers can pass attachments through
    to the agent without first emitting a SendAttachment action.
    """
    if not isinstance(payload, dict):
        return []
    return list(payload.get("attachments") or [])


def actor_id_of(payload: dict | str | None) -> str | None:
    """Extract an explicit actor ID from an inbound payload, when present."""
    return (payload or {}).get("actor_id") if isinstance(payload, dict) else None


def content_for_action(action_type: str, text: str, payload: Any) -> Any:
    """Pick the right content shape per action type.

    SendText accepts a plain string. Form/approval/attachment actions
    accept the original payload so the action can read structured fields.
    """
    if action_type == "send_text":
        return text
    if action_type == "submit_form_text":
        return text
    return payload


# ──────────────────────────────────────────────────────────────────────
# After enact: read parsed-attachment results back out
# ──────────────────────────────────────────────────────────────────────

def text_after_action(action_type: str, text: str, result: ActionResult) -> str:
    """Recover the user-visible text that should be mirrored into history after an action."""
    if action_type != "send_attachment" or not result.ok:
        return text
    parsed = (result.data or {}).get("parsed")
    return str((parsed or {}).get("text") or text) if isinstance(parsed, dict) else text


# ──────────────────────────────────────────────────────────────────────
# After enact: side-effects to surface
# ──────────────────────────────────────────────────────────────────────

def emit_state_change(session: RuntimeSession, old_phase: str, old_priority: str) -> None:
    """Broadcast phase and turn-priority changes caused by one user action."""
    if session.cs.phase != old_phase:
        bus.emit(SESSION_PHASE_CHANGED, {
            "session_key": session.key,
            "old_phase": old_phase,
            "new_phase": session.cs.phase,
        })
    if session.cs.turn_priority != old_priority:
        bus.emit(SESSION_TURN_CHANGED, {
            "session_key": session.key,
            "from_actor": old_priority,
            "to_actor": session.cs.turn_priority,
        })


def absorb_user_action(
    runtime,
    session: RuntimeSession,
    action_type: str,
    text: str,
    result: ActionResult,
) -> None:
    """Translate user-side action outcomes into history rows + side effects.

    Mirrors ``ConversationLoop._absorb`` but for actions originating from
    the frontend. Only ``send_text`` / ``send_attachment`` (with text) add
    a chat-transcript row; commands/forms/approvals have no provider-
    history impact, only state-machine impact.
    """
    if not result.ok:
        return
    if action_type in {"send_text", "send_attachment"} and text:
        msg = {"role": "user", "content": text}
        session.history.append(msg)
        if runtime.db and session.conversation_id:
            save_history_message(runtime.db, session.conversation_id, msg)
        bus.emit(SESSION_MESSAGE, {
            "session_key": session.key,
            "role": "user",
            "content": text,
            "actor_id": "user",
        })


def echo_callable_result(action_type: str, result: ActionResult, out: RuntimeResult) -> None:
    """Surface command/tool return values to the frontend (v1 behavior)."""
    if action_type not in {"call_command", "call_tool"} and getattr(result, "action", None) not in {"call_command", "call_tool"}:
        return
    if not result.ok:
        return
    value = (result.data or {}).get("result")
    if value is not None:
        out.messages.append(str(value))


def decorate_form(session: RuntimeSession, out: RuntimeResult) -> None:
    """If the session is now sitting on a form step, attach the form
    descriptor to ``out`` so the frontend can render the next field."""
    frame = session.cs.frame
    if frame and frame.step:
        display = form_step_display(frame.step)
        display["allow_back"] = bool((frame.data or {}).get("form_history"))
        out.form = {
            "name": frame.name,
            "action_type": frame.action_type,
            "field": frame.step.to_dict(),
            "collected": frame.data.get("args", {}),
            "display": display,
        }


def latest_user_text(session: RuntimeSession) -> str:
    """Return the latest user-authored text stored in session history."""
    for msg in reversed(session.history):
        if msg.get("role") == "user":
            return msg.get("content") or ""
    return ""
