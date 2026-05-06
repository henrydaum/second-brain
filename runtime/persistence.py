from __future__ import annotations

"""Conversation persistence and lifecycle helpers.

The runtime never reads the DB directly. Everything that creates a
conversation row, hydrates a session from past messages, writes a state
marker, or appends a chat message routes through this module.

The functions are arranged in lifecycle order:
1. ``open_session``: the unified create/load/rebind entry point.
2. ``load_conversation`` / ``load_history``: hydrate from DB.
3. ``reset_conversation`` / ``new_conversation``: start fresh.
4. ``inject_user_message`` / ``iterate_agent_turn``: drive a turn.
5. Marker helpers (``persist_marker``, etc.) used everywhere else.
"""

import logging
import uuid
from typing import Any

from events.event_bus import bus
from events.event_channels import (
    SESSION_CLOSED,
    SESSION_CREATED,
    SESSION_MESSAGE,
    SESSION_TURN_COMPLETED,
)
from state_machine.approval import StateMachineApprovalRequest
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST
from state_machine.serialization import latest_state, messages_to_history, save_history_message, save_state_marker
from runtime.runtime_config import new_state, refresh_specs
from runtime.session import RuntimeSession, SessionConflict

logger = logging.getLogger("Runtime.persistence")


# ──────────────────────────────────────────────────────────────────────
# Session lookup + the unified create/load entry point
# ──────────────────────────────────────────────────────────────────────

def get_or_create_session(runtime, key: str) -> RuntimeSession:
    """Return the existing session for ``key`` or create an empty one."""
    with runtime._sessions_lock:
        if key not in runtime.sessions:
            is_subagent = key.startswith("subagent:")
            session = RuntimeSession(key, new_state(runtime), is_subagent=is_subagent)
            session.cs = new_state(runtime, session=session)
            runtime.sessions[key] = session
            bus.emit(SESSION_CREATED, {
                "session_key": key,
                "is_subagent": is_subagent,
                "agent_profile": session.active_agent_profile,
            })
        return runtime.sessions[key]


def open_session(
    runtime,
    session_key: str,
    *,
    conversation_id: int | None = None,
    kind: str = "user",
    origin: str | None = None,
    title: str = "New conversation",
    agent_profile: str | None = None,
    system_prompt_extras: dict[str, Any] | None = None,
) -> RuntimeSession:
    """Single entry point for "address this conversation".

    - If a session exists for ``session_key`` and (a) ``conversation_id`` is
      None or matches the existing one, returns it.
    - If the session exists but its ``conversation_id`` differs from the
      requested one, raises :class:`SessionConflict`. Realistic case: two
      cron jobs sharing a name both trying to claim ``subagent:<name>``.
    - If no session exists and ``conversation_id`` is given, loads it.
    - If no session exists and ``conversation_id`` is None, creates a new
      conversation row first, then loads it.

    This is the API plugins, tasks, and tools should reach for instead of
    juggling ``create_conversation`` + ``load_conversation`` themselves.
    """
    with runtime._sessions_lock:
        existing = runtime.sessions.get(session_key)
        if existing is not None:
            if conversation_id is None or existing.conversation_id == conversation_id:
                return existing
            raise SessionConflict(session_key, existing.conversation_id, conversation_id)

    if conversation_id is None:
        if runtime.db is None:
            raise RuntimeError("Cannot create a conversation without a database.")
        conversation_id = runtime.db.create_conversation(title, kind=kind, origin=origin)

    return load_conversation(
        runtime, session_key, conversation_id,
        agent_profile=agent_profile,
        system_prompt_extras=system_prompt_extras,
    )


def create_conversation(
    runtime,
    title: str = "New conversation",
    *,
    kind: str = "user",
    origin: str | None = None,
) -> int | None:
    """Create a conversation row only — does not load it into a session.

    Use ``open_session`` instead unless you really want a detached row.
    """
    return runtime.db.create_conversation(title, kind=kind, origin=origin) if runtime.db else None


def load_conversation(
    runtime,
    session_key: str,
    conversation_id: int,
    *,
    agent_profile: str | None = None,
    system_prompt_extras: dict[str, Any] | None = None,
) -> RuntimeSession:
    """Hydrate a session from a stored conversation.

    Refuses to bind ``session_key`` if it is already pointing at a
    different ``conversation_id`` — see :class:`SessionConflict` for why.
    """
    existing = runtime.sessions.get(session_key)
    if existing is not None and existing.conversation_id not in (None, conversation_id):
        raise SessionConflict(session_key, existing.conversation_id, conversation_id)

    rows = runtime.db.get_conversation_messages(conversation_id) if runtime.db else []
    marker = latest_state(rows) or {}
    conv = runtime.db.get_conversation(conversation_id) if runtime.db else {}
    is_subagent = (conv or {}).get("kind") == "subagent"
    saved_profile = agent_profile or marker.get("profile_override") or marker.get("active_agent_profile")
    profile = saved_profile or runtime.config.get("active_agent_profile") or "default"
    session = RuntimeSession(
        session_key,
        new_state(runtime, marker),
        messages_to_history(rows),
        conversation_id,
        False,
        profile,
        profile_override=saved_profile,
        is_subagent=is_subagent,
        subagent_meta=dict(marker.get("subagent_meta") or {}),
        system_prompt_extras={**dict(marker.get("system_prompt_extras") or {}), **dict(system_prompt_extras or {})},
    )
    # Re-seed cs with session-aware specs.
    session.cs = new_state(runtime, marker, session=session)
    with runtime._sessions_lock:
        runtime.sessions[session_key] = session
    bus.emit(SESSION_CREATED, {
        "session_key": session_key,
        "is_subagent": is_subagent,
        "agent_profile": profile,
    })
    restore_pending_requests(runtime, session)
    return session


def load_history(runtime, session_key: str, conversation_id: int):
    """Switch a session into a previous conversation.

    Returns a :class:`RuntimeResult` with a short status line (and a
    preview of the most recent messages, so the user has context for
    where they left off).
    """
    from runtime.session import RuntimeResult

    old = runtime.sessions.get(session_key)
    old_profile = (old.profile_override or old.active_agent_profile) if old else runtime.config.get("active_agent_profile") or "default"
    if old:
        bus.emit(SESSION_CLOSED, {"session_key": session_key})
    session = load_conversation(runtime, session_key, conversation_id)
    new_profile = session.profile_override or session.active_agent_profile

    title = conversation_title(runtime, conversation_id)
    msg = f"Loaded conversation: {title}\nAgent: {new_profile}"
    if old_profile != new_profile:
        msg += f"\nSwitched agent: {old_profile} -> {new_profile}"
    preview = _format_history_preview(session.history)
    if preview:
        msg += f"\n\nWhere you left off:\n{preview}"

    return RuntimeResult(
        messages=[msg],
        data={"conversation_id": conversation_id, "history": session.history, "agent_profile": new_profile},
    )


def reset_conversation(runtime, session_key: str) -> RuntimeSession:
    is_subagent = session_key.startswith("subagent:")
    with runtime._sessions_lock:
        existed = session_key in runtime.sessions
        session = RuntimeSession(session_key, new_state(runtime), is_subagent=is_subagent)
        session.cs = new_state(runtime, session=session)
        runtime.sessions[session_key] = session
    if existed:
        bus.emit(SESSION_CLOSED, {"session_key": session_key})
    bus.emit(SESSION_CREATED, {
        "session_key": session_key,
        "is_subagent": is_subagent,
        "agent_profile": session.active_agent_profile,
    })
    return session


def new_conversation(runtime, session_key: str):
    from runtime.session import RuntimeResult

    reset_conversation(runtime, session_key)
    profile = runtime.config.get("active_agent_profile") or "default"
    return RuntimeResult(messages=[f"New conversation started. Agent: {profile}."])


# ──────────────────────────────────────────────────────────────────────
# Driving turns
# ──────────────────────────────────────────────────────────────────────

def iterate_agent_turn(
    runtime,
    session_key: str,
    prompt: str,
    *,
    image_paths: list[str] | None = None,
    actor_id: str = "user",
):
    """Drive one user prompt → agent reply round-trip.

    Used by tools (ask_subagent) and tasks (run_subagent) — anything that
    pushes a turn from outside a frontend. After the turn completes the
    full provider history is replaced atomically and a fresh state
    marker is saved.
    """
    payload = {"text": prompt, "actor_id": actor_id}
    if image_paths:
        payload["image_paths"] = image_paths
    out = runtime.handle_action(session_key, "send_text", payload)
    session = runtime.sessions.get(session_key)
    if out.ok and session and runtime.db and session.conversation_id:
        # Hold the session lock so the post-turn full-history write is
        # atomic with respect to any concurrent action targeting the
        # same session_key.
        with session.lock:
            runtime.db.replace_conversation_messages(session.conversation_id, list(session.history))
            persist_marker(runtime, session)
    final_text = "\n".join(m for m in out.messages if m).strip()
    event = {
        "session_key": session_key,
        "conversation_id": session.conversation_id if session else None,
        "final_text": final_text,
        "new_messages": list(out.data.get("new_messages") or []),
        "attachments": list(out.attachments),
    }
    (runtime.emit_event or bus.emit)(SESSION_TURN_COMPLETED, event)
    out.data.update(event)
    return out


def inject_user_message(
    runtime,
    session_key: str,
    text: str,
    *,
    conversation_id: int | None = None,
    actor_id: str = "user",
):
    """Append a user-authored message without driving the agent turn."""
    from runtime.session import RuntimeResult

    if conversation_id is not None:
        session = runtime.sessions.get(session_key)
        if session is None or session.conversation_id != conversation_id:
            session = load_conversation(runtime, session_key, conversation_id)
    else:
        session = get_or_create_session(runtime, session_key)
        ensure_conversation(runtime, session, text)
    msg = {"role": "user", "content": text}
    with session.lock:
        session.history.append(msg)
        if runtime.db and session.conversation_id:
            save_history_message(runtime.db, session.conversation_id, msg)
        bus.emit(SESSION_MESSAGE, {
            "session_key": session.key,
            "role": "user",
            "content": text,
            "actor_id": actor_id,
        })
        persist_marker(runtime, session)
    return RuntimeResult(data={"conversation_id": session.conversation_id})


# ──────────────────────────────────────────────────────────────────────
# Session disposal + restart recovery
# ──────────────────────────────────────────────────────────────────────

def close_session(runtime, session_key: str) -> bool:
    with runtime._sessions_lock:
        existed = runtime.sessions.pop(session_key, None) is not None
    if existed:
        bus.emit(SESSION_CLOSED, {"session_key": session_key})
    return existed


def restore_pending_requests(runtime, session: RuntimeSession) -> None:
    """Re-emit ``approval_requested`` events for any phase frames that were
    mid-flight when the session was last persisted, so frontend adapters
    can re-register them in their pending-request tables and re-prompt
    the user.
    """
    if not runtime.emit_event:
        return
    frames = session.cs.cache.get("phases", []) if isinstance(session.cs.cache, dict) else []
    for frame in frames:
        if getattr(frame, "phase", None) != PHASE_APPROVING_REQUEST:
            continue
        data = getattr(frame, "data", {}) or {}
        req = StateMachineApprovalRequest(
            title=data.get("title") or frame.name or "Input required",
            body=data.get("prompt") or "",
            pending_action=data.get("pending"),
            id=data.get("request_id") or f"approve_{uuid.uuid4().hex}",
            type=data.get("type", "boolean"),
            enum=data.get("enum"),
            default=data.get("default"),
        )
        req.metadata.update({"session_key": session.key, "conversation_id": session.conversation_id})
        runtime._approval_requests.setdefault(req.id, req)
        runtime.emit_event("approval_requested", req)


# ──────────────────────────────────────────────────────────────────────
# Marker + conversation-row helpers
# ──────────────────────────────────────────────────────────────────────

def persist_marker(runtime, session: RuntimeSession) -> None:
    """Snapshot the session's state machine into a system message row.

    Two-marker turns (``busy=True`` before, ``busy=False`` after) are how
    we recover from crashes mid-turn — see ``runtime_dispatch``.
    """
    if runtime.db and session.conversation_id:
        save_state_marker(runtime.db, session.conversation_id, session.to_marker())


def conversation_title(runtime, conversation_id: int) -> str:
    row = runtime.db.get_conversation(conversation_id) if runtime.db else None
    return ((row or {}).get("title") or "").strip() or "New conversation"


def ensure_conversation(runtime, session: RuntimeSession, title_text: str = "") -> None:
    if session.conversation_id is None and runtime.db:
        session.conversation_id = runtime.db.create_conversation(
            (title_text or "New conversation").replace("\n", " ")[:80] or "New conversation",
            kind="subagent" if session.is_subagent else "user",
        )


# ──────────────────────────────────────────────────────────────────────

def _format_history_preview(history: list[dict[str, Any]], limit: int = 2) -> str:
    """Render the last ``limit`` user/assistant turns as a quoted preview.

    Skips system markers, tool calls, and empty content. Trims long bodies
    so the preview stays scannable.
    """
    relevant = []
    for msg in reversed(history):
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        relevant.append((role, content))
        if len(relevant) >= limit:
            break
    if not relevant:
        return ""
    lines = []
    for role, content in reversed(relevant):
        snippet = content if len(content) <= 240 else content[:240].rstrip() + "…"
        prefix = "you" if role == "user" else "agent"
        for i, line in enumerate(snippet.splitlines() or [snippet]):
            lines.append(f"> [{prefix}] {line}" if i == 0 else f"> {line}")
    return "\n".join(lines)
