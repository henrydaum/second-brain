"""Notifications: what the system tells the user, as its own kind of event.

Two things live here, and they are related but not the same.

**Per-conversation notification mode** decides whether a background session
replays its final answer at all ("on") or stays silent ("off"). Older
"all" / "important" markers normalize to "on" so existing scheduled
conversations keep surfacing results.

**``notify()``** is the one way anything in the kernel raises a notification.
It exists because ``CHAT_MESSAGE_PUSHED`` carried two populations a frontend
had to tell apart and could not: text belonging to the agent's turn, and
announcements from elsewhere that merely had nowhere else to appear. A plugin
registering itself and the agent answering a question arrived as the same
``render_messages`` call.

The split is decided *at the emit site*, never inferred from the channel. That
matters more than it looks: the two things staying on ``CHAT_MESSAGE_PUSHED``
(the model's mid-turn narration, a tool's ``sdk.ui.render``) are recognisable
today only by being pushes made while the agent's turn owns the session, and a
rule like that is exactly the kind a future producer lands on the wrong side of
without anybody noticing. A notification whose zone never shows it looks
identical to a system with nothing to say.

**``source`` is not stated by whoever asked.** Kernel producers pass a literal
they do not choose at runtime. Sandboxed code passes nothing at all — the
handler reads the leaf of the live provenance chain, which is the part of a
chain a box cannot state about itself. That reading is done in
``sandbox/handlers/kernel.py``, not here: the chain belongs to the sandbox, and
the kernel's notification layer should not have to know it exists.
"""

from __future__ import annotations

import logging
import shlex
import time
from typing import Any

from events.event_bus import bus
from events.event_channels import NOTIFICATION_PUSHED

logger = logging.getLogger("Notifications")


NOTIFICATION_MODES = ("on", "off")
DEFAULT_NOTIFICATION_MODE = "on"

# What a frontend may key styling off. Deliberately four and deliberately
# generic: a level says how much the user should care, and anything finer —
# which subsystem, which job — is what ``source`` and ``source_id`` are for.
NOTIFICATION_LEVELS = ("info", "success", "warning", "error")
DEFAULT_LEVEL = "info"

# The database notifications are persisted to. A reference, not a lifecycle —
# the same arrangement ``parsing.bind_services()`` uses, and for the same
# reason: the producers here (the plugin watcher above all) reach the bus
# directly and hold no runtime to thread a handle through.
_DB: Any = None


def bind_db(db) -> None:
    """Point the notification layer at the live database.

    Called once from the composition root. Until it is, ``notify`` still
    delivers — it just has nowhere to persist, which is the correct behaviour
    for anything raised before the database exists.
    """
    global _DB
    _DB = db


def notification_mode(value: Any, default: str = DEFAULT_NOTIFICATION_MODE) -> str:
    """Normalize an arbitrary input to one of NOTIFICATION_MODES."""
    raw = str(value or default).strip().lower()
    mode = {"all": "on", "important": "on", "true": "on", "yes": "on", "1": "on",
            "false": "off", "no": "off", "0": "off"}.get(raw, raw)
    return mode if mode in NOTIFICATION_MODES else default


def notify_block(mode: str) -> str:
    """System-prompt suffix describing this conversation's notify behavior."""
    mode = notification_mode(mode)
    if mode == "off":
        return ""
    return (
        "\n\n## Notifications\n"
        "Notifications are on for this background conversation. The final answer you give for this run will be sent "
        "to the user, so make your last message the concise update they should see."
    )


def load_conversation_command(db, conversation_id: int | None) -> str:
    """The slash command that jumps to a conversation, or "".

    Split out from the rendered suffix below because the two consumers want
    different things from it: a text frontend wants a line to print, and a
    structured client wants ``conversation_id`` and its own affordance. Handing
    a React panel a backticked ``/conversations`` string would make it render a
    terminal command as the way to open a conversation it can already open.
    """
    if conversation_id is None or db is None:
        return ""
    try:
        conv = db.get_conversation(conversation_id)
    except Exception:
        return ""
    if not conv:
        return ""
    category = (conv.get("category") or "").strip() or "Main"
    return f"/conversations {shlex.quote(category)} {conversation_id} 'Load conversation'"


def load_conversation_suffix(db, conversation_id: int | None) -> str:
    """Render the trailing line that lets the user jump to the conversation
    a notification came from."""
    cmd = load_conversation_command(db, conversation_id)
    return f"\n\nLoad this conversation: `{cmd}`" if cmd else ""


def notify(*, title: str = "", body: str = "", source: str = "system",
           source_id: str | None = None, level: str = DEFAULT_LEVEL,
           session_key: str | None = None,
           source_session_key: str | None = None,
           conversation_id: int | None = None,
           user_id: int | None = None,
           persist: bool = True, db=None) -> int | None:
    """Raise one notification: persist it, then put it on the bus.

    ``session_key`` is *delivery* — where to show it — and is usually unset,
    which broadcasts to whatever surface the user is actually looking at.
    ``source_session_key`` is *origin*, which for a scheduled agent is a
    background session nobody is watching. Conflating the two would deliver a
    background result to a session with no frontend attached, i.e. nowhere.

    ``persist=False`` is for genuinely ephemeral progress — "Compacting
    conversation…" is worth interrupting for and not worth keeping.

    Returns the persisted row id, or None. Defensive throughout and by
    design: a notification must never break the thing that had something to
    say, which is also why the emit happens even when the write failed.
    """
    try:
        title = (title or "").strip()
        body = (body or "").strip()
        if not title and not body:
            return None
        level = level if level in NOTIFICATION_LEVELS else DEFAULT_LEVEL
        database = db if db is not None else _DB

        # Guarded separately from the emit below, and that separation is the
        # point rather than caution: delivering the notification must not
        # depend on having stored it. ``Database.record_notification`` already
        # swallows its own failures, but a stub, a closed connection or a
        # half-built kernel would raise straight through — and losing the live
        # delivery because the panel's copy could not be written is the wrong
        # failure of the two.
        notification_id = None
        if persist and database is not None:
            try:
                notification_id = database.record_notification(
                    title=title, body=body, source=source, source_id=source_id,
                    level=level, session_key=session_key,
                    conversation_id=conversation_id, user_id=user_id)
            except Exception:
                logger.exception("could not persist a notification (ignored)")

        payload: dict[str, Any] = {
            "title": title,
            "body": body,
            "source": source,
            "level": level,
            "sent_at": time.time(),
        }
        if source_id:
            payload["source_id"] = source_id
        if session_key:
            payload["session_key"] = session_key
        if source_session_key:
            payload["source_session_key"] = source_session_key
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
            hint = load_conversation_command(database, conversation_id)
            if hint:
                payload["load_hint"] = hint
        if notification_id is not None:
            payload["notification_id"] = notification_id

        bus.emit(NOTIFICATION_PUSHED, payload)
        return notification_id
    except Exception:
        logger.exception("notify failed (ignored)")
        return None


def emit_fallback_push(
    *,
    session_key: str,
    conversation_id: int | None,
    title: str,
    final_text: str,
    db,
) -> None:
    """Notify with a background turn's final answer.

    The conversation's own reply, surfaced somewhere it can be seen. It is the
    agent speaking, but not into the chat the reader is looking at — which is
    what makes it a notification rather than conversation.
    """
    text = (final_text or "").strip()
    if not text:
        return
    notify(
        title=(title or "").strip() or "Background agent finished",
        body=text,
        source="session",
        source_id=session_key,
        level="info",
        source_session_key=session_key,
        conversation_id=conversation_id,
        db=db,
    )


def announce_config_change(payload, *, session_key: str | None) -> None:
    """Tell the user which settings just changed.

    Every persisted write announces itself, whether or not it needed approval:
    a command the user typed writes config without a dialog (see
    ``sandbox.policy.classify``), so this is the only thing that keeps the
    change visible. Key *names* only — config holds tokens.

    Defensive throughout. A config write must never fail because announcing it
    did, which is also why this is a bus subscriber rather than a call inside
    ``config_manager``.
    """
    try:
        keys = sorted((payload or {}).get("keys") or [])
        if not keys or not session_key:
            return
        notify(
            title="Settings changed",
            body=", ".join(map(str, keys)),
            source="config",
            source_id=(payload or {}).get("scope") or "core",
            level="info",
            source_session_key=session_key,
        )
    except Exception:
        pass
