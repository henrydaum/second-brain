"""Per-conversation notification mode.

Each conversation persists a notification mode alongside its agent profile.
The mode controls three things at once: whether NotifyTool is attached to
the session's tool registry, what the system prompt says about
notifications, and whether a fallback push fires when the agent finishes
a background turn without calling notify.

Modes:
    off        — no NotifyTool, no system-prompt addition.
    all        — NotifyTool attached; agent is told to use it; if a
                 background turn completes without a notify call, the
                 final answer is relayed via CHAT_MESSAGE_PUSHED.
    important  — NotifyTool attached; agent is told to call it only when
                 something is genuinely worth surfacing. No fallback.

The "background" distinction (where fallback fires) reuses the same
active-conversation heuristic as ``background_safe`` tool gating — a
session whose key differs from ``runtime.active_session_key`` is, by
definition, running unattended.
"""

from __future__ import annotations

import shlex
import time
from typing import Any, Callable

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED


NOTIFICATION_MODES = ("all", "important", "off")
DEFAULT_NOTIFICATION_MODE = "all"


def notification_mode(value: Any, default: str = DEFAULT_NOTIFICATION_MODE) -> str:
    """Normalize an arbitrary input to one of NOTIFICATION_MODES."""
    mode = str(value or default).strip().lower()
    return mode if mode in NOTIFICATION_MODES else default


def notify_block(mode: str) -> str:
    """System-prompt suffix describing this conversation's notify behavior."""
    mode = notification_mode(mode)
    if mode == "off":
        return ""
    if mode == "important":
        return (
            "\n\n## Notifications\n"
            "The notify tool is available but should be used only when something noteworthy comes up — "
            "a real finding, an alert, a needed nudge, or information the user actually needs to see now. "
            "Routine completion is not important; stay silent in that case."
        )
    return (
        "\n\n## Notifications\n"
        "The notify tool is the main way to send a user-visible message. "
        "Use it for reminders, alerts, briefs, findings, check-ins, or anything the user should actually see in chat. "
        "If you do not call notify during a background run, the system will fall back to surfacing your final answer "
        "as a single push so the user is not left in the dark."
    )


def load_conversation_suffix(db, conversation_id: int | None) -> str:
    """Render the trailing line that lets the user jump to the conversation
    a notification came from."""
    if conversation_id is None or db is None:
        return ""
    try:
        conv = db.get_conversation(conversation_id)
    except Exception:
        return ""
    if not conv:
        return ""
    category = (conv.get("category") or "").strip()
    if not category:
        return ""
    cmd = f"/conversations {shlex.quote(category)} {conversation_id} 'Load conversation'"
    return f"\n\nLoad this conversation: `{cmd}`"


def make_session_notify_tool(
    *,
    session_key: str,
    conversation_id: int | None,
    recorder: Callable[[Any], None],
):
    """Build a NotifyTool wired to this session's recorder."""
    from plugins.tools.tool_notify import NotifyTool

    return NotifyTool(
        source="session",
        source_id=session_key,
        session_key=session_key,
        conversation_id=conversation_id,
        recorder=recorder,
    )


def emit_fallback_push(
    *,
    session_key: str,
    conversation_id: int | None,
    title: str,
    final_text: str,
    db,
) -> None:
    """Emit a CHAT_MESSAGE_PUSHED carrying the agent's final answer.

    Used when ``mode == "all"`` and a background turn finished without the
    agent calling notify. Mirrors the payload shape NotifyTool emits, so
    downstream subscribers (frontends) handle both paths identically.
    """
    text = (final_text or "").strip()
    if not text:
        return
    message = text + load_conversation_suffix(db, conversation_id)
    bus.emit(CHAT_MESSAGE_PUSHED, {
        "message": message,
        "title": (title or "").strip(),
        "kind": "brief",
        "source": "session",
        "source_id": session_key,
        "session_key": session_key,
        "sent_at": time.time(),
    })
