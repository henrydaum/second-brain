"""Per-conversation notification mode.

Background sessions either replay the agent's final answer to chat ("on")
or stay silent ("off"). Older "all" / "important" markers normalize to
"on" so existing scheduled conversations keep surfacing results.
"""

from __future__ import annotations

import shlex
import time
from typing import Any

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED


NOTIFICATION_MODES = ("on", "off")
DEFAULT_NOTIFICATION_MODE = "on"


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
    category = (conv.get("category") or "").strip() or "Main"
    cmd = f"/conversations {shlex.quote(category)} {conversation_id} 'Load conversation'"
    return f"\n\nLoad this conversation: `{cmd}`"


def emit_fallback_push(
    *,
    session_key: str,
    conversation_id: int | None,
    title: str,
    final_text: str,
    db,
) -> None:
    """Emit a CHAT_MESSAGE_PUSHED carrying the agent's final answer."""
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
