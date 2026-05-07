"""
Notify tool — pushes a user-visible message to chat from inside a session.

This tool is special: it carries per-session construction state (source,
source_id, session_key, recorder closure) and is instantiated manually by
its caller (see plugins/tasks/task_run_subagent.py), not via the generic
plugin discoverer. `auto_register = False` keeps discovery from picking it
up as a stateless, globally-registered tool.
"""

import time
from dataclasses import dataclass

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED
from plugins.BaseTool import BaseTool, ToolResult

NOTIFY_KINDS = ["note", "finding", "brief", "alert"]


@dataclass
class NotificationRecord:
    kind: str
    title: str
    message: str
    sent_at: float


class NotifyTool(BaseTool):
    name = "notify"
    description = (
        "Send a user-visible notification to chat from this session. Use this "
        "for reminders, briefs, alerts, findings, or any update the user should see."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The message to send to chat."},
            "kind": {"type": "string", "enum": NOTIFY_KINDS, "description": "Message type.", "default": "note"},
            "title": {"type": "string", "description": "Optional short title."},
        },
        "required": ["message"],
    }
    max_calls = 10
    background_safe = True
    auto_register = False

    def __init__(self, *, source: str = "session", source_id: str | None = None, session_key: str | None = None, recorder=None, conversation_id: int | None = None, **extra):
        self._source = source
        self._source_id = source_id
        self._session_key = session_key
        self._recorder = recorder
        self._conversation_id = conversation_id
        self._extra = extra

    def _build_load_suffix(self, context) -> str:
        cid = self._conversation_id
        db = getattr(context, "db", None)
        if cid is None or db is None:
            return ""
        try:
            conv = db.get_conversation(cid)
        except Exception:
            return ""
        if not conv:
            return ""
        category = (conv.get("category") or "").strip()
        if not category:
            return ""
        import shlex
        cmd = f"/conversations {shlex.quote(category)} {cid} 'Load conversation'"
        return f"\n\nLoad this conversation: `{cmd}`"

    def run(self, context, **kwargs):
        message = (kwargs.get("message") or "").strip()
        if not message:
            return ToolResult.failed("message is required.")
        kind = (kwargs.get("kind") or "note").strip().lower()
        kind = kind if kind in NOTIFY_KINDS else "note"
        title = (kwargs.get("title") or "").strip()
        message = message + self._build_load_suffix(context)
        sent_at = time.time()
        record = NotificationRecord(kind, title, message, sent_at)
        if self._recorder:
            self._recorder(record)
        payload = {
            "message": message, "title": title, "kind": kind,
            "source": self._source, "source_id": self._source_id,
            **self._extra,
        }
        if self._session_key:
            payload["session_key"] = self._session_key
        bus.emit(CHAT_MESSAGE_PUSHED, payload)
        return ToolResult(
            data={**payload, "sent_at": sent_at},
            llm_summary="Sent a user-visible notification to chat.",
        )
