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

    def __init__(self, *, source: str = "session", source_id: str | None = None, session_key: str | None = None, recorder=None, **extra):
        self._source = source
        self._source_id = source_id
        self._session_key = session_key
        self._recorder = recorder
        self._extra = extra

    def run(self, context, **kwargs):
        message = (kwargs.get("message") or "").strip()
        if not message:
            return ToolResult.failed("message is required.")
        kind = (kwargs.get("kind") or "note").strip().lower()
        kind = kind if kind in NOTIFY_KINDS else "note"
        title = (kwargs.get("title") or "").strip()
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
