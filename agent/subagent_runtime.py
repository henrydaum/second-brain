import time
from dataclasses import dataclass

from plugins.BaseTool import BaseTool, ToolResult
from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED

SUBAGENT_RUN_CHANNEL = "subagent.run"
SUBAGENT_PUSH_KINDS = ["note", "finding", "brief", "alert"]
SUBAGENT_NOTIFICATION_MODES = ("all", "important", "off")
SUBAGENT_DEFAULT_NOTIFICATION_MODE = "all"


@dataclass
class SubagentPushRecord:
    kind: str
    title: str
    message: str
    sent_at: float


class MessageTool(BaseTool):
    name = "message"
    description = (
        "Send a user-visible message to chat from a scheduled subagent run. "
        "Use this for reminders, briefs, alerts, findings, or any update the "
        "user should actually see during an unattended job. The user can "
        "reply between runs via /message <job> <text>; their replies appear "
        "in your conversation history on the next wake."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send to chat.",
            },
            "kind": {
                "type": "string",
                "enum": SUBAGENT_PUSH_KINDS,
                "description": "Message type.",
                "default": "note",
            },
            "title": {
                "type": "string",
                "description": "Optional short title.",
            },
        },
        "required": ["message"],
    }
    requires_services = []
    max_calls = 10
    background_safe = True

    def __init__(self, run_id: str, job_name: str, recorder):
        self._run_id = run_id
        self._job_name = job_name
        self._recorder = recorder

    def run(self, context, **kwargs):
        message = (kwargs.get("message") or "").strip()
        if not message:
            return ToolResult.failed("message is required.")

        kind = (kwargs.get("kind") or "note").strip().lower()
        if kind not in SUBAGENT_PUSH_KINDS:
            kind = "note"
        title = (kwargs.get("title") or "").strip()
        sent_at = time.time()

        record = SubagentPushRecord(
            kind=kind,
            title=title,
            message=message,
            sent_at=sent_at,
        )
        self._recorder(record)

        bus.emit(CHAT_MESSAGE_PUSHED, {
            "message": message,
            "title": title,
            "kind": kind,
            "source": "subagent",
            "source_id": self._run_id,
            "job_name": self._job_name,
        })

        return ToolResult(
            data={
                "run_id": self._run_id,
                "job_name": self._job_name,
                "kind": kind,
                "title": title,
                "message": message,
                "sent_at": sent_at,
            },
            llm_summary="Sent a user-visible message to the chat.",
        )
