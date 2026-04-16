import time
from dataclasses import dataclass

from Stage_3.BaseTool import BaseTool, ToolResult
from event_bus import bus
from event_channels import SUBAGENT_MESSAGE_PUSHED

SUBAGENT_RUN_CHANNEL = "subagent.run"
SUBAGENT_PUSH_KINDS = ["note", "finding", "brief", "alert"]


@dataclass
class SubagentPushRecord:
    kind: str
    title: str
    message: str
    sent_at: float


class PushSubagentMessageTool(BaseTool):
    name = "push_subagent_message"
    description = (
        "Send a message to the user's chat. This is the background subagent's "
        "main way to directly communicate something the user should actually "
        "see, such as a reminder, brief, alert, or finding."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to send.",
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
    agent_enabled = True
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

        bus.emit(SUBAGENT_MESSAGE_PUSHED, {
            "run_id": self._run_id,
            "job_name": self._job_name,
            "kind": kind,
            "title": title,
            "message": message,
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
