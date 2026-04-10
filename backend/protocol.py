"""
Event and command definitions for the backend WebSocket protocol.

All messages are JSON objects with a ``type`` field. The backend sends
events; frontends send commands.  This module is pure data — no I/O,
no external dependencies — so frontends can import it for type checking.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_id() -> str:
    """Generate a short unique ID for requests, sessions, turns, etc."""
    return uuid.uuid4().hex[:12]


def serialize(obj: Any) -> str:
    """Serialize a dataclass (or dict) to a JSON string."""
    data = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    return json.dumps(data, default=str)


def deserialize(raw: str | bytes) -> dict:
    """Parse a JSON message into a plain dict."""
    return json.loads(raw)


# ===================================================================
# BACKEND → FRONTEND  EVENTS
# ===================================================================

# --- Session lifecycle ---

@dataclass
class SessionCreated:
    type: str = "session.created"
    session_id: str = ""
    conversation_id: int | None = None


@dataclass
class SessionResumed:
    type: str = "session.resumed"
    session_id: str = ""
    conversation_id: int = 0
    history: list[dict] = field(default_factory=list)


@dataclass
class SessionError:
    type: str = "session.error"
    session_id: str = ""
    error: str = ""


# --- Agent chat events ---

@dataclass
class AgentThinking:
    type: str = "agent.thinking"
    session_id: str = ""
    turn_id: str = ""
    content: str = ""


@dataclass
class AgentToolCall:
    type: str = "agent.tool_call"
    session_id: str = ""
    turn_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)


@dataclass
class AgentToolResult:
    type: str = "agent.tool_result"
    session_id: str = ""
    turn_id: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    success: bool = True
    error: str = ""
    llm_summary: str = ""
    data: Any = None
    display_paths: list[str] = field(default_factory=list)


@dataclass
class AgentMessage:
    type: str = "agent.message"
    session_id: str = ""
    turn_id: str = ""
    role: str = ""
    content: str = ""
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class AgentDone:
    type: str = "agent.done"
    session_id: str = ""
    turn_id: str = ""
    content: str = ""


@dataclass
class AgentError:
    type: str = "agent.error"
    session_id: str = ""
    turn_id: str = ""
    error: str = ""


@dataclass
class AgentCancelled:
    type: str = "agent.cancelled"
    session_id: str = ""
    turn_id: str = ""


# --- Approval ---

@dataclass
class ApprovalRequest:
    type: str = "approval.request"
    session_id: str = ""
    approval_id: str = ""
    command: str = ""
    justification: str = ""


# --- Command response ---

@dataclass
class CommandResult:
    type: str = "command.result"
    session_id: str = ""
    request_id: str = ""
    text: str = ""


# --- System broadcasts ---

@dataclass
class SystemToolChanged:
    type: str = "system.tool_changed"
    action: str = ""      # "registered", "unregistered", "enabled", "disabled"
    name: str = ""


@dataclass
class SystemServiceChanged:
    type: str = "system.service_changed"
    name: str = ""
    loaded: bool = False


# ===================================================================
# FRONTEND → BACKEND  COMMANDS
# ===================================================================

@dataclass
class SessionCreateCmd:
    type: str = "session.create"
    request_id: str = ""
    conversation_id: int | None = None


@dataclass
class SessionDestroyCmd:
    type: str = "session.destroy"
    request_id: str = ""
    session_id: str = ""


@dataclass
class ChatSendCmd:
    type: str = "chat.send"
    request_id: str = ""
    session_id: str = ""
    message: str = ""


@dataclass
class ChatCancelCmd:
    type: str = "chat.cancel"
    request_id: str = ""
    session_id: str = ""


@dataclass
class CommandSendCmd:
    type: str = "command.send"
    request_id: str = ""
    session_id: str = ""
    command: str = ""
    arg: str = ""


@dataclass
class ApprovalResponseCmd:
    type: str = "approval.response"
    approval_id: str = ""
    approved: bool = False


@dataclass
class ToolCallCmd:
    type: str = "tool.call"
    request_id: str = ""
    session_id: str = ""
    tool_name: str = ""
    arguments: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Command type map  (type string → dataclass)
# ---------------------------------------------------------------------------

COMMAND_TYPES: dict[str, type] = {
    "session.create": SessionCreateCmd,
    "session.destroy": SessionDestroyCmd,
    "chat.send": ChatSendCmd,
    "chat.cancel": ChatCancelCmd,
    "command.send": CommandSendCmd,
    "approval.response": ApprovalResponseCmd,
    "tool.call": ToolCallCmd,
}


def parse_command(raw: dict) -> Any:
    """Convert a raw dict into the appropriate command dataclass.

    Raises ``ValueError`` for unknown command types.
    """
    msg_type = raw.get("type", "")
    cls = COMMAND_TYPES.get(msg_type)
    if cls is None:
        raise ValueError(f"Unknown command type: {msg_type!r}")
    # Only pass fields the dataclass actually has
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return cls(**filtered)
