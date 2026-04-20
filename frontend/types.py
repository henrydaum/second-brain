from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlatformCapabilities:
    supports_typing: bool = False
    supports_buttons: bool = False
    supports_message_edit: bool = False
    supports_attachments_in: bool = False
    supports_attachments_out: bool = False
    supports_inline_forms: bool = False
    supports_proactive_push: bool = False
    supports_rich_text: bool = False
    max_message_chars: int | None = None
    max_upload_size: int | None = None


@dataclass
class FrontendSession:
    platform: str
    user_id: str
    chat_id: str
    thread_id: str | None = None
    conversation_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontendEvent:
    type: str
    session: FrontendSession
    text: str = ""
    attachments: list[str] = field(default_factory=list)
    command_name: str | None = None
    command_arg: str | None = None
    callback_id: str | None = None
    callback_value: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class FrontendAction:
    type: str
    text: str = ""
    attachments: list[str] = field(default_factory=list)
    buttons: list[dict[str, str]] = field(default_factory=list)
    form: dict[str, Any] = field(default_factory=dict)
    status_id: str | None = None
    parse_mode: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
