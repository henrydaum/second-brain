"""Per-conversation runtime state.

Two dataclasses live here, kept apart from the runtime so each can be read
on its own:

- :class:`RuntimeSession` is the in-memory bag of *everything* the runtime
  needs to reason about a single conversation: the state machine, the
  provider-shaped history list, persistence id, profile pinning, plugin-
  pinned tools, the per-session lock, and the cancel event.
- :class:`RuntimeResult` is the transport-neutral output the runtime hands
  back to a frontend after every action.
"""

from __future__ import annotations


import threading
from dataclasses import dataclass, field
from typing import Any

from state_machine.conversation import ConversationState
from state_machine.errors import ActionResult
from runtime.notifications import DEFAULT_NOTIFICATION_MODE


@dataclass
class RuntimeResult:
    """Transport-neutral output for adapters to render."""

    ok: bool = True
    messages: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    buttons: list[dict[str, str]] = field(default_factory=list)
    form: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def add_action_result(self, result: ActionResult) -> "RuntimeResult":
        """Handle add action result."""
        self.ok = self.ok and result.ok
        self.events.extend(result.events)
        if result.message:
            self.messages.append(result.message)
        if result.error:
            self.error = result.error.to_dict()
            self.messages.append(result.error.message)
        return self


@dataclass
class RuntimeSession:
    """All mutable state for one frontend conversation/session."""

    key: str
    cs: ConversationState
    history: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: int | None = None
    busy: bool = False
    active_agent_profile: str = "default"
    # Subagent / specialist sessions pin a profile and can register extra tool
    # instances that are not part of the global tool_registry. When None /
    # empty, the session follows the runtime's active profile and registry.
    profile_override: str | None = None
    # The frontend transport that owns this session ("repl", "telegram", ...).
    # Set by BaseFrontend on first submit; lets the runtime apply that
    # frontend's profile (agent scope + command access). None for background
    # drivers, which follow the global active profile.
    frontend_name: str | None = None
    # The user whose data this session acts on. Ephemeral live binding set by the
    # frontend (like ``attended``) — None means the base user (DEFAULT_USER_ID).
    # Deliberately NOT persisted in to_marker(): ownership lives on the
    # conversation row, and persisting it here would let loading a conversation
    # silently rebind the session's identity. Identity flows frontend → session;
    # ownership flows conversation row → guard; the two never cross.
    user_id: int | None = None
    extra_tool_instances: list = field(default_factory=list)
    system_prompt_extras: dict[str, Any] = field(default_factory=dict)
    # Free-form per-plugin state bag, keyed by plugin name. The substrate for
    # on-demand plugins to stash session-scoped state without core-defined
    # fields. Persisted with the marker.
    plugin_state: dict[str, dict] = field(default_factory=dict)
    notification_mode: str = DEFAULT_NOTIFICATION_MODE
    restore_notices: list[str] = field(default_factory=list)
    # Whether a human is present at this session right now (can answer an
    # interactive prompt and see output). None = defer to the kernel's global
    # single-active rule (REPL/Telegram, background drivers). True/False = the
    # owning frontend manages attendance explicitly (concurrent multi-user
    # frontends, e.g. a website setting it on socket connect/disconnect).
    # Ephemeral live state — deliberately NOT persisted in to_marker(), so it
    # resets to None (defer-to-global) across restarts.
    attended: bool | None = None
    has_compaction_checkpoint: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_marker(self) -> dict[str, Any]:
        """Handle to marker."""
        state = self.cs.to_dict()
        state.update({
            "conversation_id": self.conversation_id,
            "active_agent_profile": self.active_agent_profile,
            "profile_override": self.profile_override,
            "frontend_name": self.frontend_name,
            "notification_mode": self.notification_mode,
            "system_prompt_extras": self.system_prompt_extras,
            "plugin_state": self.plugin_state,
            "busy": self.busy,
        })
        return state


class SessionConflict(RuntimeError):
    """Raised when a session_key is requested for a conversation_id that
    conflicts with an existing live session.

    Without this guard a second binding could silently stomp on the first;
    with it, the second bind fails loudly.
    """

    def __init__(self, session_key: str, existing_id: int | None, requested_id: int | None):
        """Initialize the session conflict."""
        super().__init__(
            f"Session '{session_key}' is already bound to conversation {existing_id}; "
            f"cannot rebind to {requested_id}."
        )
        self.session_key = session_key
        self.existing_id = existing_id
        self.requested_id = requested_id
