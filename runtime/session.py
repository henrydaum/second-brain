from __future__ import annotations

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

import threading
from dataclasses import dataclass, field
from typing import Any

from state_machine.conversation import ConversationState
from state_machine.errors import ActionResult


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
    # instances (e.g. NotifyTool) that are not part of the global
    # tool_registry. When None / empty, the session follows the runtime's
    # active profile and uses the global tool registry.
    profile_override: str | None = None
    extra_tool_instances: list = field(default_factory=list)
    is_subagent: bool = False
    subagent_meta: dict[str, Any] = field(default_factory=dict)
    system_prompt_extras: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_marker(self) -> dict[str, Any]:
        state = self.cs.to_dict()
        state.update({
            "conversation_id": self.conversation_id,
            "active_agent_profile": self.active_agent_profile,
            "profile_override": self.profile_override,
            "is_subagent": self.is_subagent,
            "subagent_meta": self.subagent_meta,
            "system_prompt_extras": self.system_prompt_extras,
            "busy": self.busy,
        })
        return state


class SessionConflict(RuntimeError):
    """Raised when a session_key is requested for a conversation_id that
    conflicts with an existing live session.

    Realistic case: two cron jobs configured with the same name both target
    ``subagent:<job_name>``. Without this guard the second job silently
    stomps on the first; with it, the second job fails loudly.
    """

    def __init__(self, session_key: str, existing_id: int | None, requested_id: int | None):
        super().__init__(
            f"Session '{session_key}' is already bound to conversation {existing_id}; "
            f"cannot rebind to {requested_id}."
        )
        self.session_key = session_key
        self.existing_id = existing_id
        self.requested_id = requested_id
