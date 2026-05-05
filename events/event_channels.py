"""
Event channel registry.

Declaring channels in one place is the discipline that keeps the event bus
from becoming a dumping ground. If adding a channel feels like it needs
justification, that's the point — use the bus only when the producer and
consumer are architecturally far apart. For anything tightly coupled or on
the hot path, call the function directly.

Payload shapes are documented here, not enforced at runtime.
"""

# ── Active channels ────────────────────────────────────────────────

APPROVAL_REQUESTED = "approval_requested"
"""Tool needs human approval for a destructive action.
Payload: an ApprovalRequest object."""

APPROVAL_RESOLVED = "approval_resolved"
"""Tool approval was resolved (allowed or denied) by some frontend.
Payload: the resolved ApprovalRequest object."""

TASK_COMPLETED = "task_completed"
"""A task finished successfully.
Payload (path-triggered tasks):
    task_name:    str
    path:         str
    rows_written: int
    duration_s:   float
Payload (event-triggered tasks):
    task_name:    str
    run_id:       str
    rows_written: int
    duration_s:   float"""

TASK_FAILED = "task_failed"
"""A task failed.
Payload (path-triggered tasks):
    task_name: str
    path:      str
    error:     str
Payload (event-triggered tasks):
    task_name: str
    run_id:    str
    error:     str"""

SERVICE_LOADED = "service_loaded"
"""A service finished (un)loading or was swapped. Lets the orchestrator
re-check tasks that were blocked on services without reaching sideways into it. Emitted on load, unload, and hot-reload.
Payload:
    name:   str   — service name (may be None for bulk events)
    loaded: bool  — True after load, False after unload"""

TOOLS_CHANGED = "tools_changed"
"""A tool was registered, re-registered, or unregistered. Lets frontends
rescope running agents so build_plugin / unload_plugin updates take effect
without /restart.
Payload:
    name:   str — tool name
    action: str — 'registered' or 'unregistered'"""

TASKS_CHANGED = "tasks_changed"
"""A task was registered or unregistered. Task registration creates a new
output table via ensure_output_table, so agents rebuild their prompt context.
Payload:
    name:   str — task name
    action: str — 'registered' or 'unregistered'"""

SUBAGENT_RUN = "subagent.run"
"""A scheduled or delegated agent conversation should run one turn.
Payload:
    prompt:          str
    title:           str (optional)
    job_name:        str (optional)
    conversation_id: int (optional)
    input_paths:     list[str] (optional)
    agent:           str (optional)"""

CHAT_MESSAGE_PUSHED = "chat_message_pushed"
"""Something in the system wants to proactively surface a message in the user's
chat view. Used by scheduled subagents pushing notes and any other background
producer that needs to reach the user.
Payload:
    message:  str            — the body text to display (required)
    title:    str (optional) — rendered as a header above the message
    kind:     str (optional) — categorical label (e.g. "note", "alert"); if
                               title is empty, may be used as a fallback header
    source:   str (optional) — identifier for the producer (e.g. "subagent",
                               "timekeeper"); frontends may show this as
                               attribution
    source_id:str (optional) — producer-specific id (subagent run_id,
                               timekeeper job_name, etc.)"""

TOOL_CALL_STARTED = "tool_call_started"
"""The agent started a tool call.
Payload:
    session_key: str
    call_id:     str
    tool_name:   str
    args:        dict"""

TOOL_CALL_FINISHED = "tool_call_finished"
"""The agent finished a tool call.
Payload:
    session_key: str
    call_id:     str
    tool_name:   str
    ok:          bool
    error:       str (optional)"""

COMMAND_CALL_STARTED = "command_call_started"
"""The runtime started a slash command.
Payload:
    session_key:  str
    call_id:      str
    command_name: str
    args:         dict"""

COMMAND_CALL_FINISHED = "command_call_finished"
"""The runtime finished a slash command.
Payload:
    session_key:  str
    call_id:      str
    command_name: str
    ok:           bool
    error:        str (optional)"""


# ── Conversation lifecycle ─────────────────────────────────────────
# Plugins (tools, tasks, services) subscribe to these to react to what
# is happening inside the state machine without having to reach into
# ConversationRuntime directly. Frontends emit and consume them too.

SESSION_CREATED = "session_created"
"""A new RuntimeSession was created (or replaced via /new or load_history).
Payload:
    session_key: str
    is_subagent: bool
    agent_profile: str"""

SESSION_CLOSED = "session_closed"
"""A RuntimeSession was discarded (replaced, deleted, app shutdown).
Payload:
    session_key: str"""

SESSION_PHASE_CHANGED = "session_phase_changed"
"""The session's phase transitioned (awaiting_input -> calling_tool, etc.).
Payload:
    session_key: str
    old_phase:   str
    new_phase:   str"""

SESSION_TURN_CHANGED = "session_turn_changed"
"""Turn priority moved between participants on a session.
Payload:
    session_key: str
    from_actor:  str
    to_actor:    str"""

SESSION_MESSAGE = "session_message"
"""A user- or agent-authored message landed on the session transcript.
Payload:
    session_key: str
    role:        str   — "user" | "assistant" | "tool"
    content:     str
    actor_id:    str"""

SESSION_TURN_COMPLETED = "session_turn_completed"
"""One full user-prompted agent turn completed.
Payload:
    session_key:     str
    conversation_id: int | None
    final_text:      str
    new_messages:    list[dict]
    attachments:     list[str]"""

SESSION_AGENT_PROFILE_CHANGED = "session_agent_profile_changed"
"""A plugin or command changed the agent profile pinned to a session.
Payload:
    session_key:  str
    old_profile:  str
    new_profile:  str"""

SYSTEM_PROMPT_EXTRA_CHANGED = "system_prompt_extra_changed"
"""A plugin added/updated/removed a system prompt extra on a session.
Useful for frontends or subscribers that want to surface what's pinned to
the agent's prompt.
Payload:
    session_key: str
    key:         str
    value:       str | None  (None on removal)"""


# ── Reserved (not yet emitted) ─────────────────────────────────────
# Documented here so future work has an obvious home instead of inventing
# new strings. Add emissions/subscriptions as needs arise.
#
# TABLE_WRITTEN    — after DB.write_outputs, enables reactive aggregate tasks
# SCHEDULE_TICK    — time-based trigger for scheduled tasks
