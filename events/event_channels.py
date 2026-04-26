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
re-check tasks that were blocked on services without the controller
reaching sideways into it. Emitted on load, unload, and hot-reload.
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
output table via ensure_output_table, so scoped agents need to rebuild
their ScopedDatabase to see the new table in views and system prompt.
Payload:
    name:   str — task name
    action: str — 'registered' or 'unregistered'"""

CHAT_MESSAGE_PUSHED = "chat_message_pushed"
"""Something in the system wants to proactively surface a message in the user's
chat view. Used by scheduled subagents pushing notes, the timekeeper announcing
a fired job, and any other background producer that needs to reach the user.
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


# ── Reserved (not yet emitted) ─────────────────────────────────────
# Documented here so future work has an obvious home instead of inventing
# new strings. Add emissions/subscriptions as needs arise.
#
# TABLE_WRITTEN    — after DB.write_outputs, enables reactive aggregate tasks
# SCHEDULE_TICK    — time-based trigger for scheduled tasks
