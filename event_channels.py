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
"""Tool needs human approval for a destructive action. Sync round-trip.
Payload (before bus.request enriches it):
    command:  str  — the action being proposed (e.g. shell command, plugin op)
    reason:   str  — justification the agent gave
Reply: subscriber sets result[0] = bool (True=allow, False=deny), then reply.set()."""

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


# ── Reserved (not yet emitted) ─────────────────────────────────────
# Documented here so future work has an obvious home instead of inventing
# new strings. Add emissions/subscriptions as needs arise.
#
# TABLE_WRITTEN    — after DB.write_outputs, enables reactive aggregate tasks
# SCHEDULE_TICK    — time-based trigger for scheduled tasks
