"""Shared event-channel contract for the scheduling packages.

``SPAWN_SUBAGENT`` is the bus channel the spawn-subagent task subscribes to
(``trigger_channels``) and that the schedule-subagent tool tags Timekeeper jobs
with. It lives here — a root-level plugin helper imported as
``..helpers.subagent_events`` — so the task (subscriber) and tool (producer)
share one source of truth instead of each hardcoding the string. This is a
plugin-owned channel; the kernel deliberately does not reserve it.
"""

SPAWN_SUBAGENT = "subagent.spawn"
