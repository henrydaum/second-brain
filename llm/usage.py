"""Recording what model calls consumed.

One helper, two callers: the conversation loop (every agent turn —
foreground, subagent, scheduled) and the ``agent.complete`` Request (a plugin
asking the kernel for a completion: the compactor, titling). Not ``Brain.chat``
itself, which is where every call passes but where nobody knows whose call it
is; a row with no session, conversation or user is the ledger's old NULL-column
failure, invisible because an empty column looks like a table nobody asked yet.

The figures are the backend's four counts (``USAGE_FIELDS``), recorded as
given. Normalizing a provider's usage block is the backend's job; the kernel
knows no provider and adds nothing here but whose call it was.
"""

from __future__ import annotations

from sandbox.guest.llm import USAGE_FIELDS


def record(db, *, identity, brain, response, origin: str, ok: bool,
           duration_s: float | None = None) -> None:
    """Append one call to ``llm_usage``. Best-effort, never raises.

    ``identity`` is ``(session_key, conversation_id, user_id)``, the shape
    ``runtime.ledger.identity_of`` answers. ``response`` may be ``None`` for a
    call that raised — it is still a call, and a failed one is worth counting.
    """
    if db is None or not hasattr(db, "record_llm_usage"):
        return
    session_key, conversation_id, user_id = identity
    db.record_llm_usage(
        origin=origin, ok=ok, session_key=session_key,
        conversation_id=conversation_id, user_id=user_id,
        # A profile's key *is* the model name (``Brain.model_name``), so one
        # column says both.
        model=getattr(brain, "model_name", None) or None,
        duration_s=None if duration_s is None else round(duration_s, 3),
        **{name: getattr(response, name, None) for name in USAGE_FIELDS},
    )
