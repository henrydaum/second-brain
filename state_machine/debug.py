"""Human-readable snapshots of the conversation state machine.

Originally a PokerMonster-style `print` inspector; now a set of pure
string formatters so the `/debug` command (and tests) can render the live
`ConversationState` — turn, phase stack, participants, legal actions, and
recent events — through any frontend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from state_machine.action_map import legal_actions_in_phase

if TYPE_CHECKING:  # pragma: no cover
    from state_machine.conversation import ConversationState


def format_state(cs: "ConversationState") -> str:
    """Return a multi-line snapshot of the state machine's current shape."""
    active = cs.active
    lines = [
        f"Turn: {active.id} ({active.kind})",
        f"Phase: {cs.phase}",
    ]

    frames = cs.cache.get("phases") or []
    if frames:
        lines.append(f"Phase stack ({len(frames)}):")
        for i, frame in enumerate(frames):
            data = getattr(frame, "data", {}) or {}
            args = data.get("args") or {}
            step = frame.step.name if getattr(frame, "step", None) else "—"
            lines.append(f"  [{i}] {frame.phase} :: {frame.action_type} (actor={frame.actor_id}, name={frame.name}, step={step}, args={args})")

    lines.append("Participants: " + ", ".join(f"{p.id}({p.kind})" for p in cs.participants.values()))

    if cs.last_error:
        lines.append(f"Last error: {cs.last_error.code} — {cs.last_error.message}")

    lines.append(f"Legal actions: {', '.join(legal_actions_in_phase(cs.phase)) or '(none)'}")
    return "\n".join(lines)


def format_recent_events(cs: "ConversationState", n: int = 5) -> str:
    """Return the last ``n`` events from the in-memory state-machine event log.

    Note: this log lives on the live ``ConversationState`` and resets when the
    session is rebuilt from a marker, so it reflects the current process only.
    """
    if not cs.history:
        return "Recent events: (none yet)"
    lines = [f"Recent events (last {min(n, len(cs.history))}):"]
    for ev in cs.history[-n:]:
        detail = ", ".join(f"{k}={v!r}" for k, v in ev.items() if k not in {"type", "actor_id", "phase"})
        lines.append(f"  · {ev.get('type')} by {ev.get('actor_id')} in {ev.get('phase')}" + (f": {detail}" if detail else ""))
    return "\n".join(lines)
