from __future__ import annotations

"""PokerMonster-style state printer for the conversation state machine.

Use this in REPL `--debug` mode or in tests to see exactly what the state
machine looks like at any moment. Mirrors PokerMonster's `display_info` /
`display_actions` so the same kind of inspection works for both projects.

    from state_machine.debug import display_state, display_actions
    display_state(cs)
    display_actions(cs)
"""

from typing import TYPE_CHECKING

from state_machine.action_map import legal_actions_in_phase

if TYPE_CHECKING:  # pragma: no cover
    from state_machine.conversation import ConversationState


# ANSI colors, matching PokerMonster's palette.
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


def _color_for(actor_kind: str) -> str:
    return GREEN if actor_kind == "user" else RED


def display_state(cs: "ConversationState") -> None:
    """Print the conversation state in PokerMonster's `display_info` style."""
    active = cs.active
    color = _color_for(active.kind)
    print(f"{color}\n=== {active.id.upper()}'S TURN ({active.kind}) ==={RESET}")
    print(f"Phase: {cs.phase}")

    frames = cs.cache.get("phases") or []
    if frames:
        print(f"{CYAN}Phase stack ({len(frames)}):{RESET}")
        for i, frame in enumerate(frames):
            data = getattr(frame, "data", {}) or {}
            args = data.get("args") or {}
            step = frame.step.name if getattr(frame, "step", None) else "—"
            print(f"  [{i}] {frame.phase} :: {frame.action_type} (actor={frame.actor_id}, name={frame.name}, step={step}, args={args})")

    print(f"Participants: " + ", ".join(
        f"{_color_for(p.kind)}{p.id}{RESET}({p.kind})" for p in cs.participants.values()
    ))

    if cs.last_error:
        print(f"{RED}Last error: {cs.last_error.code} — {cs.last_error.message}{RESET}")

    print(f"{MAGENTA}Legal actions: {legal_actions_in_phase(cs.phase) or ['(none)']}{RESET}\n")


def display_actions(cs: "ConversationState") -> None:
    """Print just the currently legal action types, one per line."""
    actions = legal_actions_in_phase(cs.phase)
    if not actions:
        print("(no legal actions for this phase)")
        return
    print("Legal actions:")
    for i, action_type in enumerate(actions):
        print(f"  [{i}] {action_type}")
    print("")


def display_history_tail(cs: "ConversationState", n: int = 5) -> None:
    """Print the last `n` events from the state machine's event history."""
    if not cs.history:
        print("(no events yet)")
        return
    print(f"{CYAN}Last {min(n, len(cs.history))} events:{RESET}")
    for ev in cs.history[-n:]:
        print(f"  · {ev.get('type')} by {ev.get('actor_id')} in {ev.get('phase')}: " + ", ".join(
            f"{k}={v!r}" for k, v in ev.items() if k not in {"type", "actor_id", "phase"}
        ))
