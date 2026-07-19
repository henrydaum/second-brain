"""Tests for BaseFrontend's turn-priority typing routing.

SESSION_TURN_CHANGED drives ``render_typing`` off the priority axis (whose
turn it is) rather than per-drive lifecycle, so a typing indicator tracks the
whole logical turn — including barrier-held turns whose interim re-drives keep
priority with the agent — scoped to sessions the frontend owns and to
transports that declare ``supports_typing``.
"""

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities


class _TypingFrontend(BaseFrontend):
    name = "typ"
    capabilities = FrontendCapabilities(supports_typing=True)

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, bool]] = []

    def render_typing(self, session_key, on):
        self.calls.append((session_key, on))

    def _live_session_keys(self):
        return ["mine"]


def _changed(key, to_actor, from_actor="user"):
    return {"session_key": key, "from_actor": from_actor, "to_actor": to_actor}


def test_priority_handoffs_toggle_typing_for_owned_session():
    f = _TypingFrontend()
    f.on_bus_session_turn_changed(_changed("mine", "agent"))
    f.on_bus_session_turn_changed(_changed("mine", "user", from_actor="agent"))
    assert f.calls == [("mine", True), ("mine", False)]


def test_foreign_session_ignored():
    f = _TypingFrontend()
    f.on_bus_session_turn_changed(_changed("spawn_subagent:9", "agent"))
    f.on_bus_session_turn_changed(_changed("spawn_subagent:9", "user", from_actor="agent"))
    assert f.calls == []


def test_supports_typing_false_ignored():
    f = _TypingFrontend()
    f.capabilities = FrontendCapabilities(supports_typing=False)
    f.on_bus_session_turn_changed(_changed("mine", "agent"))
    assert f.calls == []


def test_barrier_held_turn_stays_on_until_user_regains_priority():
    # A barrier-held turn (spawn_agent wait=false) keeps priority with the
    # agent across its interim re-drives, so no SESSION_TURN_CHANGED fires
    # between the initial handoff and the final hand-back. Typing goes on
    # once and stays on until the logical turn truly ends.
    f = _TypingFrontend()
    f.on_bus_session_turn_changed(_changed("mine", "agent"))
    f.on_bus_session_turn_changed(_changed("mine", "user", from_actor="agent"))
    assert f.calls == [("mine", True), ("mine", False)]
    assert f.calls[-1][1] is False


def test_crash_handback_turns_typing_off():
    # A crash forces priority back to the user, emitting a to_actor="user"
    # change — typing clears on that path too.
    f = _TypingFrontend()
    f.on_bus_session_turn_changed(_changed("mine", "agent"))
    f.on_bus_session_turn_changed(_changed("mine", "user", from_actor="agent"))
    assert f.calls[-1] == ("mine", False)


def test_render_typing_exception_is_swallowed():
    f = _TypingFrontend()

    def _boom(_key, _on):
        raise RuntimeError("boom")

    f.render_typing = _boom
    f.on_bus_session_turn_changed(_changed("mine", "agent"))  # must not raise
