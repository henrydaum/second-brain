"""Tests for BaseFrontend's turn-lifecycle typing routing.

SESSION_TURN_STARTED / SESSION_TURN_COMPLETED drive ``render_typing`` so a
typing indicator tracks the whole logical turn (including barrier-held turns),
scoped to sessions the frontend owns and to transports that declare
``supports_typing``.
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


def _started(key):
    return {"session_key": key, "conversation_id": 1, "actor_id": "agent"}


def _completed(key, ok=True):
    return {"session_key": key, "conversation_id": 1, "ok": ok}


def test_turn_events_toggle_typing_for_owned_session():
    f = _TypingFrontend()
    f.on_bus_session_turn_started(_started("mine"))
    f.on_bus_session_turn_completed(_completed("mine"))
    assert f.calls == [("mine", True), ("mine", False)]


def test_foreign_session_ignored():
    f = _TypingFrontend()
    f.on_bus_session_turn_started(_started("spawn_subagent:9"))
    f.on_bus_session_turn_completed(_completed("spawn_subagent:9"))
    assert f.calls == []


def test_supports_typing_false_ignored():
    f = _TypingFrontend()
    f.capabilities = FrontendCapabilities(supports_typing=False)
    f.on_bus_session_turn_started(_started("mine"))
    assert f.calls == []


def test_restarted_turn_nets_off_without_interim_off():
    # The emit site suppresses COMPLETED for interim restart drives, so a
    # barrier-held turn arrives as STARTED, STARTED, COMPLETED.
    f = _TypingFrontend()
    f.on_bus_session_turn_started(_started("mine"))
    f.on_bus_session_turn_started(_started("mine"))
    f.on_bus_session_turn_completed(_completed("mine"))
    assert f.calls == [("mine", True), ("mine", True), ("mine", False)]
    assert f.calls[-1][1] is False


def test_crash_completion_turns_typing_off():
    f = _TypingFrontend()
    f.on_bus_session_turn_started(_started("mine"))
    f.on_bus_session_turn_completed(_completed("mine", ok=False))
    assert f.calls[-1] == ("mine", False)


def test_render_typing_exception_is_swallowed():
    f = _TypingFrontend()

    def _boom(_key, _on):
        raise RuntimeError("boom")

    f.render_typing = _boom
    f.on_bus_session_turn_started(_started("mine"))  # must not raise
