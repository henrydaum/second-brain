"""Tests for the agent-turn lifecycle bus events.

``_drive_agent_turn`` is the single site every agent turn flows through
(foreground ``handle_action`` and background ``iterate_agent_turn`` alike),
so SESSION_TURN_STARTED / SESSION_TURN_COMPLETED are emitted there — plus the
agent-side SESSION_TURN_CHANGED that mirrors what user actions already get
from the dispatch layer.
"""

from types import SimpleNamespace

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from events.event_bus import bus
from events.event_channels import (
    SESSION_TURN_CHANGED,
    SESSION_TURN_COMPLETED,
    SESSION_TURN_STARTED,
)
from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime


def _response(content="", tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        is_error=False,
        prompt_tokens=0,
    )


class _FakeLLM:
    context_size = 0
    is_llm_backend = True
    model_name = "fake"
    loaded = True

    def __init__(self, responses=None):
        self._responses = list(responses or [])

    def chat_with_tools(self, messages, tools, attachments=None):
        return self._responses.pop(0) if self._responses else _response(content="done.")


def _runtime(tmp_path, responses=None):
    db = Database(str(tmp_path / "events.db"))
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db, services={"llm": _FakeLLM(responses)}, config={})
    session = rt.load_conversation("s", cid)
    return rt, session


def _capture(*channels):
    seen = []
    unsubs = [bus.subscribe(ch, (lambda c: lambda p: seen.append((c, p)))(ch)) for ch in channels]
    return seen, unsubs


def test_turn_started_and_completed_bracket_a_foreground_turn(tmp_path):
    rt, session = _runtime(tmp_path, [_response(content="Hi there.")])
    seen, unsubs = _capture(SESSION_TURN_STARTED, SESSION_TURN_COMPLETED, SESSION_TURN_CHANGED)
    try:
        out = rt.handle_action("s", "send_text", "hello")
    finally:
        for u in unsubs:
            u()

    assert out.ok
    kinds = [c for c, _ in seen]
    assert kinds.index(SESSION_TURN_STARTED) < kinds.index(SESSION_TURN_COMPLETED)

    started = next(p for c, p in seen if c == SESSION_TURN_STARTED)
    assert started["session_key"] == "s"
    assert started["conversation_id"] == session.conversation_id
    assert started["actor_id"] == "agent"

    completed = next(p for c, p in seen if c == SESSION_TURN_COMPLETED)
    assert completed["ok"] is True
    assert completed["cancelled"] is False
    assert completed["final_text"] == "Hi there."
    assert any(m.get("content") == "Hi there." for m in completed["new_messages"])

    # The agent's end_turn hand-back is broadcast, mirroring the user side.
    turn_changes = [p for c, p in seen if c == SESSION_TURN_CHANGED]
    assert {"session_key": "s", "from_actor": "agent", "to_actor": "user"} in turn_changes


def test_crashed_drive_completes_the_turn_with_ok_false(tmp_path):
    rt, session = _runtime(tmp_path)

    import runtime.conversation_runtime as _crt

    def exploding_build_loop(runtime, session_key=None):
        raise RuntimeError("boom")

    seen, unsubs = _capture(SESSION_TURN_STARTED, SESSION_TURN_COMPLETED)
    original = _crt._cfg.build_loop
    _crt._cfg.build_loop = exploding_build_loop
    try:
        out = rt.handle_action("s", "send_text", "hello")
    finally:
        _crt._cfg.build_loop = original
        for u in unsubs:
            u()

    assert not out.ok
    completed = next(p for c, p in seen if c == SESSION_TURN_COMPLETED)
    assert completed["ok"] is False
    assert "boom" in completed["error"]
    # Every started turn still completed — no dangling busy indicator.
    assert len([c for c, _ in seen if c == SESSION_TURN_STARTED]) == \
        len([c for c, _ in seen if c == SESSION_TURN_COMPLETED])
