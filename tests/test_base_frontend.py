"""Tests for BaseFrontend's AGENT_TEXT_DELTA handling and whole-message dedup.

A frontend that rendered a stream's deltas must not re-print the identical
whole message when it arrives via RuntimeResult (final answers) or
CHAT_MESSAGE_PUSHED (mid-turn narration). Frontends without
``supports_streaming`` ignore the channel entirely.
"""

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from runtime.session import RuntimeResult


class _CaptureFrontend(BaseFrontend):
    name = "cap"
    capabilities = FrontendCapabilities(supports_streaming=True)

    def __init__(self):
        super().__init__()
        self.rendered: list[str] = []
        self.stream_events: list[dict] = []

    def render_messages(self, _key, messages):
        self.rendered.extend(messages)

    def render_stream_delta(self, _key, payload):
        self.stream_events.append(payload)

    def _live_session_keys(self):
        return ["s"]

    def _current_approval_request(self, _key):
        return None  # unbound test frontend has no runtime to consult


def _delta(seq, text, stream_id="st1"):
    return {"session_key": "s", "stream_id": stream_id, "seq": seq,
            "delta": text, "done": False, "aborted": False}


def _done(seq, final_text, kind="final", aborted=False, stream_id="st1"):
    payload = {"session_key": "s", "stream_id": stream_id, "seq": seq,
               "delta": "", "done": True, "aborted": aborted}
    if not aborted:
        payload["final_text"] = final_text
        payload["kind"] = kind
    return payload


def _stream(frontend, final_text, **kwargs):
    frontend.on_bus_agent_text_delta(_delta(1, final_text[:3], **kwargs))
    frontend.on_bus_agent_text_delta(_delta(2, final_text[3:], **kwargs))
    frontend.on_bus_agent_text_delta(_done(3, final_text, **kwargs))


def test_streamed_final_suppresses_duplicate_whole_message():
    f = _CaptureFrontend()
    _stream(f, "Hello there")

    assert len(f.stream_events) == 3
    f._render_result("s", RuntimeResult(messages=["Hello there"]))
    assert f.rendered == []  # already rendered as deltas

    # The dedup entry is consumed: the same text later renders normally.
    f._render_result("s", RuntimeResult(messages=["Hello there"]))
    assert f.rendered == ["Hello there"]


def test_streamed_narration_suppresses_duplicate_push():
    f = _CaptureFrontend()
    f.on_bus_agent_text_delta(_delta(1, "checking files"))
    f.on_bus_agent_text_delta(_done(2, "checking files", kind="narration"))

    f.on_bus_message_pushed({"session_key": "s", "message": "checking files"})
    assert f.rendered == []

    f.on_bus_message_pushed({"session_key": "s", "message": "checking files"})
    assert f.rendered == ["checking files"]


def test_non_matching_message_still_renders():
    f = _CaptureFrontend()
    _stream(f, "Hello there")

    f._render_result("s", RuntimeResult(messages=["Something else"]))
    assert f.rendered == ["Something else"]


def test_aborted_stream_records_no_dedup_entry():
    f = _CaptureFrontend()
    f.on_bus_agent_text_delta(_delta(1, "par"))
    f.on_bus_agent_text_delta(_done(2, None, aborted=True))

    # The retry/cancel whole message renders normally.
    f._render_result("s", RuntimeResult(messages=["par tial"]))
    assert f.rendered == ["par tial"]


def test_done_without_prior_deltas_is_ignored():
    f = _CaptureFrontend()
    f.on_bus_agent_text_delta(_done(1, "Hello"))

    assert f.stream_events == []
    f._render_result("s", RuntimeResult(messages=["Hello"]))
    assert f.rendered == ["Hello"]


def test_foreign_session_is_ignored():
    f = _CaptureFrontend()
    f.on_bus_agent_text_delta(_delta(1, "abc", stream_id="st9") | {"session_key": "other"})

    assert f.stream_events == []


def test_non_streaming_frontend_ignores_channel():
    class _Plain(_CaptureFrontend):
        capabilities = FrontendCapabilities()

    f = _Plain()
    _stream(f, "Hello there")

    assert f.stream_events == []
    f._render_result("s", RuntimeResult(messages=["Hello there"]))
    assert f.rendered == ["Hello there"]


# ────────────────────────────────────────────────────────────────────
# Typing indicators (was test_frontend_typing.py)
# ────────────────────────────────────────────────────────────────────

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


# ── Untargeted broadcasts land on one surface once ────────────────────
#
# With two frontends running, a system announcement (a plugin registration,
# say) carries no session_key. It used to fan out across every session the
# frontend considered live — which includes sessions no frontend has claimed
# yet — so the REPL printed "Registered plugin: x" once for its own session
# and again for Telegram's untagged one. Two lines, one transport, one event.

class _BroadcastFrontend(BaseFrontend):
    name = "repl"
    capabilities = FrontendCapabilities()

    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime
        self.rendered: list[str] = []

    def render_messages(self, _key, messages):
        self.rendered.extend(messages)

    def _live_session_keys(self):
        # What the sandboxed adapter answers: own sessions plus unclaimed ones.
        return [key for key, session in self.runtime.sessions.items()
                if session.frontend_name in (None, self.name)]

    def _current_approval_request(self, _key):
        return None


class _Session:
    def __init__(self, frontend_name):
        self.frontend_name = frontend_name


class _Runtime:
    def __init__(self, sessions):
        self.sessions = sessions


def _push(frontend, message="Registered plugin: memory"):
    frontend.on_bus_message_pushed({"message": message, "kind": "plugin"})


def test_a_broadcast_renders_once_when_another_frontend_is_untagged():
    """The reported bug: repl + telegram running, one notice printed twice."""
    runtime = _Runtime({"repl:1": _Session("repl"), "tg:2": _Session(None)})
    frontend = _BroadcastFrontend(runtime)

    _push(frontend)

    assert frontend.rendered == ["Registered plugin: memory"]


def test_a_broadcast_still_reaches_every_session_this_frontend_owns():
    """Narrowing to owned sessions must not narrow to *one* of them."""
    runtime = _Runtime({"a": _Session("repl"), "b": _Session("repl")})
    frontend = _BroadcastFrontend(runtime)

    _push(frontend)

    assert frontend.rendered == ["Registered plugin: memory"] * 2


def test_a_broadcast_is_not_swallowed_before_any_session_is_tagged():
    """A fresh install has submitted nothing, so nothing carries an owner yet.

    Falling back to the live set keeps announcements visible; the alternative
    is a first-run where every registration notice silently goes nowhere.
    """
    runtime = _Runtime({"only": _Session(None)})
    frontend = _BroadcastFrontend(runtime)

    _push(frontend)

    assert frontend.rendered == ["Registered plugin: memory"]


def test_a_targeted_message_still_reaches_an_unclaimed_session():
    """Ownership scoping applies to broadcasts only.

    A targeted render names its session, and the first message of a
    conversation arrives before anything has tagged it.
    """
    runtime = _Runtime({"repl:1": _Session("repl"), "new": _Session(None)})
    frontend = _BroadcastFrontend(runtime)

    frontend.on_bus_message_pushed({"message": "hello", "session_key": "new"})

    assert frontend.rendered == ["hello"]
