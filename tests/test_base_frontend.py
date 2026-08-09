"""Tests for BaseFrontend's AGENT_TEXT_DELTA handling and whole-message dedup.

A frontend that rendered a stream's deltas must not re-print the identical
whole message when it arrives via RuntimeResult (final answers) or
CHAT_MESSAGE_PUSHED (mid-turn narration). Frontends without
``supports_streaming`` ignore the channel entirely.
"""

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from types import SimpleNamespace

from plugins.native.frontend import BaseFrontend, FrontendCapabilities
from runtime.session import RuntimeResult
from state_machine.approval import StateMachineApprovalRequest
from state_machine.conversation_phases import BASE_PHASE, PHASE_APPROVING_REQUEST


class _CaptureFrontend(BaseFrontend):
    name = "cap"
    capabilities = FrontendCapabilities(supports_streaming=True)

    def __init__(self):
        super().__init__()
        self.rendered: list[str] = []
        self.files: list[str] = []
        self.stream_events: list[dict] = []

    def render_messages(self, _key, messages):
        self.rendered.extend(messages)

    def render_attachments(self, _key, paths):
        self.files.extend(paths)

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


# Action-originated approvals (commands with require_approval=True) return on
# RuntimeResult instead of entering through the APPROVAL_REQUESTED bus.
class _ApprovalFrontend(BaseFrontend):
    name = "approval-test"
    capabilities = FrontendCapabilities(supports_buttons=True)

    def __init__(self):
        super().__init__()
        frame = SimpleNamespace(
            phase=PHASE_APPROVING_REQUEST,
            name="packages",
            step=None,
            data={"title": "packages", "prompt": "Install a package?", "type": "boolean"},
        )
        self.session = SimpleNamespace(
            cs=SimpleNamespace(phase=PHASE_APPROVING_REQUEST, frame=frame),
            conversation_id=7,
            frontend_name=self.name,
            user_id=1,
        )
        self.runtime = SimpleNamespace(get_session=lambda _key: self.session)
        self.approvals = []
        self.messages = []
        self.errors = []

    def render_approval_request(self, _key, req):
        self.approvals.append(req)

    def render_messages(self, _key, messages):
        self.messages.extend(messages)

    def render_error(self, _key, error):
        self.errors.append(error)


def test_result_approval_gets_a_stable_registered_request_id():
    frontend = _ApprovalFrontend()

    frontend._render_result("s", RuntimeResult(messages=["Approval required."]))
    frontend._render_result("s", RuntimeResult())

    request_id = frontend.session.cs.frame.data["request_id"]
    assert request_id.startswith("approve_")
    assert frontend.is_approval_pending("s", request_id)
    assert frontend._pending_approval_order["s"] == [request_id]
    assert [req.id for req in frontend.approvals] == [request_id, request_id]


def test_invalid_typed_approval_keeps_the_request_pending():
    frontend = _ApprovalFrontend()
    frontend._render_result("s", RuntimeResult())
    request_id = frontend.approvals[-1].id
    frontend.runtime.handle_action = lambda *_args: RuntimeResult(
        False, error={"code": "invalid_input", "message": "Approval needs yes or no."})

    result = frontend.submit_text("s", "Hey")

    assert not result.ok
    assert frontend.is_approval_pending("s", request_id)


def test_rich_approval_renders_the_approved_actions_result():
    frontend = _ApprovalFrontend()
    frontend._render_result("s", RuntimeResult())
    request_id = frontend.approvals[-1].id

    def handle(_key, action, payload):
        assert action == "answer_approval"
        assert payload == {"value": True, "request_id": request_id}
        frontend.session.cs.phase = BASE_PHASE
        frontend.session.cs.frame = None
        return RuntimeResult(True, messages=["Installed."])

    frontend.runtime.handle_action = handle

    assert frontend.resolve_approval("s", request_id, True)
    assert frontend.messages == ["Installed."]
    assert not frontend.is_approval_pending("s", request_id)


def test_bus_approval_does_not_render_while_original_call_is_blocked():
    frontend = _ApprovalFrontend()
    request = StateMachineApprovalRequest(
        title="Delete?", body="Delete conversation?", id="approve_policy",
        metadata={"session_key": "s"})
    frontend._register_pending_approval("s", request)

    def handle(_key, _action, _payload):
        frontend.session.cs.phase = BASE_PHASE
        frontend.session.cs.frame = None
        return RuntimeResult(True, messages=["Must be returned by the original call."])

    frontend.runtime.handle_action = handle

    assert frontend.resolve_approval("s", request.id, True)
    assert frontend.messages == []


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


# ────────────────────────────────────────────────────────────────────
# Pushed attachments — the only outbound file route a non-tool has
# ────────────────────────────────────────────────────────────────────

def test_a_push_carries_its_files_to_render_attachments():
    frontend = _CaptureFrontend()

    frontend.on_bus_message_pushed({"session_key": "s", "message": "here it is",
                                    "attachments": ["/tmp/a.png"]})

    assert frontend.rendered == ["here it is"]
    assert frontend.files == ["/tmp/a.png"]


def test_a_push_of_files_alone_is_not_dropped():
    """``sdk.ui.render`` with no caption sends files and no words.

    The handler returned early on a falsy ``message``, so the one call whose
    entire purpose is the files was the one that rendered nothing.
    """
    frontend = _CaptureFrontend()

    frontend.on_bus_message_pushed({"session_key": "s",
                                    "attachments": ["/tmp/a.png"]})

    assert frontend.rendered == []
    assert frontend.files == ["/tmp/a.png"]


def test_an_already_streamed_body_still_delivers_its_files():
    """Dedup suppresses the text it saw stream past, never the attachments."""
    frontend = _CaptureFrontend()
    _stream(frontend, "checking files")

    frontend.on_bus_message_pushed({"session_key": "s",
                                    "message": "checking files",
                                    "attachments": ["/tmp/a.png"]})

    # The body already went out as deltas, so the push does not repeat it.
    assert frontend.rendered == []
    assert frontend.files == ["/tmp/a.png"]


def test_an_empty_push_still_does_nothing():
    frontend = _CaptureFrontend()

    frontend.on_bus_message_pushed({"session_key": "s"})

    assert frontend.rendered == [] and frontend.files == []
