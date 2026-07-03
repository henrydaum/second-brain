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
