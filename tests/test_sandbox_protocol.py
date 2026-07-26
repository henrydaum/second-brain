"""The wire protocol, tested on its own before anything is built on it."""

import io
import json

import pytest

from sandbox.guest import protocol


def _pipe(*messages) -> io.BytesIO:
    """A readable stream preloaded with encoded messages."""
    return io.BytesIO(b"".join(protocol.encode(m) for m in messages))


def test_round_trip():
    """A message survives the wire unchanged."""
    stream = _pipe({"kind": protocol.REQUEST,
                    "request": {"type": "fs.read", "args": {"path": "a.txt"}}})
    message = protocol.read_message(stream)
    assert message["request"]["args"]["path"] == "a.txt"


def test_messages_are_read_one_at_a_time():
    """Several messages queued in the buffer are not conflated."""
    stream = _pipe({"kind": protocol.LOG, "message": "first"},
                   {"kind": protocol.LOG, "message": "second"})
    assert protocol.read_message(stream)["message"] == "first"
    assert protocol.read_message(stream)["message"] == "second"
    assert protocol.read_message(stream) is None


def test_embedded_newlines_do_not_split_a_message():
    """The reason newline framing is safe: JSON escapes them."""
    payload = "line one\nline two\r\nline three"
    stream = _pipe({"kind": protocol.LOG, "message": payload})
    assert protocol.read_message(stream)["message"] == payload
    assert protocol.read_message(stream) is None


def test_unicode_survives():
    """Non-ASCII content crosses intact."""
    stream = _pipe({"kind": protocol.LOG, "message": "café — \U0001f9e0"})
    assert protocol.read_message(stream)["message"] == "café — \U0001f9e0"


def test_end_of_stream_is_not_an_error():
    """A child that exited or was killed simply ends the stream."""
    assert protocol.read_message(io.BytesIO(b"")) is None


def test_garbage_is_rejected_clearly():
    """Corruption fails loudly rather than being half-parsed."""
    with pytest.raises(protocol.ProtocolError):
        protocol.read_message(io.BytesIO(b"not json at all\n"))


def test_unkinded_messages_are_rejected():
    """Every message must say what it is."""
    with pytest.raises(protocol.ProtocolError):
        protocol.read_message(io.BytesIO(json.dumps({"a": 1}).encode() + b"\n"))


def test_oversized_messages_are_refused_on_write():
    """A runaway child cannot exhaust the parent by describing something huge."""
    with pytest.raises(protocol.ProtocolError):
        protocol.encode({"kind": protocol.LOG,
                         "message": "x" * (protocol.MAX_MESSAGE_BYTES + 1)})


def test_write_flushes():
    """Buffered writes deadlock two processes that each wait on the other."""
    flushed = []

    class Stream(io.BytesIO):
        """Records flushes."""
        def flush(self):
            """Note the flush."""
            flushed.append(True)

    protocol.write_message(Stream(), {"kind": protocol.LOG, "message": "x"})
    assert flushed


# ──────────────────────────────────────────────────────────────────────
# What may cross.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    None, True, 3, 1.5, "text", [1, "two"], {"a": [1, {"b": None}]},
])
def test_simple_values_may_cross(value):
    """JSON-native data crosses."""
    assert protocol.is_simple(value)


@pytest.mark.parametrize("value", [
    object(), {1: "non-string key"}, [object()], {"f": len}, {"s": {1, 2}},
])
def test_live_objects_may_not_cross(value):
    """No live object, and therefore no route back into the kernel."""
    assert not protocol.is_simple(value)
