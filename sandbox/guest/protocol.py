"""The wire between the kernel and a subprocessed sandbox.

Newline-delimited JSON over a pair of pipes. Deliberately boring: no pickle
(which would execute code on unpickle, defeating the point), no framing
cleverness, nothing that cannot be read in a hex dump while debugging.

JSON escapes newlines inside strings, so a bare ``\\n`` is an unambiguous
message terminator.

**stdout is not the wire.** Plugin code prints, and so do libraries it
imports; if that landed in the protocol stream it would corrupt the channel in
a way that looks like a parser bug. The child redirects its own ``sys.stdout``
to stderr at startup and talks over dedicated pipes instead.

Message kinds, parent to child:

- ``start``   — load this, with these kwargs
- ``result``  — the answer to the Request you just made
- ``call``    — persistent only: run this method and answer with ``return``
- ``stop``    — persistent only: shut down gracefully

Child to parent:

- ``request`` — I want an effect; classify it
- ``log``     — write this to the kernel's log sink
- ``done``    — ephemeral only: I finished, here is my Result
- ``ready``   — persistent only: loaded and standing by
- ``return``  — persistent only: the answer to one ``call``
- ``fault``   — I broke in a way I could not report as a Result

**Direction flips for persistent boxes.** An ephemeral child always speaks
first: it makes Requests until it is done. A resident one is the mirror image
— the parent calls in, and the child answers. Between the two it is idle,
blocked on a read, executing nothing.

While serving a ``call`` the child may still send Requests of its own, so the
parent's loop expects *either* a ``request`` or the ``return``. That needs no
message ids because a box serves one call at a time; concurrency here would
require them.
"""

from __future__ import annotations

import json
from typing import Any

# Parent -> child.
START = "start"
RESULT = "result"
CALL = "call"      # persistent boxes: invoke a method and wait for RETURN
STOP = "stop"      # persistent boxes: shut down gracefully

# Child -> parent.
REQUEST = "request"
LOG = "log"
DONE = "done"      # ephemeral boxes: the one and only result
READY = "ready"    # persistent boxes: loaded, standing by for calls
RETURN = "return"  # persistent boxes: the answer to one CALL
FAULT = "fault"

# A single message may not exceed this, so a runaway child cannot exhaust the
# parent's memory by describing something enormous.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class ProtocolError(Exception):
    """The channel carried something unusable."""


def encode(message: dict) -> bytes:
    """Serialize one message for the wire."""
    raw = json.dumps(message, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    return raw + b"\n"


def write_message(stream, message: dict) -> None:
    """Write one message and flush.

    Flushing every message is the point: both sides block waiting for the
    other, so a buffered write is a deadlock.
    """
    stream.write(encode(message))
    stream.flush()


def read_message(stream) -> dict | None:
    """Read one message. Returns None at end of stream.

    End of stream is not an error — it is how a child that exited cleanly, or
    was killed, reports itself. The caller decides what that means.
    """
    line = stream.readline()
    if not line:
        return None
    if len(line) > MAX_MESSAGE_BYTES:
        raise ProtocolError("oversized message")
    try:
        message = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"unparseable message: {exc}") from None
    if not isinstance(message, dict) or "kind" not in message:
        raise ProtocolError("message is not a kinded object")
    return message


def is_simple(value: Any) -> bool:
    """Whether a value may cross the boundary.

    Only JSON-native data. A dataclass is converted to a dictionary by its
    sender and rebuilt by the SDK on the far side, so no live object — and
    therefore no route back into the kernel's module graph — ever crosses.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_simple(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_simple(v)
                   for k, v in value.items())
    return False
