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

Two optional fields carry the multi-occupant case, and both are absent for
every box that has one occupant — which is nearly all of them. ``start`` may
name ``entries``, several plugin classes to instantiate from one module
import; ``call`` may then name a ``target`` saying which of them the call is
for. A ``call`` with no ``target`` resolves to the sole occupant, so the wire
a single-service box speaks is byte-identical to what it always was.

``start`` may also carry ``parsers``: files to import into the box before its
entry, resolved by the host from the plugin's declared ``parse_modalities``.
They are sent as *paths* rather than named by modality because the child has
no registry to resolve a modality against — deciding which files provide
"image" is the kernel's standing knowledge, and this is the answer, not the
question.

Child to parent:

- ``request`` — I want an effect; classify it
- ``notice``  — the same, but I am not waiting for the answer
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

**A notice is a Request the child does not wait for.** It still passes the
same gate — classified, recorded, executed by the same handler — so it is not
a way around policy; the only thing given up is the answer. (Why streaming
needs that is in CLAUDE.md, "Streaming inverted, and lost a feature on
purpose".) Because nothing is awaited, a notice never appears in the
``expected`` set of a pump loop: it is serviced and the loop reads on.
"""

from __future__ import annotations

import base64
import json
from typing import Any

# Parent -> child.
START = "start"
RESULT = "result"
CALL = "call"      # persistent boxes: invoke a method and wait for RETURN
STOP = "stop"      # persistent boxes: shut down gracefully

# Child -> parent.
REQUEST = "request"
NOTICE = "notice"  # a Request wanting no answer; the child does not wait
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


class StreamClosed(ProtocolError):
    """The pipe itself is gone — the other side exited.

    Distinct from an unusable *message*, because the responses are opposite:
    a bad message is worth reporting down the wire, and a dead wire is worth
    nothing but a quiet exit. During shutdown the parent closes the pipes and
    the child is mid-write; the failure that surfaces there is an ordinary
    ``OSError`` (``EPIPE`` on POSIX, ``EINVAL`` on Windows), which is not a
    fault and must not be printed as a traceback.
    """


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

    Encoding is checked before the stream is touched, so a message that will
    not fit raises ``ProtocolError`` with the wire still intact and the caller
    free to report it. Only the write itself can raise ``StreamClosed``.
    """
    raw = encode(message)
    try:
        stream.write(raw)
        stream.flush()
    except (BrokenPipeError, OSError, ValueError) as exc:
        # ValueError is the "I/O operation on closed file" case; a plain
        # OSError with EINVAL is what Windows reports for the same thing.
        raise StreamClosed(f"channel closed: {exc}") from None


def read_message(stream) -> dict | None:
    """Read one message. Returns None at end of stream.

    End of stream is not an error — it is how a child that exited cleanly, or
    was killed, reports itself. The caller decides what that means.
    """
    # Bounded, because an unbounded ``readline`` would buffer whatever the
    # other side sent *before* the size check could refuse it — which made the
    # cap above describe a protection it did not provide. One extra byte, so a
    # message exactly at the limit still reads its terminator.
    try:
        line = stream.readline(MAX_MESSAGE_BYTES + 1)
    except (OSError, ValueError):
        # The pipe was closed under us. Indistinguishable, from here, from the
        # other side having hung up cleanly — and the caller handles that.
        return None
    if not line:
        return None
    if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
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

    Only JSON-native data, plus ``bytes`` — which are not JSON-native but are
    packed into it by :func:`pack` on the way out and restored on the way in,
    so from a caller's point of view they cross like anything else.

    A dataclass is converted to a dictionary by its sender and rebuilt by the
    SDK on the far side, so no live object — and therefore no route back into
    the kernel's module graph — ever crosses.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_simple(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_simple(v)
                   for k, v in value.items())
    return False


# ──────────────────────────────────────────────────────────────────────
# Bytes over a JSON wire.
# ──────────────────────────────────────────────────────────────────────
#
# JSON has no bytes type, so a value that is merely *numeric* — an embedding
# vector, a thumbnail, a BLOB column — could not cross. It failed in the worst
# possible direction: in-process there is no serialization at all, so a plugin
# writing a BLOB worked on a thread and raised ``TypeError`` deep inside
# ``json.dumps`` only once it ran in a subprocess. Identical code, two
# behaviours, and the pipe is where the failure shows up.
#
# The codec lives here rather than in the database handler because there is
# nothing database-shaped about it. Every payload that crosses funnels through
# ``Request.args`` and ``Result.data``, so packing at those two points covers
# db params, db rows, ``service.call`` arguments and ``service.call`` return
# values in one place — and leaves every handler written as if bytes were
# ordinary, which is the point.
#
# ``fs.read_bytes`` predates this and base64s by hand at the SDK level. It is
# left alone: its encoding is part of that Request's documented answer, and
# rewriting it would change a wire format for no behavioural gain.

#: The single key marking a packed bytes value. A dict is only ever decoded
#: when it has this key *and nothing else*, so plugin data that happens to
#: contain the string is not mistaken for an encoding.
BYTES_TAG = "__bytes__"
#: Escape envelope for a user dictionary that would otherwise look like one
#: of the codec's own envelopes.  Without it, ``{"__bytes__": "AA=="}``
#: silently became ``b"\x00"`` in a subprocess while remaining a dictionary
#: in-process.
DICT_TAG = "__dict__"


def pack_simple(value: Any) -> Any:
    """Validate and pack one value that is allowed across the boundary.

    Keeping this beside :func:`is_simple` makes the rule executable rather
    than advisory.  Every Request and Result passes through here, including
    the in-process transport, so a live object cannot become an accidental
    back-reference into the kernel and runner choice cannot change behaviour.
    """
    if not is_simple(value):
        raise ProtocolError(
            "payload contains a live or non-serializable Python object")
    return pack(value)


def normalize(value: Any) -> Any:
    """Return the canonical value the JSON transport would deliver.

    Tuples become lists and bytes-like objects become ``bytes``.  Applying the
    same normalization in-process is what makes handlers see the same types
    under both runners.
    """
    return unpack(pack_simple(value))


def pack(value: Any) -> Any:
    """Encode any ``bytes`` inside a payload for the JSON wire.

    Recurses through lists and dicts, since the interesting cases are both
    nested: a list of embedding vectors, a row dict with one BLOB column.
    Everything else is returned unchanged, so packing a payload with no bytes
    in it costs a walk and produces an equal value.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {BYTES_TAG: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [pack(item) for item in value]
    if isinstance(value, dict):
        packed = {key: pack(item) for key, item in value.items()}
        # A bytes envelope is intentionally a lone reserved key.  Escape a
        # user's dictionary with that shape (and the escape shape itself) so
        # unpack can distinguish data from an envelope without changing the
        # established encoding for real bytes.
        if len(packed) == 1 and (
                BYTES_TAG in packed or DICT_TAG in packed):
            return {DICT_TAG: [[key, item] for key, item in packed.items()]}
        return packed
    return value


def unpack(value: Any) -> Any:
    """Restore what :func:`pack` encoded.

    Undecodable content is passed through rather than raised on: a malformed
    tag is data somebody sent, not a protocol violation, and the honest answer
    is to hand back the dict as it arrived.
    """
    if isinstance(value, dict):
        if len(value) == 1 and isinstance(value.get(DICT_TAG), list):
            pairs = value[DICT_TAG]
            if all(isinstance(pair, list) and len(pair) == 2
                   and isinstance(pair[0], str) for pair in pairs):
                return {pair[0]: unpack(pair[1]) for pair in pairs}
        if len(value) == 1 and isinstance(value.get(BYTES_TAG), str):
            try:
                return base64.b64decode(value[BYTES_TAG], validate=True)
            except (ValueError, TypeError):
                return value
        return {key: unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack(item) for item in value]
    return value
