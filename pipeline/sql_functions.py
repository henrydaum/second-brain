"""Scalar functions the kernel's SQLite connection offers to every query.

A plugin reaches the database through ``sdk.db.query``, which answers with
rows over the wire and caps at ``DB_MAX_ROWS``. That cap is not a limitation
to route around — it is the statement that **the answer is what crosses, not
the data**. Anything whose answer is small has to say so in SQL.

Vector search had no way to say so. Cosine similarity over a BLOB column is
not expressible in SQLite, so the only way to rank embeddings was to read
every one of them into the asking box and do the arithmetic there: measured at
100k chunks of 384 dimensions, 214 MB of JSON across 200 gate-serialized round
trips, per query, to return five rows.

The precedent for the fix is already in the schema. ``task_lexical_index``
scans a hundred thousand documents and five rows cross, because FTS5 puts the
index *in the database* and SQL expresses the reduction. The vector case is
the same shape and was only missing the operator, so this module supplies the
operator rather than a search.

That distinction is the whole of why this belongs in the kernel. It is not a
search: it knows nothing about embeddings, models, streams or top-k, and it
composes with ``WHERE``, ``JOIN``, ``ORDER BY`` and ``LIMIT`` because it is an
expression. A ``vector.search`` Request would have had to take a table name, a
column, a filter and a limit — which is to say it would have reimplemented SQL,
badly, and that is the tell that the Request was the wrong shape. Registering a
deterministic scalar function is configuring the database the kernel already
owns, the same category of act as the PRAGMAs beside it.

**numpy is used when present and never required.** The stdlib path is the
contract and the numpy path must agree with it; a package that pulled numpy in
(``service_embed`` does) installs it into the app's own interpreter, so the
acceleration arrives on its own for the users who have the vectors in the first
place. SQLite's own per-call overhead is ~0.06s per 100k rows, so the whole of
the stdlib path's cost is Python arithmetic and there is real headroom to win.
"""

from __future__ import annotations

import logging
import math
import struct

logger = logging.getLogger("SQLFunctions")

try:                                            # optional accelerator
    import numpy as _np
    _F32 = _np.dtype("<f4")
except Exception:                               # noqa: BLE001 - absence is normal
    _np = None
    _F32 = None

#: float32, little-endian — what ``numpy.float32.tobytes()`` writes on every
#: platform this runs on, spelled explicitly so the two paths cannot disagree
#: about byte order on one that does not.
_ITEM = 4
_MUL = float.__mul__


class _OneSlot:
    """The last vector unpacked, with its norm.

    One slot rather than a dict, and that is a correctness property rather
    than a size choice. Within a single statement the second argument is the
    *same* blob for every row, so one slot hits on all but the first; across
    statements it is simply replaced. A growing cache keyed on query vectors
    would be an unbounded leak that only a heavy user would ever notice.
    """

    __slots__ = ("key", "value", "_filled")

    def __init__(self):
        self.key = None
        self.value = None
        self._filled = False

    def resolve(self, blob, decode):
        """The decode of ``blob``, computing it at most once per distinct blob.

        Remembering a *failed* decode matters as much as a successful one: a
        malformed query vector is malformed for every row in the scan, and
        recomputing the failure each time would make the worst case the
        slowest one.
        """
        if self._filled and self.key == blob:
            return self.value
        self.key, self.value, self._filled = blob, decode(blob), True
        return self.value


def _vector_bytes(value):
    """``value`` as blob bytes if it could be a float32 vector, else None."""
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return None                             # a TEXT or numeric column
    raw = bytes(value)
    if not raw or len(raw) % _ITEM:
        return None
    return raw


def _decode_stdlib(blob):
    """A blob as (floats, norm)."""
    values = struct.unpack(f"<{len(blob) // _ITEM}f", blob)
    return values, math.sqrt(sum(map(_MUL, values, values)))


def _make_cosine_stdlib():
    """Cosine similarity in pure Python."""
    cache = _OneSlot()

    def vec_cosine(left, right):
        """Cosine similarity of two float32 vectors, or NULL."""
        mine = _vector_bytes(left)
        other = _vector_bytes(right)
        if mine is None or other is None or len(mine) != len(other):
            return None
        theirs, norm_theirs = cache.resolve(other, _decode_stdlib)
        values, norm_mine = _decode_stdlib(mine)
        product = norm_theirs * norm_mine
        if not product:
            return None
        return sum(map(_MUL, values, theirs)) / product

    return vec_cosine


def _decode_numpy(blob):
    """A blob as (array, norm)."""
    vector = _np.frombuffer(blob, dtype=_F32)
    return vector, float(_np.linalg.norm(vector))


def _make_cosine_numpy():
    """The same function, over numpy buffers."""
    cache = _OneSlot()

    def vec_cosine(left, right):
        """Cosine similarity of two float32 vectors, or NULL."""
        mine = _vector_bytes(left)
        other = _vector_bytes(right)
        if mine is None or other is None or len(mine) != len(other):
            return None
        theirs, norm_theirs = cache.resolve(other, _decode_numpy)
        values, norm_mine = _decode_numpy(mine)
        product = norm_theirs * norm_mine
        if not product:
            return None
        return float(values @ theirs) / product

    return vec_cosine


def register(conn) -> None:
    """Install every kernel SQL function on a connection.

    Failure is logged and swallowed: a database that cannot take a scalar
    function still works for everything that does not use one, and refusing to
    open it would turn an optional capability into a boot failure.
    """
    try:
        maker = _make_cosine_numpy if _np is not None else _make_cosine_stdlib
        # ``deterministic`` lets SQLite hoist and reuse the call; the function
        # is a pure decode plus arithmetic, so the claim is true.
        conn.create_function("vec_cosine", 2, maker(), deterministic=True)
    except Exception:                           # noqa: BLE001 - optional
        logger.exception("could not register SQL functions")
