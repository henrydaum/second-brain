"""Fixture: SDK code that ranks vectors without ever holding the corpus.

This is the shape ``tool_semantic_search`` uses, reduced to the part the
kernel has to make possible. A thousand vectors go into a table; the query
vector crosses as a bound parameter; ``vec_cosine`` runs kernel-side inside
the SELECT; three rows come back. Nothing here would work if the ranking had
to happen on this side of the boundary — ``db.query`` caps its answer, and
that cap is the point rather than an obstacle.

Run in-process *and* in a subprocess from one file, so a difference between
the two is a difference in the boundary.
"""

import math
import struct

TABLE = "sandbox_vector_probe"


def _unit(*values) -> bytes:
    """A normalized float32 blob, the way an embedder writes one."""
    norm = math.sqrt(sum(v * v for v in values))
    return struct.pack(f"<{len(values)}f", *[v / norm for v in values])


def _ensure_corpus(sdk, count=1000):
    """A table of vectors sweeping one quadrant, nearest-first by index."""
    sdk.db.define(
        f"CREATE TABLE IF NOT EXISTS {TABLE} ("
        "  path TEXT, chunk_index INTEGER, model_name TEXT, embedding BLOB"
        ")"
    )
    if sdk.db.query(f"SELECT count(*) AS n FROM {TABLE}")[0]["n"]:
        return
    for i in range(count):
        angle = i * (math.pi / (2 * count))
        sdk.db.write(
            f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?)",
            [f"doc{i}.txt", i, "probe-model",
             _unit(math.cos(angle), math.sin(angle))])


def rank_by_similarity(sdk, top_k=3):
    """The whole search as one statement — the answer crosses, not the data."""
    _ensure_corpus(sdk)
    query = _unit(1.0, 0.0)
    rows = sdk.db.query(
        f"SELECT path, chunk_index, vec_cosine(embedding, ?) AS score "
        f"FROM {TABLE} WHERE model_name = ? AND length(embedding) = ? "
        f"ORDER BY score DESC LIMIT ?",
        [query, "probe-model", len(query), top_k])
    return {"count": len(rows),
            "paths": [row["path"] for row in rows],
            "scores": [round(row["score"], 6) for row in rows]}


def stale_dimensions_do_not_break_the_search(sdk):
    """A vector from a different model must drop out, not raise.

    ``length(embedding) = ?`` filters it before the arithmetic; ``vec_cosine``
    answering NULL is the backstop if a caller forgets the filter. Both are
    exercised here because getting either wrong fails the *whole statement*,
    which would turn one leftover row into a search that never works again.
    """
    _ensure_corpus(sdk)
    # Idempotent: both runners share one database, and a second stale row
    # would make the count assertion pass for whichever ran first.
    sdk.db.write(f"DELETE FROM {TABLE} WHERE path = ?", ["stale.txt"])
    sdk.db.write(f"INSERT INTO {TABLE} VALUES (?, ?, ?, ?)",
                 ["stale.txt", 0, "probe-model", _unit(1.0, 0.0, 0.0)])
    query = _unit(1.0, 0.0)
    unfiltered = sdk.db.query(
        f"SELECT path, vec_cosine(embedding, ?) AS score FROM {TABLE} "
        f"ORDER BY score DESC LIMIT 2", [query])
    filtered = sdk.db.query(
        f"SELECT count(*) AS n FROM {TABLE} "
        f"WHERE length(embedding) = ?", [len(query)])
    return {"top": unfiltered[0]["path"],
            "stale_excluded": filtered[0]["n"],
            "total": sdk.db.query(f"SELECT count(*) AS n FROM {TABLE}")[0]["n"]}
