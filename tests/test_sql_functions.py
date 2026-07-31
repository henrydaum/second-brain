"""What ``vec_cosine`` promises to any query that uses it.

The point of the function is that the *answer* crosses the sandbox boundary
instead of the data, so these tests are about the SQL contract — that it ranks
correctly, that it composes with ORDER BY / LIMIT, and that every malformed
input is NULL rather than an exception. A raising scalar function aborts the
whole statement, which would turn one stale row into a failed search.
"""

from __future__ import annotations

import math
import sqlite3
import struct

import pytest

from pipeline import sql_functions


def _vector(*values) -> bytes:
    """A float32 blob, the way ``numpy.astype(float32).tobytes()`` writes one."""
    return struct.pack(f"<{len(values)}f", *values)


def _unit(*values) -> bytes:
    """The same, L2-normalized — what ``service_embed`` stores."""
    norm = math.sqrt(sum(v * v for v in values))
    return _vector(*[v / norm for v in values])


@pytest.fixture
def conn():
    """A connection carrying the kernel's scalar functions."""
    connection = sqlite3.connect(":memory:")
    sql_functions.register(connection)
    return connection


def _cosine(conn, left, right):
    """The function's answer for one pair."""
    return conn.execute("SELECT vec_cosine(?, ?)", (left, right)).fetchone()[0]


# ── the arithmetic ──────────────────────────────────────────────────────


def test_a_vector_is_perfectly_similar_to_itself(conn):
    vector = _unit(1.0, 2.0, 3.0, 4.0)
    assert _cosine(conn, vector, vector) == pytest.approx(1.0, abs=1e-6)


def test_opposite_vectors_score_minus_one(conn):
    assert _cosine(conn, _unit(1.0, 2.0), _unit(-1.0, -2.0)) == \
        pytest.approx(-1.0, abs=1e-6)


def test_orthogonal_vectors_score_zero(conn):
    assert _cosine(conn, _unit(1.0, 0.0), _unit(0.0, 1.0)) == \
        pytest.approx(0.0, abs=1e-6)


def test_magnitude_does_not_change_the_answer(conn):
    """Cosine, not dot product.

    Stored vectors happen to be normalized, so the two agree today. A plugin
    storing raw vectors would get silently mis-ranked results from a dot
    product, which is the same class of bug as embedding a filename.
    """
    small = _vector(1.0, 2.0, 3.0)
    large = _vector(10.0, 20.0, 30.0)
    assert _cosine(conn, small, large) == pytest.approx(1.0, abs=1e-6)


# ── every bad input is NULL, never a raise ──────────────────────────────


@pytest.mark.parametrize("left, right", [
    (None, _vector(1.0)),                       # NULL column
    (_vector(1.0), None),                       # NULL parameter
    (_vector(1.0, 2.0), _vector(1.0, 2.0, 3.0)),  # a stale model's dimensions
    (b"abc", b"abc"),                           # not a multiple of four bytes
    (b"", b""),                                 # empty blob
    ("some text", "some text"),                 # a TEXT column
    (7, 7),                                     # an INTEGER column
    (_vector(0.0, 0.0), _vector(1.0, 1.0)),     # zero vector: no direction
])
def test_a_pair_it_cannot_compare_is_null(conn, left, right):
    assert _cosine(conn, left, right) is None


def test_a_null_never_aborts_the_statement(conn):
    """One unusable row must not fail the search.

    A scalar function that raises takes the whole SELECT with it, so a single
    embedding left behind by a since-changed model would break every query
    over that table rather than dropping out of the ranking.
    """
    conn.execute("CREATE TABLE e(path TEXT, embedding BLOB)")
    conn.executemany("INSERT INTO e VALUES(?, ?)", [
        ("good", _unit(1.0, 0.0)),
        ("stale", _vector(1.0, 0.0, 0.0)),       # different dimensions
        ("empty", None),
        ("text", "not a vector"),
    ])
    rows = conn.execute(
        "SELECT path, vec_cosine(embedding, ?) s FROM e ORDER BY s DESC",
        (_unit(1.0, 0.0),)).fetchall()
    assert rows[0][0] == "good"
    assert [r[0] for r in rows[1:]] == ["stale", "empty", "text"] or \
        all(r[1] is None for r in rows[1:])


# ── the shape the search tools actually use ─────────────────────────────


def test_the_answer_crosses_not_the_corpus(conn):
    """The whole reason this exists: rank in SQL, return the top few.

    Five rows leave the database for a table of a thousand vectors, which is
    what keeps a sandboxed search inside ``DB_MAX_ROWS`` instead of paging the
    entire corpus across the wire.
    """
    conn.execute("CREATE TABLE e(path TEXT, chunk_index INT, "
                 "model_name TEXT, embedding BLOB)")
    rows = []
    for i in range(1000):
        angle = i * (math.pi / 2000)             # sweep 0..90 degrees
        rows.append((f"doc{i}.txt", i, "m",
                     _unit(math.cos(angle), math.sin(angle))))
    conn.executemany("INSERT INTO e VALUES(?, ?, ?, ?)", rows)

    found = conn.execute(
        "SELECT path, chunk_index, vec_cosine(embedding, ?) AS score "
        "FROM e WHERE model_name = ? ORDER BY score DESC LIMIT 5",
        (_unit(1.0, 0.0), "m")).fetchall()

    assert len(found) == 5
    # Closest to the query axis first, and strictly descending.
    assert [r[1] for r in found] == [0, 1, 2, 3, 4]
    scores = [r[2] for r in found]
    assert scores == sorted(scores, reverse=True)


def test_it_composes_with_a_join(conn):
    """An expression, not a search — so content comes back in the same query."""
    conn.execute("CREATE TABLE emb(path TEXT, i INT, embedding BLOB)")
    conn.execute("CREATE TABLE chunks(path TEXT, i INT, content TEXT)")
    conn.execute("INSERT INTO emb VALUES('a.txt', 0, ?)", (_unit(1.0, 0.0),))
    conn.execute("INSERT INTO emb VALUES('b.txt', 0, ?)", (_unit(0.0, 1.0),))
    conn.executemany("INSERT INTO chunks VALUES(?, ?, ?)",
                     [("a.txt", 0, "about cats"), ("b.txt", 0, "about dogs")])

    row = conn.execute(
        "SELECT e.path, c.content, vec_cosine(e.embedding, ?) AS score "
        "FROM emb e JOIN chunks c ON c.path = e.path AND c.i = e.i "
        "ORDER BY score DESC LIMIT 1", (_unit(1.0, 0.0),)).fetchone()
    assert (row[0], row[1]) == ("a.txt", "about cats")


def test_the_query_vector_is_cached_without_changing_answers(conn):
    """The one-slot cache must not leak an answer between statements.

    Every row in one scan is compared against the same parameter, so it is
    decoded once; the row after a *different* query has to see the new one.
    """
    conn.execute("CREATE TABLE e(path TEXT, embedding BLOB)")
    conn.executemany("INSERT INTO e VALUES(?, ?)", [
        ("x", _unit(1.0, 0.0)), ("y", _unit(0.0, 1.0))])

    for query, expected in ((_unit(1.0, 0.0), "x"), (_unit(0.0, 1.0), "y"),
                            (_unit(1.0, 0.0), "x")):
        top = conn.execute(
            "SELECT path FROM e ORDER BY vec_cosine(embedding, ?) DESC LIMIT 1",
            (query,)).fetchone()[0]
        assert top == expected


# ── the two implementations must agree ──────────────────────────────────


def test_both_implementations_give_the_same_answer():
    """numpy is an accelerator; the stdlib path is the contract.

    Skipped where numpy is absent, which is the kernel's own baseline — the
    library only appears once a package like ``service_embed`` installs it.
    """
    pytest.importorskip("numpy")
    fast = sql_functions._make_cosine_numpy()
    slow = sql_functions._make_cosine_stdlib()
    pairs = [
        (_unit(1.0, 2.0, 3.0), _unit(3.0, 2.0, 1.0)),
        (_vector(0.5, -0.5), _vector(-2.0, 2.0)),
        (_vector(1.0, 0.0), b"bad"),
        (None, _vector(1.0)),
    ]
    for left, right in pairs:
        one, two = fast(left, right), slow(left, right)
        if one is None or two is None:
            assert one is two
        else:
            assert one == pytest.approx(two, abs=1e-6)


def test_a_database_registers_it(tmp_path):
    """Reached through ``Database``, which is how a plugin's query gets it."""
    from pipeline.database import Database

    db = Database(str(tmp_path / "t.db"))
    try:
        vector = _unit(1.0, 1.0)
        rows = db.query_rows("SELECT vec_cosine(?, ?) AS s", (vector, vector))
        assert rows[0]["s"] == pytest.approx(1.0, abs=1e-6)
    finally:
        db.conn.close()


# ── through the sandbox, on both runners ────────────────────────────────

FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" \
    / "sandbox_vector_plugin.py"


@pytest.fixture
def both(tmp_path):
    """Run one fixture function under each runner, against a real database.

    Both runners, because the query vector crosses as a *bound parameter*:
    in-process there is no serialization at all, so a bytes-handling mistake
    would only appear once the same code ran behind a pipe.
    """
    import importlib.util
    from types import SimpleNamespace

    from pipeline.database import Database
    from sandbox import Interpreter, run_in_process
    from sandbox.runner_subprocess import run_in_subprocess

    db = Database(str(tmp_path / "vectors.db"))
    interp = Interpreter(context=SimpleNamespace(db=db, user_id=1))

    def run(func_name, **kwargs):
        """Execute under in-process and subprocess runners."""
        spec = importlib.util.spec_from_file_location("vec_fixture", FIXTURE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            run_in_process(interp, getattr(module, func_name),
                           name=func_name, kwargs=kwargs, timeout=120),
            run_in_subprocess(interp, str(FIXTURE), func_name,
                              name=func_name, kwargs=kwargs, timeout=120),
        )

    yield run
    interp.shutdown()


def test_a_sandboxed_plugin_ranks_without_holding_the_corpus(both):
    """The test the function exists for.

    A thousand vectors are in the table and three rows come back. The plugin
    never sees the other 997, which is what makes a sandboxed semantic search
    possible at all — ``DB_MAX_ROWS`` would otherwise force it to page the
    whole corpus over the wire for every query.
    """
    in_proc, sub = both("rank_by_similarity", top_k=3)
    assert in_proc.ok and sub.ok, f"{in_proc.error} / {sub.error}"
    assert in_proc.data == sub.data
    assert in_proc.data["count"] == 3
    assert in_proc.data["paths"] == ["doc0.txt", "doc1.txt", "doc2.txt"]
    assert in_proc.data["scores"] == sorted(in_proc.data["scores"],
                                            reverse=True)


def test_a_stale_vector_does_not_break_a_sandboxed_search(both):
    """One leftover row from another model must not fail every query."""
    in_proc, sub = both("stale_dimensions_do_not_break_the_search")
    assert in_proc.ok and sub.ok, f"{in_proc.error} / {sub.error}"
    assert in_proc.data == sub.data
    assert in_proc.data["top"] == "doc0.txt"
    assert in_proc.data["stale_excluded"] == in_proc.data["total"] - 1
