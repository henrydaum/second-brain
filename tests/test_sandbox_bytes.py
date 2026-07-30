"""Bytes crossing the boundary — and the two runners agreeing that they do.

JSON has no bytes type, so a value that is merely *numeric* — an embedding
vector, a thumbnail, a BLOB column — had no way across. It failed in the worst
direction available: in-process there is no serialization at all, so a plugin
writing a BLOB worked perfectly on a thread and raised ``TypeError`` from deep
inside ``json.dumps`` only once the same file ran in a subprocess. Identical
code, two behaviours, and the pipe is the only place it shows.

That is why every end-to-end test here runs **both** runners and compares them.
A test that exercised one would have passed against the bug.
"""

import json
from pathlib import Path

import pytest

from sandbox import Interpreter, Result, run_in_process
from sandbox.guest import protocol
from sandbox.guest.requests import Request
from sandbox.runner_subprocess import run_in_subprocess

FIXTURE = Path(__file__).parent / "fixtures" / "sandbox_bytes_plugin.py"


# ──────────────────────────────────────────────────────────────────────
# The codec.
# ──────────────────────────────────────────────────────────────────────

def test_bytes_survive_a_json_round_trip():
    """The whole point: pack, serialize, parse, unpack, unchanged."""
    payload = {"vector": b"\x00\x01\xff", "name": "chunk"}
    wire = json.loads(json.dumps(protocol.pack(payload)))
    assert protocol.unpack(wire) == payload


def test_packing_recurses_into_lists_and_dicts():
    """Both interesting shapes are nested — a list of vectors, a row dict."""
    payload = [{"embedding": b"ab"}, {"embedding": b"cd"}]
    assert protocol.unpack(protocol.pack(payload)) == payload


def test_a_payload_without_bytes_is_unchanged():
    """Packing costs a walk and produces an equal value."""
    payload = {"a": [1, 2.5, "x", None, True], "b": {"c": []}}
    assert protocol.pack(payload) == payload


def test_a_dict_carrying_the_tag_among_other_keys_is_not_decoded():
    """Only a *lone* tag is an encoding; anything else is somebody's data.

    Without the length check, a plugin storing a field it happened to call
    ``__bytes__`` would have its whole dict silently replaced by a bytestring.
    """
    payload = {protocol.BYTES_TAG: "AAEC", "note": "mine"}
    assert protocol.unpack(payload) == payload


def test_an_undecodable_tag_is_passed_through_not_raised():
    """A malformed tag is data somebody sent, not a protocol violation."""
    payload = {protocol.BYTES_TAG: "not valid base64!!"}
    assert protocol.unpack(payload) == payload


def test_is_simple_admits_bytes():
    """Bytes may cross — they are packed on the way out."""
    assert protocol.is_simple(b"\x00")
    assert protocol.is_simple({"v": [b"\x00", "s"]})


# ──────────────────────────────────────────────────────────────────────
# The two payload surfaces.
# ──────────────────────────────────────────────────────────────────────

def test_request_args_carry_bytes():
    """A BLOB bound as a ``db.write`` parameter."""
    original = Request("db.write", {"sql": "INSERT INTO t VALUES (?)",
                                    "params": [b"\x01\x02", "text", 3]})
    rebuilt = Request.from_dict(json.loads(json.dumps(original.to_dict())))
    assert rebuilt.args == original.args


def test_result_data_carries_bytes():
    """A ``db.query`` answering with a BLOB column."""
    original = Result(data=[{"path": "a.md", "embedding": b"\xff\x00"}])
    rebuilt = Result.from_dict(json.loads(json.dumps(original.to_dict())))
    assert rebuilt.data == original.data


# ──────────────────────────────────────────────────────────────────────
# End to end, on both runners.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def both(tmp_path):
    """Run one fixture function under each runner, against a real database.

    The database is shared between the two runs on purpose — each function
    writes under its own key — so a difference in what reached sqlite is
    visible in the same table rather than only in the two Results.
    """
    from types import SimpleNamespace

    from pipeline.database import Database

    db = Database(str(tmp_path / "bytes.db"))
    interp = Interpreter(context=SimpleNamespace(db=db, user_id=1))

    def run(func_name, **kwargs):
        """Execute under in-process and subprocess runners."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("bytes_fixture", FIXTURE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        in_proc = run_in_process(
            interp, getattr(module, func_name),
            name=func_name, kwargs=kwargs, timeout=60)
        sub = run_in_subprocess(
            interp, str(FIXTURE), func_name,
            name=func_name, kwargs=kwargs, timeout=60)
        return in_proc, sub

    yield run
    interp.shutdown()


def test_a_blob_round_trips_through_the_database_identically(both):
    """``db.write`` a BLOB, ``db.query`` it back — same on both runners.

    This is the test the whole change exists for. Before it, the subprocess
    half raised ``TypeError`` while the in-process half returned the bytes.
    """
    a, b = both("store_and_fetch", key="k")
    assert a.ok and b.ok, f"{a.error} / {b.error}"
    assert a.data == b.data
    assert a.data["blob"] == b"\x00\x01\x02\xfe\xff"
    assert a.data["length"] == 5


def test_bytes_reach_a_handler_as_bytes_not_as_a_tag(both):
    """The handler never learns which side of a pipe it is on.

    sqlite decides a column's storage class from the *Python type* it was
    bound, so a parameter arriving as a tagged dict would be stored as text
    and read back as text. Asking sqlite for its own opinion is the only
    check that distinguishes the two.
    """
    a, b = both("store_and_report_type", key="typed")
    assert a.ok and b.ok, f"{a.error} / {b.error}"
    assert a.data == b.data == "blob"
