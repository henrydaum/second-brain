"""Reading a conversation without being able to be crushed by one.

``conv.read`` was an unbounded ``SELECT *``. Every row ever written came back,
state markers included — the state machine's own bookkeeping, re-serialized in
full on every action, which on a real conversation was 19.25 MB of a 20.13 MB
answer. Past ``protocol.MAX_MESSAGE_BYTES`` the answer stopped being
deliverable, and the HTTP frontend's ``poll`` raised on every attempt: no
requests drained, the client's held request destroyed by ``collect_act``'s
one-shot delivery, and a UI that reconnect-stormed a dead tunnel.

Two things are pinned here. **Bookkeeping does not reach a reader**, and
compaction markers still do, because those are a fact about the conversation
rather than about the kernel. And **the read is paged by bytes**, because a row
count bounds nothing when one row is a 100 KB ``edit_file`` argument, and
because a transcript grows without limit whatever the context window is —
compaction shrinks what the model sees and deletes nothing.
"""

import json

import pytest

from pipeline.database import Database
from sandbox.guest.codes import ERROR_TOO_LARGE
from sandbox.guest.protocol import MAX_MESSAGE_BYTES
from sandbox.guest.requests import CONV_READ
from sandbox.handlers.kernel import (CONV_MAX_BYTES, CONV_MAX_ROWS,
                                     CONV_PAGE_ROWS, _STATE_PREFIX)
from state_machine.serialization import (pack_compaction, pack_state,
                                         save_compaction_marker,
                                         save_state_marker)
from tests.support import call_handler


class _Ctx:
    """The little a conversation read asks of its context."""

    def __init__(self, db):
        self.db = db
        self.runtime = None
        self.session_key = "chat"
        self.user_id = 1


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "reads.db"))


def _conversation(db, *, turns=3, marker_every=True):
    """A conversation shaped like a real one: a marker per action."""
    cid = db.create_conversation("Long one")
    for i in range(turns):
        db.save_message(cid, "user", f"question {i}")
        if marker_every:
            save_state_marker(db, cid, {"phase": "awaiting_input",
                                        "history": ["x" * 500]})
        db.save_message(cid, "assistant", f"answer {i}")
    return cid


def _read(db, **args):
    result = call_handler(CONV_READ, _Ctx(db), args)
    assert result.ok, result.error
    return result.data


# ── The prefix is a copy, and copies drift ────────────────────────────

def test_the_state_prefix_matches_what_the_packer_writes():
    """``_STATE_PREFIX`` restates a fact ``serialization`` owns, because
    importing that module from ``pipeline.database`` closes a cycle through
    ``state_machine.conversation``. A copy that drifts does not raise — it
    silently starts shipping bookkeeping again, which is the whole bug."""
    assert pack_state({"phase": "x"}).startswith(_STATE_PREFIX)
    assert not pack_compaction("summary").startswith(_STATE_PREFIX)


# ── What reaches a reader ─────────────────────────────────────────────

def test_state_markers_do_not_reach_a_reader(db):
    cid = _conversation(db, turns=4)

    messages = _read(db, id=cid)["messages"]

    assert messages, "the conversation itself must still come back"
    assert not [m for m in messages
                if (m["content"] or "").startswith(_STATE_PREFIX)]
    assert [m["role"] for m in messages] == ["user", "assistant"] * 4


def test_a_compaction_marker_survives(db):
    """It says the agent's view was replaced at this point, which is a fact
    about the conversation. A client draws a divider from it."""
    cid = _conversation(db, turns=1)
    save_compaction_marker(db, cid, "what happened earlier")
    db.save_message(cid, "user", "after the fold")

    messages = _read(db, id=cid)["messages"]

    markers = [m for m in messages if m["role"] == "system"]
    assert len(markers) == 1
    assert json.loads(markers[0]["content"])["summary"] == "what happened earlier"


def test_the_raw_state_blob_is_not_answered_any_more(db):
    """``details`` existed to deliver the two fields derived from the marker.
    The marker itself had a producer and no consumer anywhere — kernel, store,
    UI or protocol document — and it is ~200 KB of exactly what this call now
    exists to leave behind."""
    cid = _conversation(db, turns=1)

    data = _read(db, id=cid, details=True)

    assert "state" not in data
    assert "agent_profile" in data and "notification_mode" in data


def test_details_still_answers_from_the_newest_marker(db):
    """Sought directly rather than scanned out of ``messages`` — which is why
    the markers had to be in the answer before, and which stopped working the
    moment a page might not contain the newest one."""
    cid = db.create_conversation("Profiled")
    save_state_marker(db, cid, {"active_agent_profile": "first"})
    for i in range(300):
        db.save_message(cid, "user", f"filler {i}")
    save_state_marker(db, cid, {"active_agent_profile": "latest",
                                "notification_mode": "on"})

    data = _read(db, id=cid, details=True, limit=5)

    assert data["agent_profile"] == "latest"
    assert data["notification_mode"] == "on"


# ── Paging ────────────────────────────────────────────────────────────

def test_the_default_page_is_the_newest_one(db):
    """What opening a conversation means. Taking the *first* rows would show
    somebody the beginning of a conversation they have been having all day."""
    cid = db.create_conversation("Long")
    for i in range(CONV_PAGE_ROWS + 50):
        db.save_message(cid, "user", f"m{i}")

    data = _read(db, id=cid)

    assert len(data["messages"]) == CONV_PAGE_ROWS
    assert data["has_more"] is True
    assert data["messages"][-1]["content"] == f"m{CONV_PAGE_ROWS + 49}"


def test_messages_always_arrive_oldest_first(db):
    """Whichever direction was paged to reach them, so a client never has to
    know which way it asked."""
    cid = db.create_conversation("Ordered")
    for i in range(10):
        db.save_message(cid, "user", f"m{i}")

    newest = _read(db, id=cid, limit=4)["messages"]
    older = _read(db, id=cid, limit=4, before_id=newest[0]["id"])["messages"]

    assert [m["content"] for m in newest] == ["m6", "m7", "m8", "m9"]
    assert [m["content"] for m in older] == ["m2", "m3", "m4", "m5"]


def test_paging_backwards_reaches_the_start_and_says_so(db):
    cid = db.create_conversation("Walkable")
    for i in range(25):
        db.save_message(cid, "user", f"m{i}")

    seen, cursor, pages = [], None, 0
    while True:
        data = _read(db, id=cid, limit=10, before_id=cursor)
        seen = data["messages"] + seen
        pages += 1
        if not data["has_more"]:
            break
        cursor = data["oldest_id"]
        assert pages < 10, "paging did not terminate"

    assert [m["content"] for m in seen] == [f"m{i}" for i in range(25)]


def test_since_id_zero_is_the_oldest_page(db):
    """The titler's shape, and the reason no third ``order`` argument exists:
    ``ledger.read``'s vocabulary already answers 'walk forwards from here'."""
    cid = db.create_conversation("Titled")
    for i in range(50):
        db.save_message(cid, "user", f"m{i}")

    data = _read(db, id=cid, since_id=0, limit=12)

    assert [m["content"] for m in data["messages"]] == [f"m{i}" for i in range(12)]
    assert data["has_more"] is True


def test_limit_zero_answers_metadata_only(db):
    """For a caller that came for the conversation's own row and would
    otherwise pull a whole transcript to read a title."""
    cid = _conversation(db, turns=2)

    data = _read(db, id=cid, limit=0)

    assert data["messages"] == []
    assert data["conversation"]["title"] == "Long one"
    assert data["has_more"] is True


def test_the_page_edges_come_back_so_a_client_need_not_derive_them(db):
    cid = db.create_conversation("Edged")
    for i in range(5):
        db.save_message(cid, "user", f"m{i}")

    data = _read(db, id=cid)

    assert data["oldest_id"] == data["messages"][0]["id"]
    assert data["newest_id"] == data["messages"][-1]["id"]
    assert data["has_more"] is False


def test_an_empty_conversation_answers_nulls_rather_than_raising(db):
    cid = db.create_conversation("Empty")

    data = _read(db, id=cid)

    assert data["messages"] == []
    assert data["oldest_id"] is None and data["newest_id"] is None
    assert data["has_more"] is False


# ── The cap that actually holds ───────────────────────────────────────

def test_the_byte_budget_binds_before_the_row_count(db):
    """The bug in one test. Twenty rows is nothing; twenty *fat* rows is more
    than the wire carries, and a row limit cannot tell them apart. The
    conversation that found this had a single ``edit_file`` argument of
    102 KB, re-serialized into every marker for the next hundred actions."""
    cid = db.create_conversation("Fat")
    for i in range(20):
        db.save_message(cid, "user", "x" * 100_000)

    data = _read(db, id=cid, limit=CONV_MAX_ROWS, max_bytes=250_000)

    assert 0 < len(data["messages"]) < 20
    assert data["has_more"] is True


def test_one_row_larger_than_the_budget_still_comes_back(db):
    """An empty page with ``has_more`` set is a client that pages forever and
    never advances — worse than one oversized answer, which the funnel guard
    catches anyway."""
    cid = db.create_conversation("Whale")
    db.save_message(cid, "user", "x" * 50_000)

    data = _read(db, id=cid, max_bytes=1024)

    assert len(data["messages"]) == 1


def test_a_conversation_larger_than_the_wire_still_reads(db):
    """The regression test for the crash. Twenty-five megabytes of transcript
    — more than ``MAX_MESSAGE_BYTES`` — and the answer both arrives and fits."""
    cid = db.create_conversation("Enormous")
    for i in range(50):
        db.save_message(cid, "assistant", "y" * 500_000)

    result = call_handler(CONV_READ, _Ctx(db), {"id": cid, "details": True})

    assert result.ok
    encoded = len(json.dumps(result.to_dict(), default=str).encode("utf-8"))
    assert encoded < MAX_MESSAGE_BYTES
    assert result.data["has_more"] is True


def test_the_budget_is_derived_from_the_wire_not_guessed(db):
    """``fs_net.MAX_READ_BINARY``'s lesson: a constant guessed independently
    drifts, and the failure it drifts into is an unsendable result."""
    assert CONV_MAX_BYTES < MAX_MESSAGE_BYTES
    assert CONV_MAX_BYTES == MAX_MESSAGE_BYTES - 1024 * 1024


def test_a_caller_cannot_raise_the_budget_past_the_wire(db):
    cid = _conversation(db, turns=1)

    data = call_handler(CONV_READ, _Ctx(db),
                        {"id": cid, "max_bytes": MAX_MESSAGE_BYTES * 10})

    assert data.ok    # clamped rather than refused; ``int_arg`` bounds it


def test_a_bad_paging_argument_is_named_rather_than_ignored(db):
    cid = _conversation(db, turns=1)

    result = call_handler(CONV_READ, _Ctx(db), {"id": cid, "limit": "lots"})

    assert not result.ok
    assert "limit" in result.error


# ── The unbounded reader is still there, on purpose ───────────────────

def test_the_unbounded_reader_still_returns_markers(db):
    """``get_conversation_messages`` stays as it was: its callers rebuild the
    agent's history and ``latest_state`` needs the markers. Narrowing it would
    have broken restart recovery to fix a display problem."""
    cid = _conversation(db, turns=2)

    rows = db.get_conversation_messages(cid)

    assert [r for r in rows if (r["content"] or "").startswith(_STATE_PREFIX)]
