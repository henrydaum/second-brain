"""Token usage: the four counts on ``LLMResponse`` and the ``llm_usage`` record.

The counts are what every backend normalizes into, so the tests here are about
meaning surviving the trip — ``None`` staying "not reported", the wire using
the column names and no other — and about every model call landing as one
attributed row.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import state_machine  # noqa: F401  (package-init order; see test_conversation_loop)
from pipeline.database import Database
from runtime.conversation_loop import ConversationLoop
from sandbox.guest.llm import USAGE_FIELDS, LLMResponse
from sandbox.guest.requests import AGENT_COMPLETE
from tests.support import FakeLLM, FakeRegistry, agent_state, call_handler
from tests.support import response as fake_response


def _rows(db):
    cur = db.conn.execute("SELECT * FROM llm_usage ORDER BY id")
    names = [c[0] for c in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


# ── the contract ──────────────────────────────────────────────────────

def test_counts_round_trip_and_absence_stays_absent():
    """``None`` means the provider did not say, and must cross as ``None``."""
    response = LLMResponse(content="hi", input_tokens=8177,
                           cache_read_tokens=7936, output_tokens=245)

    back = LLMResponse.from_dict(response.to_dict())

    assert back == response
    assert back.cache_write_tokens is None
    assert set(response.usage()) == set(USAGE_FIELDS)
    assert LLMResponse.from_dict({"content": "x"}).usage() == dict.fromkeys(
        USAGE_FIELDS)


def test_the_wire_uses_the_column_names_and_nothing_else():
    """No legacy spelling survives: an old key reports nothing at all."""
    wire = LLMResponse(input_tokens=1).to_dict()

    assert "prompt_tokens" not in wire
    assert LLMResponse.from_dict({"prompt_tokens": 10}).input_tokens is None


def test_a_count_that_is_not_a_count_reads_as_unreported():
    """A wrong figure in the table is worse than a missing one."""
    response = LLMResponse(input_tokens="12", cache_read_tokens=-1,
                           cache_write_tokens=True, output_tokens=0)

    assert response.usage() == {"input_tokens": None,
                                "cache_read_tokens": None,
                                "cache_write_tokens": None,
                                "output_tokens": 0}


# ── the record ────────────────────────────────────────────────────────

def test_an_agent_turn_writes_one_attributed_row(tmp_path):
    db = Database(str(tmp_path / "usage.db"))
    cid = db.create_conversation("Main")
    llm = FakeLLM([fake_response(
        content="Hi", input_tokens=900, cache_read_tokens=800,
        cache_write_tokens=50, output_tokens=40)])
    llm.model_name = "some-model"
    loop = ConversationLoop(llm, FakeRegistry([]), {}, "prompt",
                            session_key="chat")
    loop.runtime = SimpleNamespace(sessions={
        "chat": SimpleNamespace(user_id=7, conversation_id=cid)})

    loop.drive(agent_state(), "agent", [{"role": "user", "content": "hi"}],
               db=db, conversation_id=cid)

    [row] = _rows(db)
    assert row["origin"] == "agent"
    assert row["ok"] == 1
    assert (row["session_key"], row["conversation_id"], row["user_id"]) == (
        "chat", cid, 7)
    assert row["model"] == "some-model"
    assert (row["input_tokens"], row["cache_read_tokens"],
            row["cache_write_tokens"], row["output_tokens"]) == (900, 800, 50, 40)


def test_a_plugin_completion_is_metered_too(tmp_path, monkeypatch):
    """The compactor's call is billed like an agent's, so it is a row too."""
    db = Database(str(tmp_path / "usage.db"))
    cid = db.create_conversation("Main")
    session = SimpleNamespace(conversation_id=cid)
    runtime = SimpleNamespace(sessions={"chat": session}, services={})

    class Brain:
        model_name = "cheap-model"

        def chat(self, request, on_delta=None):
            return LLMResponse(content="summary", input_tokens=120,
                               output_tokens=30)

    from runtime import bootstrap  # noqa: F401  (resolves an import cycle)

    monkeypatch.setattr("runtime.runtime_config.active_llm",
                        lambda _runtime, _session: Brain())
    ctx = SimpleNamespace(runtime=runtime, services={}, db=db,
                          session_key="chat", user_id=3)

    result = call_handler(AGENT_COMPLETE, ctx, {
        "session_key": "chat",
        "messages": [{"role": "user", "content": "history"}]})

    assert result.ok
    [row] = _rows(db)
    assert row["origin"] == "plugin"
    assert (row["conversation_id"], row["user_id"]) == (cid, 3)
    assert (row["input_tokens"], row["output_tokens"]) == (120, 30)
    assert row["cache_read_tokens"] is None


def test_retention_prunes_usage_rows(tmp_path):
    db = Database(str(tmp_path / "usage.db"))
    db.record_llm_usage(origin="agent", ok=True, input_tokens=1)
    db.conn.execute("UPDATE llm_usage SET ts = ?", (time.time() - 10 * 86400,))
    db.conn.commit()

    db.prune_expired(5)

    assert _rows(db) == []


# ── reading it back ───────────────────────────────────────────────────

def _usage_ctx(db, *, user_id=1, conversation_id=None):
    sessions = {"chat": SimpleNamespace(conversation_id=conversation_id)}
    return SimpleNamespace(db=db, user_id=user_id, session_key="chat",
                           runtime=SimpleNamespace(sessions=sessions))


def _seed(db, cid, other_cid):
    db.record_llm_usage(origin="agent", ok=True, user_id=1,
                        conversation_id=cid, model="big", input_tokens=1000,
                        cache_read_tokens=800, output_tokens=100)
    db.record_llm_usage(origin="agent", ok=True, user_id=1,
                        conversation_id=cid, model="small", input_tokens=200,
                        output_tokens=20)
    db.record_llm_usage(origin="compactor", ok=False, user_id=1,
                        conversation_id=cid, model="small")
    # Somebody else's, in somebody else's conversation.
    db.record_llm_usage(origin="agent", ok=True, user_id=2,
                        conversation_id=other_cid, model="big",
                        input_tokens=99999, output_tokens=9)


def test_usage_read_totals_are_the_callers_own(tmp_path):
    from sandbox.guest.requests import USAGE_READ

    db = Database(str(tmp_path / "usage.db"))
    cid, other = db.create_conversation("Mine"), db.create_conversation("Theirs")
    _seed(db, cid, other)

    data = call_handler(USAGE_READ, _usage_ctx(db), {}).data

    totals = data["totals"]
    assert (totals["calls"], totals["failed"], totals["unreported"]) == (3, 1, 1)
    assert (totals["input_tokens"], totals["output_tokens"]) == (1200, 120)
    assert totals["cache_read_tokens"] == 800
    # Nobody reported a cache write: the sum says so rather than saying 0.
    assert totals["cache_write_tokens"] is None
    theirs = call_handler(USAGE_READ, _usage_ctx(db), {
        "conversation_id": other}).data
    assert theirs["totals"]["calls"] == 0


def test_usage_read_current_and_groups(tmp_path):
    from sandbox.guest.requests import USAGE_READ

    db = Database(str(tmp_path / "usage.db"))
    cid, other = db.create_conversation("Mine"), db.create_conversation("Theirs")
    _seed(db, cid, other)

    here = call_handler(USAGE_READ, _usage_ctx(db, conversation_id=cid), {
        "conversation_id": "current", "group_by": "model"}).data
    nowhere = call_handler(USAGE_READ, _usage_ctx(db), {
        "conversation_id": "current"}).data
    by_conv = call_handler(USAGE_READ, _usage_ctx(db), {
        "group_by": "conversation"}).data

    assert here["conversation_id"] == cid
    assert [g["key"] for g in here["groups"]] == ["big", "small"]
    assert here["latest"]["input_tokens"] == 200
    assert nowhere["totals"] is None
    assert [(g["key"], g["title"]) for g in by_conv["groups"]] == [(cid, "Mine")]
    refused = call_handler(USAGE_READ, _usage_ctx(db), {"group_by": "1; DROP"})
    assert not refused.ok
