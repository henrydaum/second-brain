"""Tests for the mid-turn queued message inbox.

A user ``send_text`` that arrives while the agent is mid-turn is queued on
``session.pending_user_messages`` instead of rejected. The running
``ConversationLoop`` drains the queue at its loop boundaries; if the turn
ends first, ``handle_action``'s re-drive loop starts a fresh turn with the
leftovers. ``/cancel`` drops the queue.
"""

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime


def _db(tmp_path):
    return Database(str(tmp_path / "queue.db"))


def _busy_runtime(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db, services={}, config={})
    session = rt.load_conversation("s", cid)
    session.busy = True
    return rt, session


def test_busy_send_text_is_queued_not_rejected(tmp_path):
    rt, session = _busy_runtime(tmp_path)

    out = rt.handle_action("s", "send_text", "hello mid-turn")

    assert out.ok
    assert out.data.get("queued") is True
    assert session.pending_user_messages == ["hello mid-turn"]

    rt.handle_action("s", "send_text", "and another")
    assert session.pending_user_messages == ["hello mid-turn", "and another"]


def test_busy_empty_text_is_still_rejected(tmp_path):
    rt, session = _busy_runtime(tmp_path)

    out = rt.handle_action("s", "send_text", "")

    assert not out.ok
    assert out.error["code"] == "empty_input"
    assert session.pending_user_messages == []


def test_cancel_clears_the_queue(tmp_path):
    rt, session = _busy_runtime(tmp_path)
    rt.handle_action("s", "send_text", "queued one")

    out = rt.handle_action("s", "cancel", None)

    assert "Cancelled." in out.messages
    assert session.pending_user_messages == []
    assert session.cancel_event.is_set()


def test_non_send_text_actions_still_get_busy_error(tmp_path):
    rt, _ = _busy_runtime(tmp_path)

    out = rt.handle_action("s", "call_command", {"name": "anything", "args": {}})

    assert not out.ok
    assert out.error["code"] == "busy"


def test_end_of_turn_leftover_starts_a_fresh_turn(tmp_path):
    """A message queued in the closing race window (after the loop's final
    drain, before busy=False) is dispatched as a real user send_text once the
    turn ends, driving a follow-up turn."""
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db, services={}, config={})
    session = rt.load_conversation("s", cid)

    turns = []

    def fake_drive(sess, out):
        turns.append(list(m["content"] for m in sess.history if m["role"] == "user"))
        if len(turns) == 1:
            # Simulate the race: a message lands after the loop's final drain.
            sess.pending_user_messages.append("leftover")
        out.messages.append(f"reply {len(turns)}")
        # Mimic the real driver's finally-block hand-back.
        sess.cs.set_priority("user")
        sess.busy = False
        return out

    rt._drive_agent_turn = fake_drive

    out = rt.handle_action("s", "send_text", "first message")

    assert len(turns) == 2
    # The follow-up turn saw the leftover as a real user history row.
    assert turns[1][-1] == "leftover"
    assert session.pending_user_messages == []
    assert "reply 1" in out.messages and "reply 2" in out.messages
    assert session.cs.turn_priority == "user"
