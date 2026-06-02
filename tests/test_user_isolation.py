"""Cross-user conversation isolation (the "user dimension").

Ownership lives on ``conversations.user_id`` and is enforced by
``runtime.assert_conversation_access`` on every load/mutate-by-id path — not just
hidden from picker lists. These tests drive the guard with a real DB and two
sessions bound to different users, asserting that a non-owner is refused on every
direct path while the owner (and ``override=True``) succeed.
"""

from types import SimpleNamespace

import pytest

# Settle the runtime<->state_machine package-init cycle before importing the
# runtime class (state_machine/__init__ pulls the runtime in).
import state_machine  # noqa: F401
from state_machine import ConversationRuntime
from pipeline.database import Database, DEFAULT_USER_ID


@pytest.fixture
def runtime(tmp_path):
    db = Database(str(tmp_path / "iso.db"))
    rt = ConversationRuntime(db=db, services={}, config={})
    # Two sessions bound to different users. Stub the session objects directly —
    # the guard only reads ``user_id`` and the DB, so this keeps the test focused
    # on enforcement rather than full session hydration.
    rt.sessions = {
        "A": SimpleNamespace(user_id=DEFAULT_USER_ID),
        "B": SimpleNamespace(user_id=2),
    }
    return rt


def _owned_by_a(runtime):
    return runtime.db.create_conversation(title="A's", user_id=DEFAULT_USER_ID)


def test_access_guard(runtime):
    cid = _owned_by_a(runtime)
    assert runtime.assert_conversation_access("A", cid) is True
    assert runtime.assert_conversation_access("B", cid) is False
    assert runtime.assert_conversation_access("B", cid, override=True) is True
    # A missing conversation is reported as inaccessible, not crashing.
    assert runtime.assert_conversation_access("A", 99999) is False


def test_load_history_refuses_non_owner_without_leaking(runtime):
    cid = _owned_by_a(runtime)
    result = runtime.load_history("B", cid)
    assert result.ok is False
    assert result.messages == ["No such conversation."]


def test_mutations_refuse_non_owner_and_are_noops(runtime):
    cid = _owned_by_a(runtime)

    assert runtime.delete_conversation("B", cid) is False
    assert runtime.set_conversation_category("B", cid, "x") is False
    assert runtime.set_conversation_notification_mode("B", cid, "on") is None
    # Nothing changed: the row still exists, uncategorised.
    row = runtime.db.get_conversation(cid)
    assert row is not None and row["category"] is None


def test_owner_can_delete(runtime):
    cid = _owned_by_a(runtime)
    assert runtime.delete_conversation("A", cid) is True
    assert runtime.db.get_conversation(cid) is None


def test_override_bypasses_guard(runtime):
    cid = _owned_by_a(runtime)
    assert runtime.delete_conversation("B", cid, override=True) is True
    assert runtime.db.get_conversation(cid) is None
