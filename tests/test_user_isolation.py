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
from plugins.commands.command_agent import AgentCommand
from plugins.commands.command_tools import _toggle_skip


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


def test_inject_user_message_refuses_non_owner_without_leaking(runtime):
    cid = _owned_by_a(runtime)
    result = runtime.inject_user_message("B", "hello", conversation_id=cid)
    assert result.ok is False
    assert result.messages == ["No such conversation."]
    assert runtime.db.get_conversation_messages(cid) == []


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


def test_last_active_conversation_is_per_user(tmp_path):
    db = Database(str(tmp_path / "last-active.db"))
    other_uid = db.upsert_user("web", "alice")
    base_cid = db.create_conversation(title="base", user_id=DEFAULT_USER_ID)
    other_cid = db.create_conversation(title="alice", user_id=other_uid)

    rt = ConversationRuntime(db=db, services={}, config={})
    rt.set_session_user("base", DEFAULT_USER_ID)
    rt.set_session_user("alice", other_uid)
    rt.active_session_key = "base"
    rt._persist_active_conversation(base_cid)
    rt.active_session_key = "alice"
    rt._persist_active_conversation(other_cid)

    assert db.get_user_config(DEFAULT_USER_ID)["last_active_conversation_id"] == base_cid
    assert db.get_user_config(other_uid)["last_active_conversation_id"] == other_cid
    assert "last_active_conversation_id" not in rt.config

    rt2 = ConversationRuntime(db=db, services={}, config={})
    rt2.set_session_user("alice", other_uid)
    notice = rt2.restore_last_active("alice")
    assert notice and "alice" in notice
    assert rt2.sessions["alice"].conversation_id == other_cid


def test_agent_switch_persists_active_profile_per_user(tmp_path):
    db = Database(str(tmp_path / "agent-profile.db"))
    uid = db.upsert_user("web", "alice")
    rt = ConversationRuntime(db=db, services={}, config={
        "agent_profiles": {"default": {"llm": "default"}, "writer": {"llm": "default"}},
    })
    rt.set_session_user("alice", uid)
    context = SimpleNamespace(
        config={"agent_profiles": rt.config["agent_profiles"], "active_agent_profile": "default"},
        runtime=rt, session_key="alice", db=db, user_id=uid,
    )

    assert AgentCommand().run({"profile_name": "writer", "action": "switch"}, context) == "Switched agent profile to: writer"
    assert db.get_user_config(uid)["active_agent_profile"] == "writer"
    assert "active_agent_profile" not in rt.config


def test_skip_permissions_persist_per_user(tmp_path):
    db = Database(str(tmp_path / "skip.db"))
    uid = db.upsert_user("web", "alice")
    rt = ConversationRuntime(db=db, services={}, config={})
    rt.set_session_user("alice", uid)
    context = SimpleNamespace(
        config={"skip_permissions": []},
        runtime=rt, session_key="alice", db=db, user_id=uid,
    )

    assert _toggle_skip(context, "run_command", True) == "Skip permissions enabled for run_command."
    assert db.get_user_config(uid)["skip_permissions"] == ["run_command"]
    assert "skip_permissions" not in rt.config
