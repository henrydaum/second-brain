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
from plugins.BaseFrontend import BaseFrontend
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


def test_set_session_user_switches_account_and_never_crosses_ownership(tmp_path):
    """Changing identity on a *live* session behaves like an account switch.

    Regression for a hazard the stateful fuzzer surfaced: ``set_session_user``
    used to overwrite ``session.user_id`` while the session still held the old
    user's conversation, leaving the new identity able to read/append to a
    conversation it does not own. It must instead detach the departing user's
    conversation (remembering it as their last-active) and load the new user's
    own last-active.
    """
    db = Database(str(tmp_path / "switch.db"))
    alice = db.upsert_user("web", "alice")
    base_cid = db.create_conversation(title="base", user_id=DEFAULT_USER_ID)
    alice_cid = db.create_conversation(title="alice", user_id=alice)
    db.set_user_config(alice, {"last_active_conversation_id": alice_cid})

    rt = ConversationRuntime(db=db, services={}, config={})
    rt.set_session_user("s", DEFAULT_USER_ID)
    rt.load_conversation("s", base_cid)
    assert rt.sessions["s"].conversation_id == base_cid

    # Switch the live session to alice.
    rt.set_session_user("s", alice)

    # Identity moved, and the session is no longer holding base's conversation.
    assert rt.session_user_id("s") == alice
    assert rt.sessions["s"].conversation_id != base_cid
    # Alice is dropped into her own last-active conversation.
    assert rt.sessions["s"].conversation_id == alice_cid
    # The departing base user's conversation was remembered for switch-back.
    assert db.get_user_config(DEFAULT_USER_ID)["last_active_conversation_id"] == base_cid


def test_delete_conversation_detaches_live_sessions(tmp_path):
    """Deleting a conversation must reconcile any session still holding it.

    Regression for a bug of the same class as the identity-switch one: a
    conversation can be deleted from a different session than the one viewing it
    (another tab/frontend, the agent, or ``/conversations`` deleting the
    currently-open conversation). The holding session used to keep
    ``conversation_id`` pointing at the deleted row and crash on its next write
    with a FOREIGN KEY violation. It must be detached to ``None`` instead.
    """
    db = Database(str(tmp_path / "del.db"))
    rt = ConversationRuntime(db=db, services={}, config={})
    rt.set_session_user("A", DEFAULT_USER_ID)
    cid = db.create_conversation(title="x", user_id=DEFAULT_USER_ID)
    rt.load_conversation("A", cid)
    assert rt.sessions["A"].conversation_id == cid

    # Delete from a *different* session owned by the same user.
    rt.set_session_user("B", DEFAULT_USER_ID)
    assert rt.delete_conversation("B", cid) is True

    assert db.get_conversation(cid) is None
    assert rt.sessions["A"].conversation_id is None  # detached, not dangling


def test_set_session_user_with_no_prior_conversation_is_a_plain_bind(tmp_path):
    """The up-front bind path (no conversation yet) stays a simple identity set."""
    db = Database(str(tmp_path / "bind.db"))
    alice = db.upsert_user("web", "alice")
    rt = ConversationRuntime(db=db, services={}, config={})

    rt.set_session_user("s", alice)  # no session/conversation existed yet

    assert rt.session_user_id("s") == alice
    assert rt.sessions["s"].conversation_id is None


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


def test_frontend_identify_can_mint_user_type(tmp_path):
    class WebFrontend(BaseFrontend):
        name = "web"

    db = Database(str(tmp_path / "frontend-user-type.db"))
    rt = ConversationRuntime(db=db, services={}, config={})
    frontend = WebFrontend()
    frontend.runtime = rt

    uid = frontend.identify("s", "alice", user_type="creator")

    assert db.get_user(uid)["user_type"] == "creator"
    assert rt.session_user_id("s") == uid
