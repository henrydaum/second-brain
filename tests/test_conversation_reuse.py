"""Starting a new conversation takes over one nobody ever used.

Blank conversations used to pile up forever: every ``/new`` inserted a row, and
nothing reclaimed it — ``prune_expired`` deletes by ``updated_at``, and a state
marker is written after almost every action, so a conversation nobody ever
spoke in never looks idle.

Two things about the predicate are easy to get wrong and silent when wrong, so
they are stated here rather than left to the SQL: emptiness is *no row that is
not a marker* (a message count is never zero), and a placeholder title plus no
category is part of being unused, because those are the only intent an empty
conversation can carry.
"""

from types import SimpleNamespace

import pytest

from pipeline.database import DEFAULT_USER_ID, PLACEHOLDER_TITLES, Database
from sandbox.guest.requests import CONV_CREATE
from state_machine.serialization import save_state_marker
from tests.support import call_handler, plain_runtime


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "reuse.db"))


def blank(db, title="New conversation", **kwargs):
    """A conversation as ``/new`` leaves it: a placeholder title and a marker."""
    cid = db.create_conversation(title=title,
                                 user_id=kwargs.pop("user_id", DEFAULT_USER_ID),
                                 **kwargs)
    save_state_marker(db, cid, {"conversation_id": cid})
    return cid


def age(db, conversation_id):
    """Push a row out of the quiet window.

    A conversation is hidden from the *finder* for ``REUSE_QUIET_SECONDS``
    after it was touched, so a row created moments ago is deliberately not a
    candidate. Every test that walks the finder path has to say how old the
    row is, the same way one walking the own-session path has to open a
    session — the window is not an obstacle to work around, it is the
    reservation that stops two callers being handed one conversation in the
    gap before either binds a session to it.
    """
    # Imported here, not at module scope: ``state_machine/__init__`` imports
    # back out of this module, so naming it first is a circular import.
    from runtime.conversation_runtime import REUSE_QUIET_SECONDS

    db.conn.execute(
        "UPDATE conversations SET updated_at = updated_at - ? WHERE id = ?",
        (REUSE_QUIET_SECONDS + 1, conversation_id))
    db.conn.commit()
    return conversation_id


# ── The predicate ────────────────────────────────────────────────────

def test_a_conversation_holding_only_markers_is_unused(db):
    """The trap a message count falls into.

    ``save_state_marker`` writes a ``role='system'`` row, so
    ``conversation_message_count`` on a conversation nobody has spoken in is
    one, not zero.
    """
    cid = blank(db)
    assert db.conversation_message_count(cid) > 0
    assert db.find_unused_conversation() == cid


def test_a_conversation_somebody_spoke_in_is_not_unused(db):
    cid = blank(db)
    db.save_message(cid, "user", "hello")
    assert db.find_unused_conversation() is None


def test_an_unrecognised_role_still_counts_as_use(db):
    """The predicate is a negative, and this is why.

    ``messages_to_history`` skips ``'system'`` and silently drops any role it
    does not know, so a positive list of transcript roles would make an
    unfamiliar row invisible to a reader and then delete it.
    """
    cid = blank(db)
    db.save_message(cid, "annotation", "written by something else")
    assert db.find_unused_conversation() is None


def test_a_retitled_conversation_is_not_unused(db):
    cid = blank(db)
    db.update_conversation_title(cid, "Q3 ideas")
    assert db.find_unused_conversation() is None


def test_a_cleared_conversation_is_not_unused(db):
    """`/clear` appends " (cleared)", which is a title somebody caused.

    Emptying a conversation on purpose is a decision to keep it, so the row
    falls out of the candidate set by the same rule that protects any other
    title — no special case needed.
    """
    cid = blank(db)
    db.save_message(cid, "user", "hello")
    db.clear_conversation_messages(cid)
    db.update_conversation_title(cid, "New conversation (cleared)")
    assert db.find_unused_conversation() is None


def test_a_filed_conversation_is_not_unused(db):
    assert blank(db, category="Work")
    assert db.find_unused_conversation() is None
    # An empty category is the Main bucket, not a decision.
    cid = blank(db, category="")
    assert db.find_unused_conversation() == cid


def test_a_subagent_conversation_is_not_unused(db):
    blank(db, category="Subagent")
    assert db.find_unused_conversation() is None


def test_candidates_are_scoped_to_the_owner(db):
    theirs = blank(db, user_id=2)
    assert db.find_unused_conversation(user_id=DEFAULT_USER_ID) is None
    assert db.find_unused_conversation(user_id=2) == theirs


def test_a_legacy_null_owner_belongs_to_the_base_user(db):
    """Owner scoping follows ``assert_conversation_access``, not the listings.

    A bare ``user_id = ?`` would leave every pre-ownership row permanently
    unreclaimable — the exact rows most likely to be stale.
    """
    cid = blank(db)
    db.conn.execute("UPDATE conversations SET user_id = NULL WHERE id = ?", (cid,))
    db.conn.commit()
    assert db.find_unused_conversation(user_id=DEFAULT_USER_ID) == cid
    assert db.find_unused_conversation(user_id=2) is None


def test_exclude_and_the_quiet_window_each_drop_a_candidate(db):
    import time

    cid = blank(db)
    assert db.find_unused_conversation(exclude={cid}) is None
    assert db.find_unused_conversation(updated_before=time.time() - 30) is None
    assert db.find_unused_conversation(updated_before=time.time() + 1) == cid


def test_the_newest_candidate_comes_back_first(db):
    blank(db)
    newest = blank(db)
    assert db.find_unused_conversation() == newest


# ── The claim ────────────────────────────────────────────────────────

def test_claiming_resets_the_row(db):
    cid = blank(db)
    db.update_conversation_title_check_count(cid, 12)
    before = db.get_conversation(cid)["created_at"]

    assert db.claim_conversation(cid, "New conversation (Main)") is True

    row = db.get_conversation(cid)
    assert row["title"] == "New conversation (Main)"
    assert row["category"] is None
    assert db.conversation_message_count(cid) == 0
    assert row["created_at"] >= before
    # Cleared with the messages: ``list_conversations_for_title_check``
    # compares the live count against this, so a stale high-water mark hides
    # the reused conversation from the re-titling sweep.
    assert row["last_title_check_message_count"] is None


def test_claiming_rechecks_the_predicate_and_changes_nothing_on_a_loss(db):
    """The re-check is what makes the whole operation safe to race."""
    cid = blank(db)
    db.save_message(cid, "user", "landed between the lookup and the claim")

    assert db.claim_conversation(cid, "New conversation") is False
    assert db.conversation_message_count(cid) == 2
    assert db.get_conversation(cid)["title"] == "New conversation"


def test_claiming_refuses_another_owners_row(db):
    cid = blank(db, user_id=2)
    assert db.claim_conversation(cid, "mine now", user_id=DEFAULT_USER_ID) is False
    assert db.get_conversation(cid)["title"] == "New conversation"


# ── The runtime ──────────────────────────────────────────────────────

def test_an_abandoned_conversation_is_taken_over(db):
    runtime = plain_runtime(db)
    first = age(db, runtime.create_conversation("New conversation"))
    second = runtime.reuse_unused_conversation(
        None, title="New conversation", user_id=DEFAULT_USER_ID)

    assert second == first
    assert len(db.list_conversations()) == 1


def test_a_conversation_just_handed_out_is_not_taken_again(db):
    """The reservation, and the gap it covers.

    ``create_conversation`` returns an id some lines before its caller binds a
    session to it — ``open_session`` and the subagent supervisor both spend a
    few statements in that gap — and during it the row is in no session's
    ``conversation_id``. Bumping ``updated_at`` hides it meanwhile, with no
    state to leak and nothing to clean up.
    """
    runtime = plain_runtime(db)
    runtime.create_conversation("New conversation")
    assert runtime.reuse_unused_conversation(None, title="New conversation") is None


def test_create_conversation_never_reuses_on_its_own(db):
    """The subagent guarantee, without ``subagents.py`` needing to opt out.

    Reuse is a decision the ``conv.create`` handler makes. Every other caller
    of ``create_conversation`` — ``open_session``, ``ensure_conversation``, the
    subagent supervisor — goes straight to an insert, so two children can never
    be handed one conversation.
    """
    runtime = plain_runtime(db)
    ids = {runtime.create_conversation("New conversation") for _ in range(3)}
    assert len(ids) == 3


def test_reuse_wipes_the_marker_so_nothing_carries_over(db):
    """A leftover marker is read straight back by ``load_conversation``.

    Without the wipe, the abandoned conversation's ``profile_override`` becomes
    the "new" conversation's agent, silently.
    """
    runtime = plain_runtime(db)
    cid = db.create_conversation(title="New conversation")
    save_state_marker(db, cid, {"conversation_id": cid,
                                "profile_override": "memory_curator"})
    age(db, cid)

    assert runtime.reuse_unused_conversation(None, title="New conversation") == cid
    session = runtime.load_conversation("s", cid)
    assert session.profile_override is None
    assert session.history == []


def test_a_conversation_another_session_holds_is_never_reused(db):
    runtime = plain_runtime(db)
    cid = db.create_conversation(title="New conversation")
    runtime.load_conversation("other", cid)

    assert runtime.reuse_unused_conversation(
        "mine", title="New conversation", allow_own=True) is None


def test_the_callers_own_conversation_is_reusable_only_when_activating(db):
    """The whole point, and the one way it could go wrong.

    Taking over your own live conversation is safe because activation rebuilds
    the session from the row. A detached create handed the same row would erase
    the state of a session somebody is sitting in.
    """
    runtime = plain_runtime(db)
    cid = db.create_conversation(title="New conversation")
    runtime.load_conversation("mine", cid)

    assert runtime.reuse_unused_conversation(
        "mine", title="New conversation", allow_own=False) is None
    assert runtime.reuse_unused_conversation(
        "mine", title="New conversation", allow_own=True) == cid


def test_a_busy_session_keeps_its_conversation(db):
    runtime = plain_runtime(db)
    cid = db.create_conversation(title="New conversation")
    session = runtime.load_conversation("mine", cid)
    session.busy = True

    assert runtime.reuse_unused_conversation(
        "mine", title="New conversation", allow_own=True) is None


def test_a_database_double_without_the_primitives_falls_through(db):
    """Answering None is always safe: the caller creates, as it would have."""
    runtime = plain_runtime(db)
    runtime.db = SimpleNamespace()
    assert runtime.reuse_unused_conversation(None, title="x") is None


def test_a_reuse_is_recorded_as_a_conversation_beginning(db):
    """The flight recorder must not show a create that never happened, and
    must not show nothing at all for the id somebody is suddenly looking at."""
    import json

    runtime = plain_runtime(db)
    cid = age(db, runtime.create_conversation("New conversation"))
    runtime.reuse_unused_conversation(None, title="New conversation")

    rows = db.conn.execute(
        "SELECT * FROM action_ledger WHERE action_type = 'conversation_create'"
        " ORDER BY id").fetchall()
    assert len(rows) == 2
    assert json.loads(rows[0]["args_json"])["reused"] is False
    assert json.loads(rows[1]["args_json"])["reused"] is True
    assert rows[1]["conversation_id"] == cid


def test_the_bus_still_says_a_conversation_was_created(db):
    """A catalog view refreshes on ``created`` and must keep working."""
    from events.event_bus import bus
    from events.event_channels import CONVERSATION_CHANGED

    seen = []
    unsub = bus.subscribe(CONVERSATION_CHANGED, seen.append)
    try:
        runtime = plain_runtime(db)
        age(db, runtime.create_conversation("New conversation"))
        runtime.reuse_unused_conversation(None, title="New conversation")
    finally:
        unsub()

    assert [e["action"] for e in seen] == ["created", "created"]
    assert [e["reused"] for e in seen] == [False, True]


# ── The handler ──────────────────────────────────────────────────────

def ctx_for(runtime, db, session_key="s"):
    return SimpleNamespace(runtime=runtime, db=db, session_key=session_key,
                           user_id=DEFAULT_USER_ID)


def test_creating_twice_through_the_handler_reuses(db):
    runtime = plain_runtime(db)
    ctx = ctx_for(runtime, db)

    first = call_handler(CONV_CREATE, ctx, {"title": "New conversation"})
    age(db, first.data)
    second = call_handler(CONV_CREATE, ctx, {"title": "New conversation"})

    assert second.data == first.data
    assert len(db.list_conversations()) == 1


def test_reuse_empty_false_always_inserts(db):
    runtime = plain_runtime(db)
    ctx = ctx_for(runtime, db)

    first = call_handler(CONV_CREATE, ctx, {"title": "New conversation"})
    age(db, first.data)
    second = call_handler(CONV_CREATE, ctx, {"title": "New conversation",
                                             "reuse_empty": False})

    assert second.data != first.data
    assert len(db.list_conversations()) == 2


def test_naming_a_category_always_inserts(db):
    runtime = plain_runtime(db)
    ctx = ctx_for(runtime, db)

    call_handler(CONV_CREATE, ctx, {"title": "New conversation"})
    call_handler(CONV_CREATE, ctx, {"title": "New conversation",
                                    "category": "Work"})
    assert len(db.list_conversations()) == 2


def test_a_detached_create_leaves_a_live_session_alone(db):
    """``activate=False`` is a plugin asking for a row of its own."""
    runtime = plain_runtime(db)
    ctx = ctx_for(runtime, db)
    held = call_handler(CONV_CREATE, ctx,
                        {"title": "New conversation", "activate": True})
    detached = call_handler(CONV_CREATE, ctx, {"title": "New conversation"})

    assert detached.data != held.data["id"]
    assert runtime.sessions["s"].conversation_id == held.data["id"]


def test_new_twice_over_does_not_advance_the_conversation(db):
    """`/new` in a blank conversation is a reset, not conversation #48."""
    runtime = plain_runtime(db)
    ctx = ctx_for(runtime, db)
    args = {"title": "New conversation (Main)", "activate": True}

    ids = {call_handler(CONV_CREATE, ctx, args).data["id"] for _ in range(3)}
    assert len(ids) == 1
    assert len(db.list_conversations()) == 1


def test_new_after_a_real_message_starts_a_fresh_conversation(db):
    runtime = plain_runtime(db)
    ctx = ctx_for(runtime, db)
    args = {"title": "New conversation (Main)", "activate": True}

    first = call_handler(CONV_CREATE, ctx, args).data["id"]
    db.save_message(first, "user", "hello")
    second = call_handler(CONV_CREATE, ctx, args).data["id"]

    assert second != first
    assert len(db.list_conversations()) == 2


# ── The coupling ─────────────────────────────────────────────────────

def test_the_title_new_uses_is_a_placeholder():
    """`/new` picks its own title, and the kernel decides what a placeholder is.

    Nothing links the two, and getting it wrong has no symptom: reuse simply
    never fires, which looks exactly like the feature being switched off.
    """
    import ast
    from pathlib import Path

    source = Path("bundled/commands/command_new.py").read_text(encoding="utf-8")
    titles = {node.value for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert any(title.strip().lower() in PLACEHOLDER_TITLES for title in titles), (
        "no string in command_new.py is a placeholder title — /new will never "
        "reuse the conversation it just made")
