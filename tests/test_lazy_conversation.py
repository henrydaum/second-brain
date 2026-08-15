"""A conversation is created by the first message, not before it.

Blank conversations used to pile up because a conversation had to exist before
anything could be said: the runtime refused ``send_text`` without one, so every
caller — ``/new``, and the web client on every page load — made a row in
advance and hoped it got used. Nothing reclaimed the ones that did not.

Creating on the first message removes the failure mode rather than
compensating for it: a conversation with no messages cannot exist, because
nothing makes one. These tests pin the parts of that which are silent when
broken — an unpersisted opening message, a conversation the ledger never heard
about — rather than the parts a person would notice immediately.
"""

from types import SimpleNamespace

import pytest

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)

from events.event_bus import bus
from events.event_channels import CONVERSATION_CHANGED
from pipeline.database import Database
from sandbox.guest.requests import CONV_NEW
from tests.support import call_handler, plain_runtime

CONFIGURED = {"llm_profiles": {"main": {"backend": "x"}}}


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "lazy.db"))


@pytest.fixture
def runtime(db):
    rt = plain_runtime(db, config=dict(CONFIGURED))
    rt.get_session("repl")
    rt.active_session_key = "repl"
    return rt


def rows(db, conversation_id):
    """Transcript rows only — state markers are ``role='system'`` in here."""
    return [m for m in db.get_conversation_messages(conversation_id)
            if m["role"] != "system"]


# ── Nothing until something is said ──────────────────────────────────

def test_a_session_at_rest_holds_no_conversation(runtime, db):
    assert runtime.sessions["repl"].conversation_id is None
    assert db.list_conversations() == []


def test_a_command_creates_nothing(runtime, db):
    """Commands never needed a conversation, which is what makes this possible.

    If they had, every ``/config`` at a fresh prompt would mint a row and the
    pile would come back through a door nobody was watching.
    """
    runtime.handle_action("repl", "call_command", {"name": "mode", "args": {}})
    assert db.list_conversations() == []
    assert runtime.sessions["repl"].conversation_id is None


def test_starting_a_new_conversation_twice_over_leaves_nothing_behind(runtime, db):
    for _ in range(3):
        assert runtime.handle_action("repl", "new_conversation", {}).ok
    assert db.list_conversations() == []


def test_conv_new_unbinds_and_writes_nothing(runtime, db):
    """``/new``'s route in. A frontend reaches the action directly; a command
    holds no desk token and needs the Request."""
    cid = db.create_conversation("Real one")
    db.save_message(cid, "user", "hi")
    runtime.load_conversation("repl", cid)

    ctx = SimpleNamespace(runtime=runtime, db=db, session_key="repl", user_id=1)
    assert call_handler(CONV_NEW, ctx, {}).ok
    assert runtime.sessions["repl"].conversation_id is None
    assert len(db.list_conversations()) == 1


# ── The first message ────────────────────────────────────────────────

def test_a_message_creates_the_conversation_and_is_in_it(runtime, db):
    """The regression an ordering mistake causes, asserted against the table.

    ``absorb_user_action`` writes the user's row under ``if runtime.db and
    session.conversation_id``, so a conversation created any later than this
    drops the opening message — it survives in ``session.history``, so the
    model and the live UI both look right, and only the stored transcript is
    wrong, beginning with the assistant's reply.
    """
    runtime.handle_action("repl", "send_text", {"text": "what is a microkernel"})

    cid = runtime.sessions["repl"].conversation_id
    assert cid is not None
    assert len(db.list_conversations()) == 1
    assert [(m["role"], m["content"]) for m in rows(db, cid)] == [
        ("user", "what is a microkernel")]


def test_the_conversation_is_titled_with_the_placeholder(runtime, db):
    """The kernel names it, the ``update_titles`` package renames it.

    Naming the row after the first message reads better for the few seconds
    before anything else happens, and it costs the only real titler its
    trigger: that package replaces a title still looking kernel-generated and
    leaves anything else alone, which is what protects a rename you made
    yourself. A first-message title is indistinguishable from one you chose, so
    the sweep skipped every conversation and every title stayed the opening
    sentence, cut off at eighty characters.
    """
    runtime.handle_action("repl", "send_text", {"text": "what is a microkernel"})
    cid = runtime.sessions["repl"].conversation_id
    assert db.get_conversation(cid)["title"] == "New Conversation"


def test_a_second_message_does_not_create_a_second_conversation(runtime, db):
    runtime.handle_action("repl", "send_text", {"text": "first"})
    runtime.handle_action("repl", "send_text", {"text": "second"})
    assert len(db.list_conversations()) == 1


def test_starting_a_new_one_after_a_message_creates_a_second(runtime, db):
    runtime.handle_action("repl", "send_text", {"text": "first"})
    runtime.handle_action("repl", "new_conversation", {})
    assert len(db.list_conversations()) == 1      # nothing said yet
    runtime.handle_action("repl", "send_text", {"text": "second"})
    assert len(db.list_conversations()) == 2


# ── It has to behave like a real creation ────────────────────────────

def test_the_ledger_and_the_bus_hear_about_it(runtime, db):
    """``ensure_conversation`` used to call ``db.create_conversation`` directly.

    Survivable while it was a rare path; as *the* path it would mean the flight
    recorder never recording a conversation starting, and a client's list never
    learning of one — both silent.
    """
    seen = []
    unsubscribe = bus.subscribe(CONVERSATION_CHANGED, seen.append)
    try:
        runtime.handle_action("repl", "send_text", {"text": "hello"})
    finally:
        unsubscribe()
    cid = runtime.sessions["repl"].conversation_id

    assert [(e["action"], e["conversation_id"]) for e in seen] == [("created", cid)]
    recorded = db.conn.execute(
        "SELECT conversation_id FROM action_ledger "
        "WHERE action_type = 'conversation_create'").fetchall()
    assert [r["conversation_id"] for r in recorded] == [cid]


def test_what_the_session_said_first_is_carried_in(runtime, db):
    """``reveal_user_commands`` writes a note per command into history.

    Run ``/config`` before saying anything and the note has nowhere to go —
    ``absorb_user_action`` has already skipped it. Carrying it in at creation
    is what lets the agent still learn you changed something out of band, and
    keeps the table agreeing with what the model was shown.
    """
    session = runtime.sessions["repl"]
    session.history.append({
        "role": "user", "author": "command_note",
        "content": "[SYSTEM NOTE] The user ran the slash command /config."})

    runtime.handle_action("repl", "send_text", {"text": "what did I change"})

    cid = session.conversation_id
    assert [(m["author"], m["content"][:24]) for m in rows(db, cid)] == [
        ("command_note", "[SYSTEM NOTE] The user r"),
        (None, "what did I change"),
    ]


# ── The one refusal that stays ───────────────────────────────────────

def test_a_fresh_install_is_still_sent_to_setup(db):
    """The ``no_llm`` half of the old guard, and it must answer *before*
    anything is created — otherwise a first-run message leaves a conversation
    behind on its way to failing."""
    rt = plain_runtime(db, config={})
    rt.get_session("repl")
    rt.active_session_key = "repl"

    out = rt.handle_action("repl", "send_text", {"text": "hello"})
    assert not out.ok
    assert out.error["code"] == "no_llm"
    assert db.list_conversations() == []
    assert rt.sessions["repl"].conversation_id is None
