"""Who actually wrote a row.

``role`` cannot answer it. ``'system'`` already means a state-machine or
compaction marker, so six kernel mechanisms write ``role='user'`` rows — a
cancel notice, a doorman's note, the compaction bridge, the emergency-truncation
bridge, the ``reveal_user_commands`` note, and any plugin's ``conv.append``.
Every one of them read, to a client and to the store's memory curator, exactly
like something the person typed.

``author`` is the answer, and its whole design is in the NULL: absent means
"this row is what its role says", which is true of every row written before the
column existed, so nothing is backfilled and nothing is rewritten.
"""

import pytest

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)

from runtime.conversation_loop import _for_provider
from state_machine.serialization import messages_to_history, save_history_message


@pytest.fixture
def db(tmp_path):
    """A real database, since the point is what a column does."""
    from pipeline.database import Database

    return Database(str(tmp_path / "test.db"))


# ──────────────────────────────────────────────────────────────────────
# The column.
# ──────────────────────────────────────────────────────────────────────

def test_an_ordinary_row_has_no_author(db):
    """NULL, not "". The absence *is* the meaning — it says the role is the
    whole answer — so nothing normalizes it the way ``attachments`` normalizes
    to ``[]``. A row somebody typed and a row written before this column
    existed have to read identically, because they are the same thing."""
    cid = db.create_conversation("Main")
    db.save_message(cid, "user", "what is this?")

    assert db.get_conversation_messages(cid)[0]["author"] is None


def test_a_synthesized_row_names_who_wrote_it(db):
    cid = db.create_conversation("Main")
    db.save_message(cid, "user", "[cancelled]", author="cancel_notice")

    assert db.get_conversation_messages(cid)[0]["author"] == "cancel_notice"


def test_the_author_survives_a_whole_conversation_rewrite(db):
    """``iterate_agent_turn`` rewrites every row from the live history.

    The same trap ``attachments`` fell into: a key dropped here survives until
    the next *background* turn and then does not, so it needs a subagent or a
    scheduled task to reproduce and looks perfect until then.
    """
    cid = db.create_conversation("Main")
    db.replace_conversation_messages(cid, [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "[cancelled]", "author": "cancel_notice"},
        {"role": "assistant", "content": "Right."},
    ])

    assert [r["author"] for r in db.get_conversation_messages(cid)] == [
        None, "cancel_notice", None]


def test_the_author_round_trips_through_history(db):
    """Row -> provider history -> row, the twin of the ``attachments`` pair."""
    cid = db.create_conversation("Main")
    save_history_message(db, cid, {
        "role": "user", "content": "[cancelled]", "author": "cancel_notice"})
    save_history_message(db, cid, {"role": "user", "content": "carry on"})

    history = messages_to_history(db.get_conversation_messages(cid))

    assert history == [
        {"role": "user", "content": "[cancelled]", "author": "cancel_notice"},
        {"role": "user", "content": "carry on"},
    ]


# ──────────────────────────────────────────────────────────────────────
# The leak.
# ──────────────────────────────────────────────────────────────────────

def test_no_provider_ever_sees_the_author_key():
    """``messages`` goes to a provider API verbatim, so a field no schema knows
    is either rejected outright or silently believed.

    Checked with **no attachments**, which is the whole point: ``_for_provider``
    returned the message untouched on that path, and every authored row — a
    cancel notice, a doorman's note — carries no files at all. So the one
    shortcut in the function was the one case the key always took.
    """
    rendered = _for_provider({"role": "user", "content": "[cancelled]",
                              "author": "cancel_notice"})

    assert "author" not in rendered
    assert rendered["content"] == "[cancelled]"


def test_a_plain_row_is_still_handed_straight_through():
    """The shortcut still exists for the ordinary case; it just asks about both
    keys now. Identity, not merely equality — this runs for every row of every
    model call, and copying each one would be a real cost for no gain."""
    msg = {"role": "user", "content": "hello"}

    assert _for_provider(msg) is msg


# ──────────────────────────────────────────────────────────────────────
# The six who tag themselves.
# ──────────────────────────────────────────────────────────────────────

def test_the_compaction_bridge_is_authored():
    """Re-derived on every load past a compaction marker, so it is never a row
    anybody wrote — and the model is told a summary either way."""
    rows = [{"role": "system", "content":
             '{"__second_brain_compaction__": true, "summary": "Earlier.", '
             '"tail_count": 2}'},
            {"role": "user", "content": "and then?"}]

    history = messages_to_history(rows)

    assert [m.get("author") for m in history] == [
        "compaction", "compaction", None]


def test_a_cancel_notice_is_not_something_the_person_said():
    """The row exists so the model stops offering to wait for cancelled work.
    It is addressed *to* the model in the person's slot, which is exactly why
    it needs to say it is not the person."""
    from runtime.conversation_loop import ConversationLoop

    recorded = []
    loop = ConversationLoop.__new__(ConversationLoop)
    loop._record = lambda msg, *a, **k: recorded.append(msg)
    ConversationLoop._record_cancellation(loop, [], [], None, None)

    assert recorded[0]["role"] == "user"
    assert recorded[0]["author"] == "cancel_notice"
    assert recorded[0]["content"] == ConversationLoop.CANCEL_NOTICE


# ──────────────────────────────────────────────────────────────────────
# What it is for.
# ──────────────────────────────────────────────────────────────────────

def test_a_conversation_is_not_titled_after_a_cancel_notice():
    """The visible bug the column was built for.

    ``latest_user_text`` feeds the new conversation's title. Reading ``role``
    alone titled conversations "[The user cancelled the previous turn…]" after
    a ``/cancel``, or "[SYSTEM NOTE] The user ran /config" with
    ``reveal_user_commands`` on — and nothing anywhere said so.
    """
    from types import SimpleNamespace

    from runtime.dispatch import latest_user_text

    session = SimpleNamespace(history=[
        {"role": "user", "content": "plan my week"},
        {"role": "assistant", "content": "Sure."},
        {"role": "user", "content": "[cancelled]", "author": "cancel_notice"},
    ])

    assert latest_user_text(session) == "plan my week"


def test_conv_append_stamps_the_plugin_the_kernel_saw(db):
    """The site the vocabulary is deliberately open for.

    ``conv.append`` lets any plugin write a ``role='user'`` row, and no frozen
    list of kernel reasons can name one. Read off the provenance chain for the
    same reason a notification's ``source`` is: a plugin allowed to state its
    own authorship could leave it blank and write a row indistinguishable from
    something the person typed, which is the exact forgery the column exists to
    prevent.
    """
    from types import SimpleNamespace

    from sandbox import provenance
    from sandbox.handlers.kernel import _conv_append
    from sandbox.policy import Chain

    cid = db.create_conversation("Main")
    ctx = SimpleNamespace(db=db, user_id=None, session_key=None)

    chain = Chain(root="user").push("service_x").push("tool_y")
    with provenance.serving(chain, None, None):
        assert _conv_append(ctx, {"id": cid, "content": "noted"}).ok

    row = db.get_conversation_messages(cid)[0]
    assert row["role"] == "user"
    assert row["author"] == "tool_y"


def test_with_nothing_but_synthesized_rows_there_is_no_user_text():
    """Falling back to the newest authored row would be worse than empty: the
    caller would title a conversation after the kernel's own words while
    believing it had the person's."""
    from types import SimpleNamespace

    from runtime.dispatch import latest_user_text

    session = SimpleNamespace(history=[
        {"role": "user", "content": "[cancelled]", "author": "cancel_notice"},
    ])

    assert latest_user_text(session) == ""
