"""SDK and behavior coverage for the sandboxed ``/clear`` command."""

from types import SimpleNamespace

import state_machine  # noqa: F401  (break the runtime import cycle)

from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime
from sandbox.facade import Sandbox
from sandbox.guest.requests import CONV_CLEAR, Request
from sandbox.handlers.kernel import _conv_clear
from sandbox.policy import Chain, SAFE, classify


def _rig(tmp_path):
    db = Database(str(tmp_path / "clear.db"))
    cid = db.create_conversation("Notes", user_id=9)
    db.save_message(cid, "user", "remove me")
    db.save_message(cid, "assistant", "and me")
    runtime = ConversationRuntime(db=db, services={}, config={})
    runtime.set_session_user("repl", 9)
    runtime.load_conversation("repl", cid)
    context = SimpleNamespace(
        runtime=runtime,
        db=db,
        session_key="repl",
        user_id=9,
    )
    return db, runtime, cid, context


def _run(context):
    sandbox = Sandbox(context=context, approve=lambda *_: True)
    try:
        return sandbox.run(
            "plugins/commands/command_clear.py",
            "ClearCommand",
            kwargs={"args": {}},
        )
    finally:
        sandbox.shutdown()


def test_clear_command_preserves_database_session_and_identity(tmp_path):
    db, runtime, cid, context = _rig(tmp_path)

    result = _run(context)

    assert result.ok, result.error
    assert result.data == "Conversation cleared."
    assert db.get_conversation_messages(cid) == []
    assert db.get_conversation(cid)["title"] == "Notes (cleared)"
    session = runtime.sessions["repl"]
    assert session.conversation_id == cid
    assert session.user_id == 9
    assert session.history == []

    repeated = _run(context)
    assert repeated.ok
    assert db.get_conversation(cid)["title"] == "Notes (cleared)"


def test_clear_command_keeps_empty_state_messages(tmp_path):
    db = Database(str(tmp_path / "empty.db"))
    runtime = ConversationRuntime(db=db, services={}, config={})

    no_session = _run(SimpleNamespace(
        runtime=runtime, db=db, session_key="missing", user_id=1))
    assert no_session.data == "No active session."

    runtime.set_session_user("repl", 1)
    no_conversation = _run(SimpleNamespace(
        runtime=runtime, db=db, session_key="repl", user_id=1))
    assert no_conversation.data == "No conversation loaded."


def test_conv_clear_is_owned_and_scoped(tmp_path):
    db, runtime, cid, context = _rig(tmp_path)
    request = Request(CONV_CLEAR, {"id": cid})
    assert not request.read_only
    assert classify(request, Chain().push("clear")).level == SAFE

    other_user = SimpleNamespace(
        runtime=runtime, db=db, session_key="other", user_id=1)
    refused = _conv_clear(other_user, {"id": cid})
    assert refused.denied
    assert db.get_conversation_messages(cid)
