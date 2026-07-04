"""Tests for the ``reveal_user_commands`` kernel setting: completed slash
commands are mirrored into provider history as a bracketed note (name +
argument names only), so the agent knows the user changed state out-of-band.
"""

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)

from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime
from state_machine.conversation import CallableSpec, FormStep


def _runtime(tmp_path, config, commands):
    db = Database(str(tmp_path / "reveal.db"))
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db, services={}, config=config, commands=commands)
    rt.load_conversation("s", cid)
    return rt, db, cid


def _notes(history):
    return [m for m in history if "[SYSTEM NOTE]" in (m.get("content") or "")]


def test_off_by_default_adds_nothing(tmp_path):
    rt, db, cid = _runtime(tmp_path, {}, {"ping": CallableSpec("ping", lambda *_: "pong")})

    assert rt.handle_action("s", "call_command", {"name": "ping", "args": {}}).ok

    assert _notes(rt.sessions["s"].history) == []
    assert all("[SYSTEM NOTE]" not in m["content"] for m in db.get_conversation_messages(cid))


def test_reveals_name_and_field_names_only(tmp_path):
    spec = CallableSpec("config", lambda *_: "saved")
    rt, db, cid = _runtime(tmp_path, {"reveal_user_commands": True}, {"config": spec})

    out = rt.handle_action("s", "call_command", {
        "name": "config", "args": {"key": "brave_search_api_key", "value": "sk-SECRET"}})

    assert out.ok
    [note] = _notes(rt.sessions["s"].history)
    assert note["role"] == "user"
    assert "/config" in note["content"]
    assert "key" in note["content"] and "value" in note["content"]
    assert "sk-SECRET" not in note["content"]  # arg values never leak
    # Persisted so the agent sees it after restart/compaction reload too.
    assert any("[SYSTEM NOTE]" in m["content"] for m in db.get_conversation_messages(cid))


def test_form_start_is_silent_and_completion_is_noted(tmp_path):
    spec = CallableSpec("greet", lambda cs, actor, args: f"hi {args['who']}",
                        form=[FormStep("who", "Who?")])
    rt, _db, _cid = _runtime(tmp_path, {"reveal_user_commands": True}, {"greet": spec})

    out = rt.handle_action("s", "call_command", {"name": "greet", "args": {}})
    assert out.ok
    assert _notes(rt.sessions["s"].history) == []  # form just opened, nothing ran

    out = rt.handle_action("s", "submit_form_text", {"text": "Henry"})
    assert out.ok
    [note] = _notes(rt.sessions["s"].history)
    assert "/greet" in note["content"] and "who" in note["content"]
    assert "Henry" not in note["content"]


def test_failed_command_is_not_noted(tmp_path):
    rt, _db, _cid = _runtime(tmp_path, {"reveal_user_commands": True}, {})

    out = rt.handle_action("s", "call_command", {"name": "nope", "args": {}})

    assert not out.ok
    assert _notes(rt.sessions["s"].history) == []
