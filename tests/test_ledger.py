"""Tests for the action ledger (the kernel's flight recorder).

Every action flows into the append-only ``action_ledger`` table: user-side
enacts in ``ConversationRuntime._dispatch``, agent-side enacts through
``ConversationLoop._enact_logged``, and ``origin="system"`` rows for acts
outside the state machine (package installs, config saves, conversation
lifecycle ops — including refused attempts). Writes are best-effort: a ledger
failure must never break an action path.
"""

import json
from types import SimpleNamespace

from config import config_manager
from pipeline.database import DEFAULT_USER_ID, Database
from plugins.commands.helpers import package_manager

# Import the state_machine package before runtime.conversation_loop to settle
# the package-init circular import (state_machine/__init__ pulls in the loop).
from state_machine.conversation import CallableSpec, ConversationState, Participant
from state_machine.conversation_phases import BASE_PHASE

from runtime.conversation_loop import ConversationLoop
from runtime.conversation_runtime import ConversationRuntime


def _db(tmp_path):
    return Database(str(tmp_path / "ledger.db"))


# ── Database API ─────────────────────────────────────────────────────

def test_record_action_inserts_well_formed_row(tmp_path):
    db = _db(tmp_path)
    db.record_action(origin="system", action_type="config_save", ok=True,
                     name="core", args={"changed": ["max_workers"]}, duration_ms=3)

    [row] = db.get_ledger_rows()
    assert row["origin"] == "system"
    assert row["action_type"] == "config_save"
    assert row["ok"] == 1
    assert row["ts"] > 0
    assert json.loads(row["args_json"]) == {"changed": ["max_workers"]}


def test_oversized_args_stay_valid_json(tmp_path):
    db = _db(tmp_path)
    db.record_action(origin="agent_enact", action_type="call_tool", ok=True,
                     args={"blob": "x" * 50000})

    [row] = db.get_ledger_rows()
    decoded = json.loads(row["args_json"])  # truncation wrapper is still JSON
    assert decoded["_truncated_chars"] > Database.LEDGER_JSON_CAP
    assert len(row["args_json"]) < 50000


def test_unserializable_args_do_not_raise(tmp_path):
    db = _db(tmp_path)
    db.record_action(origin="user_enact", action_type="send_text", ok=True,
                     args=object())
    assert len(db.get_ledger_rows()) == 1


def test_ledger_write_failure_never_raises(tmp_path, monkeypatch):
    db = _db(tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(db, "conn", SimpleNamespace(execute=boom))
    db.record_action(origin="system", action_type="x", ok=True)  # must not raise


def test_prune_keeps_newest_rows(tmp_path):
    db = _db(tmp_path)
    for i in range(10):
        db.record_action(origin="system", action_type=f"op_{i}", ok=True)

    deleted = db.prune_action_ledger(3)

    assert deleted == 7
    assert [r["action_type"] for r in db.get_ledger_rows()] == ["op_9", "op_8", "op_7"]
    assert db.prune_action_ledger(0) == 0  # 0 = unlimited, no-op


# ── User-side enacts (the _dispatch chokepoint) ──────────────────────

def test_command_call_records_user_enact_row(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    spec = CallableSpec("ping", lambda *_: "pong")
    rt = ConversationRuntime(db=db, services={}, config={}, commands={"ping": spec})
    rt.load_conversation("s", cid)

    assert rt.handle_action("s", "call_command", {"name": "ping", "args": {}}).ok

    [row] = db.get_ledger_rows(origin="user_enact")
    assert row["action_type"] == "call_command"
    assert row["name"] == "ping"
    assert row["ok"] == 1
    assert row["session_key"] == "s"
    assert row["conversation_id"] == cid
    assert row["user_id"] == DEFAULT_USER_ID
    assert row["call_id"]
    assert row["duration_ms"] is not None


def test_failed_action_records_error_row(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db, services={}, config={})
    rt.load_conversation("s", cid)

    out = rt.handle_action("s", "call_command", {"name": "nope", "args": {}})

    assert not out.ok
    [row] = db.get_ledger_rows(origin="user_enact")
    assert row["ok"] == 0
    assert row["error_code"]


# ── Agent-side enacts (the _enact_logged gateway) ────────────────────

def _response(content):
    return SimpleNamespace(content=content, tool_calls=[], has_tool_calls=False,
                           is_error=False, prompt_tokens=0)


class _FakeLLM:
    context_size = 0

    def __init__(self, responses):
        self._responses = list(responses)

    def chat_with_tools(self, messages, tools, attachments=None):
        return self._responses.pop(0)


def test_agent_turn_records_send_text_and_end_turn(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    cs = ConversationState(
        [Participant("user", "user"), Participant("agent", "agent")],
        "agent", BASE_PHASE, {"session_key": "chat"})
    loop = ConversationLoop(_FakeLLM([_response("Hello!")]), None, {}, "prompt",
                            session_key="chat")

    loop.drive(cs, "agent", [{"role": "user", "content": "hi"}], db, cid)

    rows = db.get_ledger_rows(origin="agent_enact")
    assert [r["action_type"] for r in rows] == ["end_turn", "send_text"]  # newest first
    assert all(r["ok"] == 1 for r in rows)
    assert all(r["conversation_id"] == cid for r in rows)
    assert all(r["actor_id"] == "agent" for r in rows)


# ── System acts ──────────────────────────────────────────────────────

def test_refused_conversation_delete_is_recorded(tmp_path):
    db = _db(tmp_path)
    other = db.upsert_user("web", "intruder-target")
    cid = db.create_conversation("theirs", user_id=other)
    rt = ConversationRuntime(db=db, services={}, config={})
    rt.get_session("s")  # base user (1) session

    assert rt.delete_conversation("s", cid) is False

    [row] = db.get_ledger_rows(origin="system")
    assert row["action_type"] == "conversation_delete"
    assert row["ok"] == 0
    assert row["error_code"] == "access_denied"
    assert row["conversation_id"] == cid


def test_config_save_records_changed_key_names_only(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.setattr(config_manager, "_LEDGER_DB", db)
    path = str(tmp_path / "config.json")
    config_manager.save({}, path)  # first write: defaults
    before = len(db.get_ledger_rows(origin="system"))

    config_manager.save({"max_workers": 9}, path)

    rows = db.get_ledger_rows(origin="system")
    assert len(rows) == before + 1
    changed = json.loads(rows[0]["args_json"])["changed"]
    assert changed == ["max_workers"]
    assert "9" not in rows[0]["args_json"]  # names only, never values


def test_install_records_provenance_with_hashes(tmp_path, monkeypatch):
    db = _db(tmp_path)
    installed = tmp_path / "installed_plugins"
    monkeypatch.setattr(package_manager, "INSTALLED_PLUGINS", installed)
    content = b"dependencies_files = []\n"
    plan = package_manager.InstallPlan(
        target="tool_demo",
        files=[package_manager.PlannedFile("tools/tool_demo.py", content)],
        pip_packages=[], existing_files=[], parser_reload_needed=False,
        progress_steps=[], store_commit="abc123")
    context = SimpleNamespace(db=db, user_id=DEFAULT_USER_ID, config={},
                              runtime=None, services={})

    assert package_manager.execute_install_plan(plan, context).ok

    [row] = db.get_ledger_rows(origin="system")
    assert row["action_type"] == "package_install"
    assert row["name"] == "tool_demo"
    data = json.loads(row["data_json"])
    assert data["commit"] == "abc123"
    assert data["files"]["tools/tool_demo.py"] == package_manager._sha256(content)
