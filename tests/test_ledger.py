"""Tests for the action ledger (the kernel's flight recorder).

Every action flows into the append-only ``action_ledger`` table: user-side
enacts in ``ConversationRuntime._dispatch``, agent-side enacts through
``ConversationLoop._enact_logged``, and ``origin="system"`` rows for acts
outside the state machine (package installs, config saves, conversation
lifecycle ops — including refused attempts). Writes are best-effort: a ledger
failure must never break an action path.
"""

import json
import time
from types import SimpleNamespace

from config import config_manager
from pipeline.database import DEFAULT_USER_ID, Database
from bundled.commands.helpers import package_manager

# Import the state_machine package before runtime.conversation_loop to settle
# the package-init circular import (state_machine/__init__ pulls in the loop).
from state_machine.conversation import CallableSpec, ConversationState, Participant
from state_machine.conversation_phases import BASE_PHASE

from runtime.conversation_loop import ConversationLoop
from runtime.conversation_runtime import ConversationRuntime
from tests.support import FakeLLM, response


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


def test_retention_prunes_ledger_conversations_and_task_runs(tmp_path):
    db = _db(tmp_path)
    old = time.time() - 30 * 86400
    stale = db.create_conversation("stale")
    db.save_message(stale, "user", "old news")
    fresh = db.create_conversation("fresh")
    db.save_message(fresh, "user", "current")
    db.record_action(origin="system", action_type="ancient_op", ok=True)
    with db.lock:
        db.conn.execute("UPDATE conversations SET updated_at = ?, created_at = ? WHERE id = ?", (old, old, stale))
        db.conn.execute("UPDATE action_ledger SET ts = ?", (old,))
        db.conn.execute(
            "INSERT INTO task_runs (run_id, task_name, status, created_at, finished_at) VALUES ('r1', 't', 'SUCCESS', ?, ?)",
            (old, old))
        db.conn.commit()

    deleted = db.prune_expired(7)

    assert deleted == 3  # ledger row + task run + stale conversation (cascades its messages)
    assert db.get_conversation(stale) is None
    assert db.get_conversation_messages(stale) == []  # messages cascaded
    assert db.get_conversation(fresh) is not None
    with db.lock:
        assert db.conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0
    # The old ledger row is gone; the prune itself was recorded (deleting
    # data is an auditable act).
    assert [r["action_type"] for r in db.get_ledger_rows()] == ["retention_prune"]
    assert db.prune_expired(0) == 0  # 0 = keep forever, no-op


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

def test_agent_turn_records_send_text_and_end_turn(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    cs = ConversationState(
        [Participant("user", "user"), Participant("agent", "agent")],
        "agent", BASE_PHASE, {"session_key": "chat"})
    loop = ConversationLoop(FakeLLM([response("Hello!")]), None, {}, "prompt",
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
        pip_packages=[], existing_files=[], helper_rescan_needed=False,
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


import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox, provenance
from sandbox.bridge import adapt, configure
from sandbox.console import Console
from sandbox.guest.requests import Request, Result
from sandbox.interpreter import Execution, Interpreter
from sandbox.policy import SAFE, UNSAFE, Chain, Decision, classify

# ──────────────────────────────────────────────────────────────────────
# The flight recorder was not recording.
# ──────────────────────────────────────────────────────────────────────

def _sink(tmp_path):
    """A real database and the sandbox sink that writes to it."""
    from pipeline.database import Database
    from runtime.ledger import sandbox_sink

    db = Database(str(tmp_path / "ledger.db"))
    return db, sandbox_sink(db)


def test_an_effect_is_recorded_with_its_whole_chain(tmp_path):
    """Nothing wired ``record=``, so no Request a plugin ever made reached the
    ledger — the one place unattended work is meant to be reconstructable
    from."""
    from runtime.ledger import SANDBOX_ORIGIN

    db, record = _sink(tmp_path)
    chain = Chain(root="cron:nightly").push("task_index").push("service_web")
    record(chain, Request("fs.write", {"path": "/tmp/x"}),
           Decision(SAFE, "scratch"), Result(data=True))

    [row] = db.get_ledger_rows(origin=SANDBOX_ORIGIN)
    assert row["action_type"] == "fs.write"
    assert row["ok"] == 1
    assert "cron:nightly -> task_index -> service_web" in row["data_json"]


def test_polling_reads_are_not_recorded_but_denials_are(tmp_path):
    """A console frontend reads every poll — twenty rows a second, forever,
    burying everything worth reading. But a *denied* read is a real event."""
    from runtime.ledger import SANDBOX_ORIGIN

    db, record = _sink(tmp_path)
    chain = Chain(root="user").push("frontend_repl")

    record(chain, Request("console.read", {}), Decision(SAFE, "ok"),
           Result(data="hi"))
    record(chain, Request("fs.read", {"path": "/etc/x"}),
           Decision(UNSAFE, "no"), Result.refusal("nope"))

    rows = db.get_ledger_rows(origin=SANDBOX_ORIGIN)
    assert [r["action_type"] for r in rows] == ["fs.read"]
    assert rows[0]["error_code"] == "denied"


def test_a_coded_failure_reaches_the_ledger_as_its_code(tmp_path):
    """The sink records what the Result says, not a guess from its message."""
    from runtime.ledger import SANDBOX_ORIGIN
    from sandbox.guest.codes import ERROR_NOT_PERMITTED, ERROR_TIMEOUT

    db, record = _sink(tmp_path)
    chain = Chain(root="user").push("tool_x")

    record(chain, Request("fs.read", {"path": "/x"}), Decision(UNSAFE, "no"),
           Result.refusal("protected", code=ERROR_NOT_PERMITTED))
    record(chain, Request("net.http", {"url": "https://x/"}),
           Decision(UNSAFE, "no"),
           Result.failure("timed out", code=ERROR_TIMEOUT))
    # An uncoded failure keeps the old catch-all, so rows stay comparable.
    record(chain, Request("proc.run", {"command": "x"}),
           Decision(UNSAFE, "no"), Result.failure("something broke"))

    # get_ledger_rows is ORDER BY id DESC — newest first.
    codes = [r["error_code"] for r in db.get_ledger_rows(origin=SANDBOX_ORIGIN)]
    assert codes == ["failed", ERROR_TIMEOUT, ERROR_NOT_PERMITTED]
