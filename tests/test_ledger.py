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


def test_streamed_tokens_are_not_recorded(tmp_path):
    """A streaming backend sends one ``llm.delta`` per token, and each row was
    an INSERT and a commit under the database's one lock — so a single reply
    wrote thousands of them and serialized the whole database against the
    model. It buys nothing: the call and the enact are already recorded, and
    this is only the model's own output kept a character at a time.

    Asserted alongside a *refused* delta, because the drop is about volume and
    never about the type: anything the kernel refused is still an event."""
    from runtime.ledger import SANDBOX_ORIGIN

    db, record = _sink(tmp_path)
    chain = Chain(root="user").push("llm_litellm")

    for word in ("Hello", " there", "!"):
        record(chain, Request("llm.delta", {"token": "t", "text": word}),
               Decision(SAFE, "llm.delta"), Result(data=None))
    record(chain, Request("llm.delta", {"token": "stolen", "text": "x"}),
           Decision(UNSAFE, "no sink"), Result.refusal("no such token"))

    rows = db.get_ledger_rows(origin=SANDBOX_ORIGIN)
    assert [r["ok"] for r in rows] == [0]


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


# ──────────────────────────────────────────────────────────────────────
# A row says whose work it was.
#
# The columns existed from the beginning and the sandbox origin — the only
# per-effect record the system has — filled none of them. That is the shape of
# bug this block exists for: nothing raised, nothing looked wrong, and every
# query that asked "what did this conversation do" answered nothing.
# ──────────────────────────────────────────────────────────────────────

def _ctx(session_key=None, conversation_id=None, user_id=None):
    """A context shaped like the one an execution's handlers answer from."""
    sessions = ({session_key: SimpleNamespace(conversation_id=conversation_id)}
                if session_key else {})
    return SimpleNamespace(session_key=session_key, user_id=user_id,
                           runtime=SimpleNamespace(sessions=sessions))


def test_an_effect_records_the_conversation_it_happened_in(tmp_path):
    """``idx_ledger_conv`` is an index on ``(conversation_id, id)`` and the
    sandbox rows left that column NULL, so it was dead weight for the one
    origin worth seeking on."""
    from runtime.ledger import SANDBOX_ORIGIN

    db, record = _sink(tmp_path)
    record(Chain(root="http:main").push("tool_write"),
           Request("fs.write", {"path": "/w/notes.md", "data": "hi"}),
           Decision(SAFE, "workspace"), Result(data={"path": "/w/notes.md",
                                                     "bytes": 2}),
           _ctx("http:main", conversation_id=7, user_id=3))

    [row] = db.get_ledger_rows(conversation_id=7)
    assert row["origin"] == SANDBOX_ORIGIN
    assert (row["session_key"], row["user_id"]) == ("http:main", 3)


def test_work_belonging_to_no_conversation_still_says_so(tmp_path):
    """A resident service polling on its own initiative is handed a context
    with no session. Inventing a conversation for it would put a service's
    housekeeping in somebody's transcript."""
    db, record = _sink(tmp_path)
    record(Chain(root="service:timekeeper"), Request("fs.write", {"path": "/w/j"}),
           Decision(SAFE, "own"), Result(data=True), _ctx())

    [row] = db.get_ledger_rows()
    assert row["conversation_id"] is None and row["session_key"] is None


def test_a_sink_that_does_not_care_about_context_still_works(tmp_path):
    """The trailing context is optional. Four-argument sinks predate it and
    are the shape every test double in the suite is written in."""
    db, record = _sink(tmp_path)
    record(Chain(root="user").push("tool_x"), Request("fs.delete", {"path": "/w/x"}),
           Decision(SAFE, "scratch"), Result(data=True))

    assert db.get_ledger_rows()[0]["conversation_id"] is None


def test_the_path_survives_an_args_blob_too_big_to_keep(tmp_path):
    """``args_json`` caps at ``LEDGER_JSON_CAP`` and past it the *object* is
    replaced by a head/tail wrapper — and the argument that blows the cap is
    the file's own contents. So the rows whose paths are hardest to recover
    were exactly the big edits somebody most wants to see."""
    db, record = _sink(tmp_path)
    record(Chain(root="user").push("tool_write"),
           Request("fs.write", {"path": "/w/big.md", "data": "x" * 9000}),
           Decision(SAFE, "workspace"),
           Result(data={"path": "/w/big.md", "bytes": 9000}),
           _ctx("s", conversation_id=1))

    [row] = db.get_ledger_rows()
    assert json.loads(row["args_json"])["_truncated_chars"]  # the cap bit
    data = json.loads(row["data_json"])
    assert data["paths"] == ["/w/big.md"] and data["bytes"] == 9000


def test_a_move_records_both_ends_and_a_refusal_records_no_size(tmp_path):
    """``fs.move`` is the one file Request with two paths, and a refusal has
    no answer to take a byte count from."""
    db, record = _sink(tmp_path)
    record(Chain(root="user").push("tool_x"),
           Request("fs.move", {"src": "/w/a", "dst": "/w/b"}),
           Decision(SAFE, "both writable"), Result(data={"src": "/w/a", "dst": "/w/b"}))
    record(Chain(root="user").push("tool_x"),
           Request("fs.write", {"path": "/etc/passwd", "data": "no"}),
           Decision(UNSAFE, "protected"), Result.refusal("denied"))

    denied, moved = db.get_ledger_rows()
    assert json.loads(moved["data_json"])["paths"] == ["/w/a", "/w/b"]
    assert json.loads(denied["data_json"])["paths"] == ["/etc/passwd"]
    assert "bytes" not in json.loads(denied["data_json"])


def test_a_request_that_touches_no_file_records_no_paths(tmp_path):
    """Only the four filesystem Requests are lifted. Everything else keeps the
    three keys the sink has always written."""
    db, record = _sink(tmp_path)
    record(Chain(root="user").push("tool_x"),
           Request("net.http", {"url": "https://example.com/"}),
           Decision(UNSAFE, "egress"), Result(data={"status": 200}))

    assert set(json.loads(db.get_ledger_rows()[0]["data_json"])) == {
        "chain", "level", "reason"}


# ──────────────────────────────────────────────────────────────────────
# Reading it targeted.
#
# "Query the ledger targeted, never linearly" was advice with nothing to
# target *with*: the Request took a row limit and no filters at all, while the
# database helper underneath had supported three since it was written.
# ──────────────────────────────────────────────────────────────────────

def _ledger_ctx(db, session_key="s", allowed=True):
    """A context ``_ledger_read`` can answer from, owning conversations or not."""
    return SimpleNamespace(
        db=db, session_key=session_key, user_id=DEFAULT_USER_ID,
        runtime=SimpleNamespace(sessions={},
                                assert_conversation_access=lambda k, cid: allowed))


def _seed(db):
    for cid, action, path in [(1, "fs.write", "/a"), (1, "fs.delete", "/b"),
                              (1, "net.http", None), (2, "fs.write", "/c")]:
        db.record_action(origin="sandbox", action_type=action, ok=True,
                         conversation_id=cid, data={"paths": [path] if path else []})


def test_ledger_read_narrows_by_conversation_and_type(tmp_path):
    from sandbox.handlers.kernel import _ledger_read

    db = _db(tmp_path)
    _seed(db)
    result = _ledger_read(_ledger_ctx(db), {
        "conversation_id": 1, "action_types": ["fs.write", "fs.delete"]})

    assert [r["action_type"] for r in result.data] == ["fs.delete", "fs.write"]


def test_ledger_read_takes_only_what_followed_a_row(tmp_path):
    """The incremental form: a reader holding rows up to N asks for the rest
    rather than re-reading the conversation every time something happens."""
    from sandbox.handlers.kernel import _ledger_read

    db = _db(tmp_path)
    _seed(db)
    ctx = _ledger_ctx(db)
    first = _ledger_read(ctx, {"conversation_id": 1})
    cursor = min(r["id"] for r in first.data)

    later = _ledger_read(ctx, {"conversation_id": 1, "since_id": cursor})
    assert [r["id"] for r in later.data] == sorted(
        (r["id"] for r in first.data if r["id"] > cursor), reverse=True)


def test_ledger_read_refuses_a_conversation_the_user_does_not_own(tmp_path):
    """The rows carry ``user_id`` and ``conversation_id`` now, so naming one is
    a question about somebody's data and is answered by the same check
    ``conv.read`` makes."""
    from sandbox.handlers.kernel import _ledger_read

    db = _db(tmp_path)
    _seed(db)
    refused = _ledger_read(_ledger_ctx(db, allowed=False), {"conversation_id": 1})

    assert refused.denied and "not available" in refused.error


def test_ledger_read_still_answers_bare(tmp_path):
    """No filters is the call every existing caller makes."""
    from sandbox.handlers.kernel import _ledger_read

    db = _db(tmp_path)
    _seed(db)
    assert len(_ledger_read(_ledger_ctx(db), {}).data) == 4


def test_an_action_type_cannot_smuggle_sql(tmp_path):
    """The IN clause is built from the *count* of values; the values stay
    bound."""
    from sandbox.handlers.kernel import _ledger_read

    db = _db(tmp_path)
    _seed(db)
    result = _ledger_read(_ledger_ctx(db),
                          {"action_types": ["fs.write') OR 1=1 --"]})
    assert result.data == []


# ──────────────────────────────────────────────────────────────────────
# Files the agent put in front of you.
#
# These reach a frontend as an ``attachments`` render frame and were then gone:
# ``conversation_messages`` has no metadata column, so a reload could not tell
# that a turn had shown anything at all.
# ──────────────────────────────────────────────────────────────────────

def test_a_tool_that_showed_files_records_which(tmp_path):
    from plugins.native.tool import ToolResult

    db = _db(tmp_path)
    cid = db.create_conversation("x")
    loop = ConversationLoop(FakeLLM([]), None, {}, "prompt", session_key="chat")
    loop._active_db, loop._active_conversation_id = db, cid

    shown = ToolResult(success=True, attachment_paths=["/w/chart.png", "/w/report.md"])
    loop._record_ledger("call_tool", {"name": "show_files"}, "agent",
                        SimpleNamespace(ok=True, error=None,
                                        data={"result": shown}), None, time.perf_counter())

    [row] = db.get_ledger_rows(origin="agent_enact")
    assert json.loads(row["data_json"])["attachments"] == [
        "/w/chart.png", "/w/report.md"]


def test_an_act_that_showed_nothing_records_no_attachments_key(tmp_path):
    """Every action type flows through this gateway, and most carry no
    ``ToolResult`` at all — an ``end_turn``'s data is not even a dict."""
    from runtime.conversation_loop import _attachment_paths

    assert _attachment_paths(SimpleNamespace(data={"result": None})) == []
    assert _attachment_paths(SimpleNamespace(data="just a string")) == []
    assert _attachment_paths(SimpleNamespace(data=None)) == []
    assert _attachment_paths(None) == []


# ──────────────────────────────────────────────────────────────────────
# Whether the tool actually worked.
#
# The enact answers "was the action performed", which for a tool call is true
# however the tool went — so every tool failure in the table's history was
# recorded ok=1. Across 39k rows not one of the eighteen tools had ever failed
# according to the ledger, which made the flight recorder blind to the single
# most common thing worth reconstructing after the fact.
# ──────────────────────────────────────────────────────────────────────

def _enact_tool(db, cid, tool_result):
    loop = ConversationLoop(FakeLLM([]), None, {}, "prompt", session_key="chat")
    loop._active_db, loop._active_conversation_id = db, cid
    loop._record_ledger("call_tool", {"name": "edit_file"}, "agent",
                        SimpleNamespace(ok=True, error=None,
                                        data={"result": tool_result}),
                        None, time.perf_counter())
    [row] = db.get_ledger_rows(origin="agent_enact")
    return row


def test_a_failed_tool_call_is_recorded_as_failed(tmp_path):
    from plugins.native.tool import ToolResult

    db = _db(tmp_path)
    row = _enact_tool(db, db.create_conversation("x"),
                      ToolResult(success=False, error="old_text was not found."))

    assert row["ok"] == 0
    assert row["error_message"] == "old_text was not found."


def test_a_successful_tool_call_is_still_recorded_as_ok(tmp_path):
    from plugins.native.tool import ToolResult

    db = _db(tmp_path)
    row = _enact_tool(db, db.create_conversation("x"),
                      ToolResult(success=True, llm_summary="Replaced text."))

    assert row["ok"] == 1
    assert not row["error_message"]


def test_an_action_with_no_tool_underneath_keeps_the_enacts_own_verdict(tmp_path):
    """``end_turn`` and friends carry no ``ToolResult``, so there is nothing to
    narrow with and the row must read exactly as it always did."""
    from runtime.conversation_loop import _tool_outcome

    assert _tool_outcome(SimpleNamespace(data={"result": None})) is None
    assert _tool_outcome(SimpleNamespace(data="just a string")) is None
    assert _tool_outcome(SimpleNamespace(data=None)) is None
    assert _tool_outcome(None) is None

    db = _db(tmp_path)
    cid = db.create_conversation("x")
    loop = ConversationLoop(FakeLLM([]), None, {}, "prompt", session_key="chat")
    loop._active_db, loop._active_conversation_id = db, cid
    loop._record_ledger("end_turn", None, "agent",
                        SimpleNamespace(ok=True, error=None, data=None),
                        None, time.perf_counter())

    [row] = db.get_ledger_rows(origin="agent_enact")
    assert row["ok"] == 1


def test_an_enact_that_itself_failed_keeps_its_own_reason(tmp_path):
    """The inner verdict only ever narrows. A dispatch that failed has a
    reason about the dispatch, which must not be overwritten by a tool's."""
    from plugins.native.tool import ToolResult
    from runtime.ledger import record_enact

    db = _db(tmp_path)
    cid = db.create_conversation("x")
    record_enact(db, origin="agent_enact", session_key="chat",
                 conversation_id=cid, user_id=None, actor_id="agent",
                 action_type="call_tool", content={"name": "edit_file"},
                 result=SimpleNamespace(
                     ok=False,
                     error=SimpleNamespace(code="denied", message="refused")),
                 outcome=_tool_ok_pair(ToolResult(success=True)))

    [row] = db.get_ledger_rows(origin="agent_enact")
    assert row["ok"] == 0
    assert row["error_message"] == "refused"


def _tool_ok_pair(tool_result):
    return bool(tool_result.success), str(tool_result.error or "")


def test_plugin_code_can_finally_see_its_own_sandbox_rows(tmp_path):
    """``my_action_ledger`` scopes on ``user_id``, which every sandbox row left
    NULL — so the virtual table built to let a plugin read the ledger hid from
    it the whole origin describing what plugins do."""
    from sandbox.users import scope_sql

    db, record = _sink(tmp_path)
    record(Chain(root="http:main").push("tool_x"),
           Request("fs.write", {"path": "/w/a"}), Decision(SAFE, "workspace"),
           Result(data=True), _ctx("http:main", conversation_id=4, user_id=9))

    sql, params = scope_sql("SELECT action_type FROM my_action_ledger", [], 9)
    assert [r["action_type"] for r in db.conn.execute(sql, params)] == ["fs.write"]


def test_a_plugins_own_note_reaches_the_table(tmp_path):
    """``ledger.record`` passed ``data_json=`` to a function whose parameter is
    ``data=``. Every call raised ``TypeError`` at binding, the handler's own
    guard swallowed it, and the Request answered ``False`` — which is
    indistinguishable from a busy database, so a best-effort write that had
    never once succeeded looked entirely healthy."""
    from sandbox.handlers.kernel import _ledger_record

    db = _db(tmp_path)
    ctx = SimpleNamespace(
        db=db, session_key="s", user_id=DEFAULT_USER_ID,
        runtime=SimpleNamespace(sessions={"s": SimpleNamespace(conversation_id=5)}))

    assert _ledger_record(ctx, {"action": "reindexed", "data": {"files": 12}}).data

    [row] = db.get_ledger_rows(origin="sandbox")
    assert row["action_type"] == "reindexed"
    assert json.loads(row["data_json"]) == {"files": 12}
    # Same identity the sink resolves, so a plugin's own note lands in the
    # conversation the rows written *about* it landed in.
    assert row["conversation_id"] == 5 and row["session_key"] == "s"


def test_an_execution_with_no_context_of_its_own_still_says_whose_it_was(tmp_path):
    """The sink must read the context the *handler* answered from, which is
    ``Interpreter._context_for`` — not ``execution.context``.

    Most executions carry no context of their own and fall back to the
    interpreter's, so reading the attribute directly saw ``None`` for exactly
    the ordinary case. Every unit test here passes a context explicitly, so
    this went green while a real box recorded rows belonging to nobody; it took
    driving an actual write through an actual `Sandbox` to see it.
    """
    from sandbox.interpreter import Execution, Interpreter

    db, record = _sink(tmp_path)
    interp = Interpreter(record=record, context=_ctx("http:main", 12, 4))
    try:
        execution = Execution(name="tool_x", chain=Chain(root="http:main"))
        assert execution.context is None  # the case that was silently wrong
        interp._settle(execution, Request("fs.write", {"path": "/w/a"}),
                       Decision(SAFE, "workspace"), Result(data=True))
    finally:
        interp.shutdown()

    [row] = db.get_ledger_rows()
    assert (row["conversation_id"], row["user_id"]) == (12, 4)


def test_a_shell_deletion_is_recorded_as_the_files_it_removed(tmp_path):
    """``rm -rf build`` changes files as surely as ``fs.delete`` does; the
    paths are just inside ``argv`` where nothing was reading them."""
    db, record = _sink(tmp_path)
    record(Chain(root="user").push("tool_run_command"),
           Request("proc.run", {"argv": ["rm", "-rf", "build"], "cwd": "/srv"}),
           Decision(UNSAFE, "shell"), Result(data={"code": 0, "stdout": ""}))

    data = json.loads(db.get_ledger_rows()[0]["data_json"])
    assert data["deleted"] == data["paths"]
    assert data["paths"][0].endswith("build")
    # Weaker than a path the kernel serviced, and says so.
    assert data["via"] == "shell" and data["command"] == "rm -rf build"


def test_a_command_that_failed_claims_no_deletion(tmp_path):
    """A non-zero exit deleted nothing. A row saying otherwise is worse than
    a missing one — the drawer would show a file gone that is still there."""
    db, record = _sink(tmp_path)
    record(Chain(root="user").push("tool_run_command"),
           Request("proc.run", {"argv": ["rm", "-rf", "build"]}),
           Decision(UNSAFE, "shell"), Result(data={"code": 1, "stderr": "no such"}))

    assert "paths" not in json.loads(db.get_ledger_rows()[0]["data_json"])


def test_an_ordinary_command_still_records_no_files(tmp_path):
    """Most commands say nothing about files, and the row stays as it was."""
    db, record = _sink(tmp_path)
    record(Chain(root="user").push("tool_run_command"),
           Request("proc.run", {"argv": ["npm", "install"]}),
           Decision(UNSAFE, "shell"), Result(data={"code": 0}))

    assert set(json.loads(db.get_ledger_rows()[0]["data_json"])) == {
        "chain", "level", "reason"}
