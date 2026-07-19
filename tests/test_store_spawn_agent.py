"""Tests for the store spawn_agent tool, the spawn task's completion notice,
and the subagents service's end-of-turn barrier.

The tool is materialized as a package (it relatively imports the task's
channel constant); the task and service load standalone.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _store_source(rel: str) -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{rel}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


def _load_standalone(name: str, rel: str, tmp_path_factory):
    src = _store_source(rel)
    if src is None:
        pytest.skip(f"{rel} not present on a local store ref")
    path = tmp_path_factory.mktemp(name) / Path(rel).name
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def store_pkg(tmp_path_factory):
    """Materialize the tool+task+service as one package — the tool and the
    service relatively import the task, mirroring the installed-plugin tree."""
    sources = {rel: _store_source(rel) for rel in (
        "tools/tool_spawn_agent.py",
        "tasks/task_spawn_subagent.py",
        "services/service_subagents.py")}
    if any(src is None for src in sources.values()):
        pytest.skip("spawn_agent package not present on a local store ref")
    root = tmp_path_factory.mktemp("spawn_pkg")
    pkg = root / "spawn_store_pkg"
    for family in ("tools", "tasks", "services"):
        (pkg / family).mkdir(parents=True)
        (pkg / family / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for rel, src in sources.items():
        (pkg / rel).write_text(src, encoding="utf-8")
    sys.path.insert(0, str(root))
    try:
        yield "spawn_store_pkg"
    finally:
        sys.path.remove(str(root))


@pytest.fixture(scope="module")
def spawn_mod(store_pkg):
    return importlib.import_module(f"{store_pkg}.tools.tool_spawn_agent")


@pytest.fixture(scope="module")
def task_mod(store_pkg):
    return importlib.import_module(f"{store_pkg}.tasks.task_spawn_subagent")


@pytest.fixture(scope="module")
def svc_mod(store_pkg):
    return importlib.import_module(f"{store_pkg}.services.service_subagents")


class FakeDB:
    """Minimal db: lock + conn.execute(...).fetchone() driven by a script."""

    def __init__(self, statuses):
        self.lock = threading.RLock()
        self.statuses = list(statuses)  # popped one per poll; last repeats
        self.conn = self

    def execute(self, sql, params=()):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        self._row = None if status is None else ("run_x", status, None)
        if "SELECT status FROM" in sql and self._row:
            self._row = (self._row[1],)
        return self

    def fetchone(self):
        return self._row

    def get_conversation_messages(self, cid):
        return [{"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "the child's answer"}]


def _context(spawn_mod, monkeypatch, *, db=None, session=None, session_key="repl", cid=42):
    emits, markers = [], []
    monkeypatch.setattr(spawn_mod.bus, "emit", lambda ch, payload: emits.append((ch, payload)))
    monkeypatch.setattr(spawn_mod, "save_state_marker", lambda *a, **k: markers.append(a))
    sessions = {}
    if session is not None:
        sessions[session_key] = session
    runtime = SimpleNamespace(
        sessions=sessions,
        create_conversation=lambda title, **kw: cid,
    )
    context = SimpleNamespace(
        db=db if db is not None else FakeDB([None]),
        runtime=runtime, session_key=session_key, user_id=1, config={},
    )
    return context, emits, markers


def test_dependency_literals(spawn_mod):
    assert spawn_mod.dependencies_files == ['tasks/task_spawn_subagent.py', 'services/service_subagents.py']
    assert spawn_mod.dependencies_pip == []
    assert spawn_mod.SPAWN_SUBAGENT == "subagent.spawn"


def test_depth_guard(spawn_mod, monkeypatch):
    context, emits, _ = _context(spawn_mod, monkeypatch, session_key="spawn_subagent:7")
    out = spawn_mod.SpawnAgent().run(context, prompt="do work")
    assert out.success is False and "recursive" in out.error
    assert emits == []


def test_empty_prompt_and_missing_attachment(spawn_mod, monkeypatch):
    context, emits, _ = _context(spawn_mod, monkeypatch)
    assert spawn_mod.SpawnAgent().run(context, prompt="  ").success is False
    out = spawn_mod.SpawnAgent().run(context, prompt="x", attachments=["Z:/nope/missing.txt"])
    assert out.success is False and "Attachment" in out.error
    assert emits == []


def test_background_path(spawn_mod, monkeypatch):
    session = SimpleNamespace(cancel_event=None, conversation_id=7)
    context, emits, markers = _context(spawn_mod, monkeypatch, session=session)
    out = spawn_mod.SpawnAgent().run(context, prompt="research this", title="Researcher", wait=False)
    assert out.success is True and "#42" in out.llm_summary
    assert len(emits) == 1
    channel, payload = emits[0]
    assert channel == "subagent.spawn"
    assert list(payload)[0] == "conversation_id"  # find_run relies on key order
    assert payload == {"conversation_id": 42, "title": "Researcher", "prompt": "research this",
                       "attachments": [], "report_session_key": "repl",
                       "report_conversation_id": 7}
    assert session.pending_subagents[42] > time.time()
    assert markers  # state marker written for the child conversation


def test_wait_returns_child_result(spawn_mod, monkeypatch):
    monkeypatch.setattr(spawn_mod, "POLL_SECONDS", 0.01)
    db = FakeDB(["PENDING", "PENDING", "DONE"])
    session = SimpleNamespace(cancel_event=None)
    context, _, _ = _context(spawn_mod, monkeypatch, db=db, session=session)
    out = spawn_mod.SpawnAgent().run(context, prompt="answer me", wait=True)
    assert out.success is True
    assert "the child's answer" in out.llm_summary
    assert "#42" in out.llm_summary


def test_wait_reports_child_failure(spawn_mod, monkeypatch):
    monkeypatch.setattr(spawn_mod, "POLL_SECONDS", 0.01)
    db = FakeDB(["FAILED"])
    context, _, _ = _context(spawn_mod, monkeypatch, db=db, session=SimpleNamespace(cancel_event=None))
    out = spawn_mod.SpawnAgent().run(context, prompt="boom", wait=True)
    assert out.success is False and "#42" in out.error


def test_cancel_stops_wait_and_child(spawn_mod, monkeypatch):
    monkeypatch.setattr(spawn_mod, "POLL_SECONDS", 0.01)
    cancel = threading.Event()
    cancel.set()
    child_cancel = threading.Event()
    session = SimpleNamespace(cancel_event=cancel)
    context, _, _ = _context(spawn_mod, monkeypatch, db=FakeDB(["PENDING"]), session=session)
    context.runtime.sessions["spawn_subagent:42"] = SimpleNamespace(cancel_event=child_cancel)
    out = spawn_mod.SpawnAgent().run(context, prompt="slow", wait=True)
    assert out.success is False and "Cancelled" in out.error
    assert child_cancel.is_set()


def test_timeout_cancels_child_and_config_caps(spawn_mod, monkeypatch):
    monkeypatch.setattr(spawn_mod, "POLL_SECONDS", 0.01)
    session = SimpleNamespace(cancel_event=None)
    child_cancel = threading.Event()
    context, _, _ = _context(spawn_mod, monkeypatch, db=FakeDB(["PENDING"]), session=session)
    context.runtime.sessions["spawn_subagent:42"] = SimpleNamespace(cancel_event=child_cancel)
    context.config = {"subagent_timeout_seconds": 1}
    out = spawn_mod.SpawnAgent().run(context, prompt="slow", wait=True, timeout_seconds=9999)
    assert out.success is False  # hard cutoff: timeout is a failure
    assert "after 1s" in out.error and "cancelled" in out.error  # capped by the config ceiling
    assert child_cancel.is_set()
    assert 42 not in getattr(session, "pending_subagents", {})
    assert 42 in session.cancelled_subagents


def test_marker_seeds_notifications_off(spawn_mod, monkeypatch):
    session = SimpleNamespace(cancel_event=None)
    context, _, markers = _context(spawn_mod, monkeypatch, session=session)
    spawn_mod.SpawnAgent().run(context, prompt="x", wait=False)
    assert markers[0][2]["notification_mode"] == "off"


def test_task_report_queues_notice(task_mod):
    session = SimpleNamespace(lock=threading.RLock(), pending_user_messages=[])
    runtime = SimpleNamespace(sessions={"repl": session})
    payload = {"report_session_key": "repl", "title": "Researcher"}
    task_mod._report(runtime, payload, 42, ok=True, text="all done")
    assert len(session.pending_user_messages) == 1
    notice = session.pending_user_messages[0]
    assert "Researcher" in notice and "#42" in notice and "all done" in notice
    # failure variant
    task_mod._report(runtime, payload, 42, ok=False, text="exploded")
    assert "FAILED" in session.pending_user_messages[1]
    # missing session / missing key: no crash, no queue
    task_mod._report(runtime, {"report_session_key": "gone"}, 42, ok=True, text="x")
    task_mod._report(runtime, {}, 42, ok=True, text="x")
    assert len(session.pending_user_messages) == 2


def test_report_sized_like_a_real_deliverable_arrives_whole(task_mod):
    """The notice cap is a backstop against pathological dumps, not the
    delivery format: a realistic research report (~6k chars) must arrive
    uncut."""
    session = SimpleNamespace(lock=threading.RLock(), pending_user_messages=[])
    runtime = SimpleNamespace(sessions={"repl": session})
    report = "R" * 6228
    task_mod._report(runtime, {"report_session_key": "repl", "title": "X"}, 42, ok=True, text=report)
    [notice] = session.pending_user_messages
    assert report in notice
    assert "truncated" not in notice


def test_report_truncation_names_the_missing_content(task_mod):
    session = SimpleNamespace(lock=threading.RLock(), pending_user_messages=[])
    runtime = SimpleNamespace(sessions={"repl": session})
    task_mod._report(runtime, {"report_session_key": "repl", "title": "X"}, 42, ok=True, text="R" * 20000)
    [notice] = session.pending_user_messages
    assert "report truncated: 20,000 chars total" in notice
    assert "#42" in notice  # the transcript pointer survives the cut


def test_report_suppressed_for_cancelled_child(task_mod):
    session = SimpleNamespace(lock=threading.RLock(), pending_user_messages=[],
                              cancelled_subagents={42})
    runtime = SimpleNamespace(sessions={"repl": session})
    task_mod._report(runtime, {"report_session_key": "repl", "title": "X"}, 42, ok=True, text="late")
    assert session.pending_user_messages == []  # timeout notice is authoritative
    assert 42 not in session.cancelled_subagents  # entry consumed, no leak


def test_report_dropped_after_conversation_switch(task_mod):
    session = SimpleNamespace(lock=threading.RLock(), pending_user_messages=[],
                              conversation_id=99)  # session moved off conversation 7
    runtime = SimpleNamespace(sessions={"repl": session})
    payload = {"report_session_key": "repl", "title": "X", "report_conversation_id": 7}
    task_mod._report(runtime, payload, 42, ok=True, text="done")
    assert session.pending_user_messages == []
    # still on the originating conversation: delivery proceeds
    session.conversation_id = 7
    task_mod._report(runtime, payload, 42, ok=True, text="done")
    assert len(session.pending_user_messages) == 1


def test_scheduled_marker_keeps_default_notifications(task_mod, monkeypatch):
    markers = []
    monkeypatch.setattr(task_mod, "save_state_marker", lambda *a, **k: markers.append(a))
    db = SimpleNamespace(create_conversation=lambda title, **kw: 7)
    cid = task_mod._create_conversation(db, {"title": "Sched"})
    assert cid == 7
    # Scheduled subagents keep notification_mode "on" (the kernel default):
    # the chat push is their only delivery surface.
    assert "notification_mode" not in markers[0][2]


def _service(svc_mod, db):
    svc = svc_mod.SubagentsService(None)
    svc.runtime = SimpleNamespace(db=db, sessions={})
    return svc


def _ending_ctx(svc, session):
    """The end_turn envelope the kernel hands the doorman."""
    from runtime.hooks import HookContext
    return HookContext(session=session, runtime=svc.runtime, moment="end_turn")


def _ending():
    from runtime.hooks import TurnEnding
    return TurnEnding(final_text="done", reason="model_finished")


def test_barrier_registers_as_an_end_turn_doorman(svc_mod):
    from runtime.hooks import HookRegistry
    svc = _service(svc_mod, db=None)
    svc.runtime.hooks = HookRegistry()
    svc.loaded = True
    svc._register()
    assert svc._barrier in svc.runtime.hooks._hooks["end_turn"]
    assert svc._barrier not in svc.runtime.hooks._hooks["turn_finish"]


def test_barrier_abstains_without_pending(svc_mod):
    svc = _service(svc_mod, db=None)
    session = SimpleNamespace()  # no pending_subagents attr at all
    assert svc._barrier(_ending_ctx(svc, session), _ending()) is None  # must not block


def test_barrier_waits_then_redrives(svc_mod, monkeypatch):
    from runtime.hooks import Redrive
    monkeypatch.setattr(svc_mod, "POLL_SECONDS", 0.01)
    db = FakeDB(["PENDING", "DONE"])
    svc = _service(svc_mod, db)
    session = SimpleNamespace(
        pending_subagents={42: time.time() + 30},
        pending_user_messages=["[Background agent 'x' finished] ..."],
        cancel_event=None, restart_turn=False)
    verdict = svc._barrier(_ending_ctx(svc, session), _ending())
    assert isinstance(verdict, Redrive)
    assert session.pending_subagents == {}
    # The doorman answers with a verdict; the kernel's Redrive branch owns
    # the restart_turn flag.
    assert session.restart_turn is False


def test_barrier_deadline_cancels_and_reports(svc_mod, monkeypatch):
    from runtime.hooks import Redrive
    monkeypatch.setattr(svc_mod, "POLL_SECONDS", 0.01)
    child_cancel = threading.Event()
    svc = _service(svc_mod, FakeDB(["PENDING"]))
    svc.runtime.sessions["spawn_subagent:42"] = SimpleNamespace(cancel_event=child_cancel)
    session = SimpleNamespace(
        pending_subagents={42: time.time() - 1}, lock=threading.RLock(),
        pending_user_messages=[], cancel_event=None, restart_turn=False)
    verdict = svc._barrier(_ending_ctx(svc, session), _ending())
    assert child_cancel.is_set()  # hard cutoff: the child is cancelled
    assert session.pending_subagents == {}
    assert len(session.pending_user_messages) == 1
    notice = session.pending_user_messages[0]
    assert "TIMED OUT" in notice and "#42" in notice
    assert isinstance(verdict, Redrive)  # the model must learn of the cancellation
    assert 42 in session.cancelled_subagents


def test_barrier_cancel_propagates(svc_mod, monkeypatch):
    monkeypatch.setattr(svc_mod, "POLL_SECONDS", 0.01)
    cancel = threading.Event()
    cancel.set()
    child_cancel = threading.Event()
    svc = _service(svc_mod, FakeDB(["PENDING"]))
    svc.runtime.sessions["spawn_subagent:42"] = SimpleNamespace(cancel_event=child_cancel)
    session = SimpleNamespace(
        pending_subagents={42: time.time() + 30},
        pending_user_messages=[], cancel_event=cancel, restart_turn=False)
    assert svc._barrier(_ending_ctx(svc, session), _ending()) is None  # turn may end
    assert child_cancel.is_set()
    assert session.pending_subagents == {}
    assert session.restart_turn is False


def test_barrier_holds_a_real_turn_and_the_redrive_delivers(svc_mod, monkeypatch, tmp_path):
    """End-to-end through the real doorman gate: the barrier holds the turn
    while a child finishes, the redrive absorbs the report, and the user
    never sees a no-reply fallback or a mid-wait priority flip."""
    from pipeline.database import Database
    from runtime.conversation_runtime import ConversationRuntime

    class _LLM:
        context_size = 0
        is_llm_backend = True
        model_name = "fake"
        loaded = True

        def __init__(self, responses):
            self._responses = list(responses)

        def chat_with_tools(self, messages, tools, attachments=None):
            return SimpleNamespace(content=self._responses.pop(0), tool_calls=[],
                                   has_tool_calls=False, is_error=False, prompt_tokens=0)

    monkeypatch.setattr(svc_mod, "POLL_SECONDS", 0.01)
    db = Database(str(tmp_path / "barrier.db"))
    rt = ConversationRuntime(db=db, services={"llm": _LLM(["Hang tight.", "Here is the briefing."])}, config={})
    session = rt.load_conversation("s", db.create_conversation("x"))
    session.pending_subagents = {42: time.time() + 30}

    svc = svc_mod.SubagentsService(None)
    svc.runtime = SimpleNamespace(db=FakeDB(["DONE"]), sessions=rt.sessions)
    rt.hooks.add("end_turn", svc._barrier)

    priorities_at_poll = []

    def finishing_run_status(_db, _cid):
        # The child completes mid-wait: its report lands on the queue just
        # before the run flips terminal, exactly like task_spawn_subagent.
        priorities_at_poll.append(session.cs.turn_priority)
        session.pending_user_messages.append("[Background agent 'x' finished] report")
        return "DONE"

    monkeypatch.setattr(svc_mod, "_run_status", finishing_run_status)

    out = rt.handle_action("s", "send_text", "fan out")

    assert out.ok
    assert "Here is the briefing." in out.messages
    assert not any("without a reply" in m for m in out.messages)
    assert priorities_at_poll == ["agent"]  # the agent held priority through the wait
    assert session.pending_subagents == {}
    assert session.cs.turn_priority == "user"
