"""SDK and parity coverage for the sandboxed ``/debug`` command."""

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from pipeline.database import Database
from plugins.frontends.helpers.command_registry import CommandRegistry
from plugins.plugin_discovery import discover_commands
from runtime.context import build_context
from runtime.conversation_runtime import ConversationRuntime
from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.console import CONSOLE
from sandbox.handlers.kernel import _session_get
from state_machine.conversation import ConversationState, Participant


def _context(cs=None, *, busy=False, services=None):
    sessions = {}
    if cs is not None:
        sessions["chat"] = SimpleNamespace(
            cs=cs, busy=busy, conversation_id=7)
    runtime = SimpleNamespace(
        sessions=sessions,
        is_attended=lambda _key: True,
    )
    return SimpleNamespace(
        runtime=runtime,
        session_key="chat",
        services=services or {},
    )


def _run(context):
    sandbox = Sandbox(context=context)
    try:
        return sandbox.run(
            "plugins/commands/command_debug.py",
            "DebugCommand",
            kwargs={"args": {}},
        )
    finally:
        sandbox.shutdown()


def test_session_get_details_adds_debug_data_without_changing_default():
    cs = ConversationState([
        Participant("user", "user"),
        Participant("agent", "agent"),
    ])
    service = SimpleNamespace(
        debug_flags=lambda _session: ["sample extension"])
    context = _context(cs, busy=True, services={"sample": service})

    plain = _session_get(context, {"key": "chat"})
    detailed = _session_get(context, {"key": "chat", "details": True})

    assert plain.ok and detailed.ok
    assert "debug" not in plain.data
    assert "Turn: user (user)" in detailed.data["debug"]["state"]
    assert detailed.data["debug"]["service_flags"] == ["sample extension"]
    assert detailed.data["busy"] is True


def test_session_get_details_handles_a_session_without_machine_state():
    context = _context()
    context.runtime.sessions["chat"] = SimpleNamespace(
        cs=None, busy=False, conversation_id=None)

    result = _session_get(context, {"key": "chat", "details": True})

    assert result.ok
    assert result.data["debug"] is None


def test_debug_reports_state_machine_snapshot_and_log_tail(
        tmp_path, monkeypatch):
    monkeypatch.setattr("paths.DATA_DIR", tmp_path)
    (tmp_path / "app.log").write_text(
        "01:00PM | Main         | INFO  | ok\n"
        "01:01PM | Discovery    | WARNING | Plugin registration failed: demo\n"
        "01:02PM | Main         | ERROR | Auto-load failed for 'llm': boom\n",
        encoding="utf-8",
    )
    cs = ConversationState([
        Participant("user", "user"),
        Participant("agent", "agent"),
    ])
    service = SimpleNamespace(
        debug_flags=lambda _session: ["sample extension"])

    result = _run(_context(
        cs, busy=True, services={"sample": service}))

    assert result.ok, result.error
    assert "**Conversation state**" in result.data
    assert "Turn: user (user)" in result.data
    assert f"Phase: {cs.phase}" in result.data
    assert "Participants: user(user), agent(agent)" in result.data
    assert "Session: sample extension" in result.data
    assert "Session: agent turn in progress" in result.data
    assert "Plugin registration failed: demo" in result.data
    assert "Auto-load failed for 'llm': boom" in result.data
    assert "INFO  | ok" not in result.data


def test_debug_handles_no_active_session_and_missing_log(
        tmp_path, monkeypatch):
    monkeypatch.setattr("paths.DATA_DIR", tmp_path)

    result = _run(_context())

    assert result.ok, result.error
    assert "(no active session)" in result.data
    assert f"No log file found at {tmp_path / 'app.log'}." in result.data


def test_live_repl_runs_discovered_debug_command(tmp_path, monkeypatch):
    """Console -> guest poll -> runtime -> command box -> console render."""
    monkeypatch.setattr("paths.DATA_DIR", tmp_path)
    (tmp_path / "app.log").write_text(
        "01:00PM | Main | WARNING | live warning\n", encoding="utf-8")
    db = Database(str(tmp_path / "debug-live.db"))
    holder = {}
    registry = CommandRegistry(
        lambda key=None: build_context(
            db, {}, {}, runtime=holder.get("runtime"),
            root_dir=Path.cwd(), session_key=key,
        )
    )
    discover_commands(Path.cwd(), registry, {})
    runtime = ConversationRuntime(
        db=db, services={}, config={}, commands=registry.to_callable_specs())
    holder["runtime"] = runtime

    sandbox = Sandbox()
    configure(sandbox)
    written = []
    original_claim = CONSOLE.claim

    def claim(token, source=None, writer=None):
        return original_claim(
            token, source=io.StringIO("/debug\n"), writer=written.append)

    monkeypatch.setattr(CONSOLE, "claim", claim)
    module = adapt(Path("plugins/frontends/frontend_repl.py").resolve())
    frontend_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    frontend = frontend_cls(shutdown_event=threading.Event())
    frontend.bind(runtime, registry, {})
    thread = threading.Thread(target=frontend.start, daemon=True)

    try:
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline and not any(
                "live warning" in text for text in written):
            time.sleep(0.01)
        assert any("Conversation state" in text for text in written)
        assert any("live warning" in text for text in written)
    finally:
        frontend.unbind()
        frontend.stop()
        thread.join(timeout=2)
        sandbox.shutdown()
        configure(None)
