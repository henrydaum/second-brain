"""SDK and parity coverage for the sandboxed ``/debug`` command."""

from types import SimpleNamespace

from sandbox import Sandbox
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
