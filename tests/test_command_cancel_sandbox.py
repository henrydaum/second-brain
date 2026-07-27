"""Parity and request-shape coverage for the sandboxed ``/cancel`` command."""

from types import SimpleNamespace

import pytest

from sandbox.handlers.kernel import _session_cancel
from sandbox.parity import compare


def _result(ok=True, messages=None, error=None, data=None):
    """A transport-shaped result without importing the runtime package."""
    return SimpleNamespace(
        ok=ok,
        messages=list(messages or []),
        error=error,
        data=dict(data or {}),
    )


class _Runtime:
    """The cancellation surface shared by the native and sandboxed versions."""

    def __init__(self, result, *, active=True):
        self.sessions = (
            {"repl": SimpleNamespace(
                conversation_id=1,
                busy=False,
                cs=SimpleNamespace(phase="awaiting_input"),
            )}
            if active else {}
        )
        self.result = result

    def handle_action(self, _key, action):
        assert action == "cancel"
        return self.result

    def cancel_session(self, _key):
        return self.result

    def is_attended(self, _key):
        return True


def _context(result=None, *, active=True, session_key="repl"):
    runtime = _Runtime(result or _result(messages=["Cancelled."]),
                       active=active)
    return SimpleNamespace(runtime=runtime, session_key=session_key)


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (_context(session_key=None), "No active session to cancel."),
        (_context(_result(messages=["Nothing to cancel."])),
         "Nothing to cancel."),
        (_context(_result(messages=["Cancelled."])), "Cancelled."),
        (_context(_result(ok=False, error={
            "code": "cancel_failed", "message": "Could not cancel."
        })), "Could not cancel."),
    ],
)
def test_cancel_command_matches_native_output(context, expected):
    verdict = compare(
        "plugins/commands/command_cancel.py",
        "CancelCommand",
        family="command",
        payload={},
        context=context,
    )
    assert verdict.matched, verdict.render()
    assert verdict.sandboxed["data"] == expected


def test_session_cancel_preserves_runtime_result():
    runtime = _Runtime(_result(
        ok=False,
        messages=["Stopped one operation."],
        error={"code": "partial", "message": "One operation remained."},
        data={"cancelled": 1},
    ))
    result = _session_cancel(
        SimpleNamespace(runtime=runtime, session_key="repl"), {})

    assert result.ok
    assert result.data == {
        "ok": False,
        "messages": ["Stopped one operation."],
        "error": {"code": "partial", "message": "One operation remained."},
        "data": {"cancelled": 1},
    }
