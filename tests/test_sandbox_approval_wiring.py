"""The sandbox must actually have somebody to ask.

A sandbox with no approver refuses every unsafe Request outright. That is the
correct default — there is genuinely nobody to put a dialog in front of — but
it was also the permanent state in production, because nothing ever wired the
runtime in. Every approval-gated capability failed with the *policy's* refusal
reason instead of prompting: ``/packages update`` reported "plugin.update
changes what the system can do" and no dialog ever appeared.

Boot order is what makes this easy to get wrong. Plugins are discovered and
loaded long before the conversation runtime exists, so the wiring cannot
happen when the sandbox is constructed.
"""

from types import SimpleNamespace

import pytest

from sandbox.facade import Sandbox
from sandbox.guest.requests import PLUGIN_UPDATE, Request
from sandbox.interpreter import Execution
from sandbox.policy import Chain


class _Pending:
    """A dialog the user has already answered."""

    def __init__(self, approved: bool):
        self.id = "p1"
        self.approved = approved
        self.metadata: dict = {}

    def wait(self, timeout=None) -> bool:
        return True


def _runtime(approved: bool, asked: list):
    """A runtime that renders an approval dialog and gets an answer."""
    def request_input(key, title, body, **kwargs):
        asked.append((title, body))
        return _Pending(approved)

    return SimpleNamespace(
        active_session_key="repl",
        sessions={"repl": SimpleNamespace(attended=True)},
        hooks=None,
        user_setting=lambda key, name: [],
        is_attended=lambda key: True,
        request_input=request_input,
        answer_request=lambda *a, **k: None,
    )


def _gate(sandbox, request):
    """Push one Request through the real gate and return its Result.

    The read blocks because the answer is now *asynchronous*: the gate hands
    an unsafe Request to an approval worker and returns immediately, so the
    settle lands on another thread.
    """
    execution = Execution(name="packages",
                          chain=Chain(root="user").push("packages"))
    sandbox.interpreter._gate_one(execution, request)
    return execution.inbox.get(timeout=5)


def test_asking_the_user_does_not_block_the_gate():
    """The deadlock, stated.

    The gate is the single ordering point for every Request in the process,
    including the ones the frontend makes to *draw* the dialog and to read the
    answer. Asking on the gate thread therefore made the question unaskable:
    ``/packages install`` showed no dialog, ignored ``y``, and froze the whole
    app until the wait expired.

    So while one dialog is open, an unrelated Request must still be classified
    and served.
    """
    import threading

    from sandbox.guest.requests import PATH_GET

    opened = threading.Event()
    release = threading.Event()

    def request_input(key, title, body, **kwargs):
        opened.set()
        release.wait(5)
        return _Pending(True)

    runtime = _runtime(True, [])
    runtime.request_input = request_input

    sandbox = Sandbox()
    sandbox.bind_runtime(runtime)
    interpreter = sandbox.interpreter

    def ask():
        """Occupy an approval with a dialog nobody has answered."""
        interpreter.submit(
            Execution(name="packages",
                      chain=Chain(root="user").push("packages")),
            Request(PLUGIN_UPDATE, {"name": "tool_edit_file"}))

    served: list = []

    def unrelated():
        """A perfectly safe Request from somewhere else entirely."""
        served.append(interpreter.submit(
            Execution(name="repl", chain=Chain(root="frontend:repl")),
            Request(PATH_GET, {"name": "data"})))

    try:
        # Both go through the real gate queue and the real gate thread —
        # calling ``_gate_one`` directly would run the approver on the test's
        # own thread and prove nothing about the gate.
        threading.Thread(target=ask, daemon=True).start()
        assert opened.wait(5), "the approver was never reached"

        answered = threading.Thread(target=unrelated, daemon=True)
        answered.start()
        answered.join(timeout=5)
        assert served, "the gate was blocked behind an open dialog"
        assert served[0].ok
    finally:
        release.set()
        sandbox.shutdown()


@pytest.fixture()
def request_update():
    return Request(PLUGIN_UPDATE, {"name": "tool_edit_file"})


def test_an_unwired_sandbox_refuses_without_asking(request_update):
    """The bug, stated: this is what /packages update hit."""
    sandbox = Sandbox()

    assert sandbox.interpreter.can_ask is False
    result = _gate(sandbox, request_update)
    assert not result.ok
    assert "changes what the system can do" in result.error


def test_binding_a_runtime_puts_the_question_to_the_user(request_update):
    asked: list = []
    sandbox = Sandbox()
    sandbox.bind_runtime(_runtime(True, asked))

    assert sandbox.interpreter.can_ask is True
    result = _gate(sandbox, request_update)

    assert asked, "no dialog was rendered"
    assert result.ok


def test_a_user_saying_no_is_still_a_refusal(request_update):
    """Wiring the dialog must not turn approval into a formality."""
    asked: list = []
    sandbox = Sandbox()
    sandbox.bind_runtime(_runtime(False, asked))

    result = _gate(sandbox, request_update)
    assert asked and not result.ok


def test_binding_does_not_clobber_an_explicit_approver(request_update):
    """A caller that supplied its own decision keeps it.

    Tests and the stress harness wire an approver directly; bootstrap calling
    ``bind_runtime`` afterwards must not silently replace it.
    """
    calls: list = []
    sandbox = Sandbox(approve=lambda chain, req, dec: calls.append(req) or True)
    sandbox.bind_runtime(_runtime(False, []))

    result = _gate(sandbox, request_update)
    assert calls and result.ok


def test_binding_nothing_is_a_no_op():
    """Absent a runtime there is still nobody to ask, and that must hold."""
    sandbox = Sandbox()
    sandbox.bind_runtime(None)
    assert sandbox.interpreter.can_ask is False


def test_the_lazy_sandbox_knows_where_plugin_trees_are():
    """``get_sandbox`` used to build a bare Sandbox directly.

    That skipped ``configure``, which is what sets ``plugin_roots`` — so
    ``dependencies_files`` resolved only inside a plugin's own tree, and an
    installed tool declaring a kernel helper would not have found it.
    """
    import sandbox.bridge as bridge

    saved = bridge._SANDBOX
    bridge._SANDBOX = None
    try:
        assert bridge.get_sandbox().plugin_roots
    finally:
        bridge._SANDBOX = saved
