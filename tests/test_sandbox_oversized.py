"""An answer that will not fit is a failure, never a fault.

A handler that produced more than ``protocol.MAX_MESSAGE_BYTES`` used to break
two different ways depending on which side of a pipe the asking code was on.
In-process ``InterpreterChannel.send`` caught the ``ProtocolError`` and turned
it into a failure, so the plugin saw something. Over a pipe nothing caught it:
``runner_subprocess.send`` catches ``OSError`` and ``ValueError`` only, so the
error escaped ``service_until`` and killed the box outright.

That is how one long conversation stopped a frontend. ``conv.read`` answered
with 20 MB, the HTTP frontend's ``poll`` raised on it every tick, and because
``facade.collect_act`` deletes an act's result *before* it crosses, the
client's held request was destroyed rather than answered — a browser waiting
forever on an answer nobody still had.

The guard is at ``interpreter._settle``, which the module's own docstring
already calls "the one funnel every serviced Request passes through". Putting
it there is what makes this a property of the kernel rather than of whichever
handler was patched last, so these tests deliberately reach it through
``db.query`` — a Request that has nothing to do with conversations.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.database import Database
from sandbox import Interpreter, run_in_process
from sandbox.guest import protocol
from sandbox.guest.codes import ERROR_TOO_LARGE
from sandbox.guest.requests import CONFIG_READ, Request, Result
from sandbox.interpreter import _deliverable
from sandbox.runner_subprocess import run_in_subprocess

FIXTURE = Path(__file__).parent / "fixtures" / "sandbox_oversized_plugin.py"


@pytest.fixture(scope="module")
def both_runners(tmp_path_factory):
    """The same fixture file under both boundaries, as ``test_sandbox_bytes``
    does — a test that exercised one would have passed against the bug."""
    tmp_path = tmp_path_factory.mktemp("oversized")
    db = Database(str(tmp_path / "oversized.db"))
    interp = Interpreter(context=SimpleNamespace(db=db, user_id=1))

    def run(func_name, **kwargs):
        import importlib.util

        spec = importlib.util.spec_from_file_location("oversized_fixture",
                                                      FIXTURE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        in_proc = run_in_process(interp, getattr(module, func_name),
                                 name=func_name, kwargs=kwargs, timeout=120)
        sub = run_in_subprocess(interp, str(FIXTURE), func_name,
                                name=func_name, kwargs=kwargs, timeout=120)
        return in_proc, sub

    yield run
    interp.shutdown()


# ── The unit: what the funnel substitutes ─────────────────────────────

def test_an_oversized_answer_becomes_a_coded_failure():
    request = Request(CONFIG_READ, {"key": "k"})
    huge = Result(data="x" * (protocol.MAX_MESSAGE_BYTES + 1))

    answer = _deliverable(request, huge)

    assert not answer.ok
    assert answer.code == ERROR_TOO_LARGE
    assert "config.read" in answer.error


def test_the_substitute_can_itself_cross():
    """Otherwise the guard reproduces the bug it exists to prevent."""
    request = Request(CONFIG_READ, {"key": "k"})

    answer = _deliverable(request, Result(data="x" * (protocol.MAX_MESSAGE_BYTES + 1)))

    assert answer.crossing() == answer


def test_too_large_is_breakage_rather_than_a_refusal():
    """``sdk.Denied`` means a person or a policy said no. Nobody said no here
    — the answer was simply too big to carry, and a plugin catching ``Denied``
    to apologise for a permission it does not need would be wrong."""
    from sandbox.guest.codes import DENIAL_CODES

    assert ERROR_TOO_LARGE not in DENIAL_CODES


def test_an_ordinary_answer_passes_through_unchanged():
    request = Request(CONFIG_READ, {"key": "k"})
    ordinary = Result(data={"n": 1})

    assert _deliverable(request, ordinary) is ordinary


def test_a_live_object_on_a_result_is_reported_rather_than_raised():
    """Unserializable rather than merely large, and the same treatment: the
    plugin gets a failure it can read, not a traceback from the plumbing."""
    from sandbox.guest.codes import ERROR_HANDLER_ERROR

    answer = _deliverable(Request(CONFIG_READ, {"key": "k"}),
                          Result(data=object()))

    assert not answer.ok and answer.code == ERROR_HANDLER_ERROR


# ── The pipe: what used to escape ─────────────────────────────────────

def test_send_drops_an_uncarryable_message_instead_of_raising():
    """``send``'s ``ProtocolError`` escaped ``service_until`` and killed the
    box. It is caught separately from ``OSError``/``ValueError`` because it
    means the opposite thing: the pipe is fine, the message is not."""
    from sandbox.runner_subprocess import send

    class _Stdin:
        def write(self, raw):
            raise AssertionError("nothing should reach the pipe")

        def flush(self):
            pass

    proc = SimpleNamespace(stdin=_Stdin())

    assert send(proc, {"kind": "result",
                       "result": {"ok": True,
                                  "data": "x" * (protocol.MAX_MESSAGE_BYTES + 1)}}) is False


# ── Both boundaries, end to end ───────────────────────────────────────

def test_both_runners_report_the_same_failure(both_runners):
    in_proc, sub = both_runners("ask_for_too_much")

    for label, result in (("in-process", in_proc), ("subprocess", sub)):
        assert result.ok, f"{label}: the run itself failed — {result.error}"
        assert result.data["raised"] is True, label
        assert result.data["code"] == ERROR_TOO_LARGE, label
        assert result.data["denied"] is False, label


def test_the_box_survives_being_told_no(both_runners):
    """The property the crash was actually about. A frontend is a resident
    box; when an oversized answer killed it, the poll loop had nothing left to
    poll and the UI stopped without saying anything."""
    in_proc, sub = both_runners("still_usable_afterwards")

    for label, result in (("in-process", in_proc), ("subprocess", sub)):
        assert result.ok, f"{label}: {result.error}"
        assert result.data["alive"] is True, label
        assert result.data["counted"] == 20, label
