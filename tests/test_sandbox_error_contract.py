"""What a failed Request promises, on both sides of the boundary.

Two claims live here, and until now neither was tested.

**The interpreter catches what a handler throws.** ``Interpreter._execute``
wraps every handler call, so a handler may raise and the guest still gets an
ordinary failed Result rather than a dead sandbox. Two-thirds of the handlers
in ``sandbox/handlers/kernel.py`` additionally guard themselves, which is
redundant *and* costs the traceback -- so before any of those guards can be
removed, the net underneath them has to be something more than an assumption.
That is what ``test_a_raising_handler_...`` and its traceback sibling pin.

**A Result survives the wire.** ``to_dict``/``from_dict`` enumerate fields by
hand, so a field added to the dataclass and forgotten in ``to_dict`` is lost
only when a box is subprocessed -- silent in-process, silent in most of the
suite. ``test_every_result_field_crosses_the_wire`` derives the expectation
from the dataclass itself, so it keeps working for fields that do not exist
yet.
"""

import dataclasses
import logging

import pytest

from sandbox import Interpreter, Result, run_in_process
from sandbox.interpreter import HANDLERS
from sandbox.guest.requests import CONFIG_READ


@pytest.fixture
def interp():
    """An interpreter that refuses everything unsafe (no approver)."""
    it = Interpreter()
    yield it
    it.shutdown()


# ──────────────────────────────────────────────────────────────────────
# The net under every handler.
# ──────────────────────────────────────────────────────────────────────

def test_a_raising_handler_fails_the_request_and_not_the_sandbox(
        interp, monkeypatch):
    """A handler may raise. The guest hears a failure; the box stays usable."""
    def boom(_ctx, _args):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(HANDLERS, CONFIG_READ, boom)

    def plugin(sdk):
        try:
            sdk.config.read("anything")
        except sdk.Failed as exc:
            return sdk.ok(str(exc))
        return sdk.ok("no failure raised")

    result = run_in_process(interp, plugin, name="raiser")
    assert result.ok
    assert "kaboom" in result.data

    # The interpreter is still serving: a second request works.
    monkeypatch.undo()
    assert run_in_process(interp, lambda sdk: sdk.ok("alive"),
                          name="after").data == "alive"


def test_a_raising_handler_leaves_a_traceback_in_the_log(
        interp, monkeypatch, caplog):
    """The reason a per-handler ``except Exception`` is a downgrade.

    The net logs the stack trace; a handler that catches for itself reports a
    tidier sentence and throws the trace away. Removing those guards is only
    safe because this assertion holds.
    """
    def boom(_ctx, _args):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(HANDLERS, CONFIG_READ, boom)

    with caplog.at_level(logging.ERROR, logger="Sandbox"):
        run_in_process(interp, lambda sdk: sdk.config.read("k"), name="raiser")

    failures = [r for r in caplog.records if "handler failed" in r.getMessage()]
    assert failures, "the interpreter did not log the handler failure"
    assert failures[0].exc_info is not None, "logged without a traceback"


def test_an_invented_request_type_cannot_even_be_built():
    """The guest never reaches the interpreter's dispatch miss.

    ``Request.__post_init__`` validates the type against the closed vocabulary,
    so ``no handler for ...`` at the top of ``_execute`` is defence in depth
    against a type that was added to the vocabulary and never wired -- which
    ``test_sandbox_catalogue`` separately pins cannot happen. Worth stating,
    because that failure message reads like something a plugin could provoke.
    """
    from sandbox.guest.requests import Request

    with pytest.raises(ValueError, match="unknown request type"):
        Request("not.a.request", {})


# ──────────────────────────────────────────────────────────────────────
# The return contract crosses the wire intact.
# ──────────────────────────────────────────────────────────────────────

def _populated_result() -> Result:
    """A Result with every field set to something distinguishable."""
    values = {"ok": False, "data": {"n": 1}, "error": "went wrong",
              "retryable": True, "llm_summary": "a summary",
              "attachment_paths": ["/tmp/a.png"], "also_contains": ["x"],
              "discovered_paths": ["/tmp/b"]}
    missing = {f.name for f in dataclasses.fields(Result)} - set(values)
    assert not missing, (
        f"add {sorted(missing)} to this fixture — a new Result field is only "
        f"covered here once it is given a distinctive value")
    return Result(**values)


def test_every_result_field_crosses_the_wire():
    """``to_dict`` enumerates fields by hand, so it can silently fall behind.

    Derived from ``dataclasses.fields`` rather than a written-out list: a field
    added to Result and forgotten in ``to_dict`` is lost *only* on the
    subprocess hop, which most of the suite never exercises.
    """
    payload = _populated_result().to_dict()
    assert set(payload) == {f.name for f in dataclasses.fields(Result)}


def test_a_result_round_trips_through_the_wire_format():
    """What the child sends is what the parent rebuilds."""
    original = _populated_result()
    assert Result.from_dict(original.to_dict()) == original


def test_from_dict_tolerates_a_peer_that_omits_a_field():
    """Only ``ok`` is required, so an older peer degrades rather than crashes."""
    rebuilt = Result.from_dict({"ok": False})
    assert not rebuilt.ok
    assert rebuilt.error == ""
    assert rebuilt.attachment_paths == []
