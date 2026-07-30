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
              "retryable": True, "code": "a_code",
              "llm_summary": "a summary",
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


# ──────────────────────────────────────────────────────────────────────
# The code vocabulary.
# ──────────────────────────────────────────────────────────────────────

def test_the_vocabulary_is_closed_and_unique():
    """Two constants sharing a value would make them indistinguishable."""
    from sandbox.guest import codes

    named = {name: value for name, value in vars(codes).items()
             if name.startswith("ERROR_")}
    assert len(set(named.values())) == len(named), "duplicate code value"
    assert codes.ALL_CODES == set(named.values())


def test_every_denial_code_is_a_real_code():
    """DENIAL_CODES is what sdk.Denied is raised for — a typo here widens it."""
    from sandbox.guest import codes

    assert codes.DENIAL_CODES <= codes.ALL_CODES
    assert codes.ERROR_DENIED in codes.DENIAL_CODES
    # Breakage must never be catchable as a policy refusal.
    assert codes.ERROR_TIMEOUT not in codes.DENIAL_CODES
    assert codes.ERROR_HANDLER_ERROR not in codes.DENIAL_CODES


def test_a_refusal_carries_the_denial_code_without_being_asked():
    """All 20 existing `refusal` sites classify correctly untouched."""
    from sandbox.guest.codes import ERROR_DENIED, ERROR_NOT_PERMITTED

    assert Result.refusal("nope").code == ERROR_DENIED
    assert Result.refusal().code == ERROR_DENIED
    # A narrower reason can be named where it is known.
    assert Result.refusal("outside", code=ERROR_NOT_PERMITTED).code == \
        ERROR_NOT_PERMITTED


def test_an_ordinary_failure_carries_no_code_by_default():
    """The 166 existing `failure` sites are unchanged."""
    assert Result.failure("went wrong").code == ""
    assert Result.failure("went wrong", retryable=True).retryable is True


# ──────────────────────────────────────────────────────────────────────
# The codes the kernel mints for itself.
# ──────────────────────────────────────────────────────────────────────

def test_a_handler_that_raises_is_coded_as_breakage(interp, monkeypatch):
    """Not a refusal — a plugin must not catch this as sdk.Denied."""
    from sandbox.guest.codes import ERROR_HANDLER_ERROR

    def boom(_ctx, _args):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(HANDLERS, CONFIG_READ, boom)

    def plugin(sdk):
        answer = sdk._send(_read_request())
        return sdk.ok({"code": answer.code, "denied": answer.denied})

    out = run_in_process(interp, plugin, name="raiser").data
    assert out["code"] == ERROR_HANDLER_ERROR
    assert out["denied"] is False


def test_an_unapproved_request_is_coded_as_a_declined_approval(interp):
    """No approver is wired, so anything unsafe is refused at the gate."""
    from sandbox.guest.codes import ERROR_APPROVAL_DECLINED

    def plugin(sdk):
        try:
            sdk.net.http("https://example.invalid/")
        except sdk.Denied as exc:
            return sdk.ok(exc.result.code)
        return sdk.ok("not denied")

    assert run_in_process(interp, plugin, name="egress").data == \
        ERROR_APPROVAL_DECLINED


def test_reading_a_protected_file_is_coded_not_permitted(interp):
    """The credential store, which is the one thing reads are narrow about.

    Reads are otherwise deliberately broad — egress is the control — so a
    protected path is where ``fs.read`` actually refuses rather than merely
    failing to find something.
    """
    from sandbox.guest.codes import ERROR_NOT_PERMITTED
    from sandbox.protected import protected_paths

    guarded = sorted(protected_paths())
    if not guarded:
        pytest.skip("no protected paths configured in this environment")

    def plugin(sdk, path):
        try:
            sdk.fs.read(path)
        except sdk.Denied as exc:
            return sdk.ok(exc.result.code)
        return sdk.ok("not denied")

    out = run_in_process(interp, plugin, name="peeker",
                         kwargs={"path": str(guarded[0])})
    assert out.data == ERROR_NOT_PERMITTED


def _read_request():
    """A CONFIG_READ Request, built where the import is cheap."""
    from sandbox.guest.requests import Request
    return Request(CONFIG_READ, {"key": "anything"})
