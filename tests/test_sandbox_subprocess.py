"""Subprocess execution, and the property the whole design rests on: the same
plugin file behaves identically under both runners.

The security contract's claim is that *the code functions the same either way,
just with different levels of security*. These tests are what makes that a
fact rather than an intention.
"""

import importlib.util
import time
from pathlib import Path

import pytest

from sandbox import Chain, Interpreter, Result, run_in_process
from sandbox.runner_subprocess import run_in_subprocess

FIXTURE = Path(__file__).parent / "fixtures" / "sandbox_plugin.py"


def _load_fixture():
    """Import the fixture in-process so both runners get the same code."""
    spec = importlib.util.spec_from_file_location("sandbox_fixture", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def interp():
    """An interpreter that refuses everything unsafe."""
    it = Interpreter()
    yield it
    it.shutdown()


@pytest.fixture
def both(interp):
    """Run one fixture function under each runner and return both Results."""
    module = _load_fixture()

    def run(func_name, **kwargs):
        """Execute under in-process and subprocess runners."""
        in_proc = run_in_process(
            interp, getattr(module, func_name),
            name=func_name, kwargs=kwargs, timeout=30)
        sub = run_in_subprocess(
            interp, str(FIXTURE), func_name,
            name=func_name, kwargs=kwargs, timeout=30)
        return in_proc, sub

    return run


def _same(a: Result, b: Result):
    """Assert two Results are equivalent in every observable way."""
    assert a.ok == b.ok, f"ok differs: {a} vs {b}"
    assert a.data == b.data, f"data differs: {a.data!r} vs {b.data!r}"
    assert a.denied == b.denied, f"denied differs: {a} vs {b}"


# ──────────────────────────────────────────────────────────────────────
# The core claim.
# ──────────────────────────────────────────────────────────────────────

def test_read_is_identical(both, tmp_path):
    """A successful read behaves the same on both runners."""
    target = tmp_path / "notes.md"
    target.write_text("hello sandbox", encoding="utf-8")
    a, b = both("read_and_truncate", path=str(target), limit=5)
    _same(a, b)
    assert a.data == "he..."


def test_helper_requests_are_identical(both, tmp_path):
    """A plain helper making Requests works the same on both."""
    target = tmp_path / "a.txt"
    target.write_text("A", encoding="utf-8")
    a, b = both("via_helper", path=str(target))
    _same(a, b)
    assert a.data == "A"


def test_egress_is_refused_on_both(both):
    """Policy is enforced by the one gate, so both runners refuse alike."""
    a, b = both("attempt_egress")
    _same(a, b)
    assert a.data["denied"] is True


def test_denial_is_survivable_on_both(both, tmp_path):
    """A refusal is an ordinary failure in both runners - not a kill."""
    target = tmp_path / "ok.txt"
    target.write_text("still here", encoding="utf-8")
    a, b = both("survives_denial", path=str(target))
    _same(a, b)
    assert a.data == "still here"


def test_respond_terminates_on_both(both):
    """Asking to end actually ends it, identically."""
    a, b = both("responds_early")
    _same(a, b)
    assert a.data == "early"


def test_unhandled_errors_become_failures_on_both(both):
    """A plugin that breaks yields a failure Result, never a crash."""
    a, b = both("raises")
    assert not a.ok and not b.ok
    assert "something went wrong" in a.error
    assert "something went wrong" in b.error


def test_missing_file_is_reported_the_same(both):
    """The failure path matches too, not just the happy path."""
    a, b = both("read_and_truncate", path="does/not/exist.txt")
    _same(a, b)
    assert not a.ok


# ──────────────────────────────────────────────────────────────────────
# Properties specific to the subprocess boundary.
# ──────────────────────────────────────────────────────────────────────

def test_plugin_stdout_does_not_corrupt_the_wire(interp):
    """Plugin prints go to stderr so the protocol stream stays clean."""
    result = run_in_subprocess(interp, str(FIXTURE), "prints_to_stdout",
                               name="printer", timeout=30)
    assert result.ok
    assert result.data == "survived"


def test_a_runaway_is_actually_killed(interp):
    """The real win over threads: a process can be stopped."""
    started = time.perf_counter()
    result = run_in_subprocess(interp, str(FIXTURE), "spins",
                               name="runaway", timeout=1.0)
    elapsed = time.perf_counter() - started
    assert not result.ok
    assert "timed out" in result.error
    assert elapsed < 15.0


def test_missing_function_faults_cleanly(interp):
    """A bad entry point is reported, not hung on."""
    result = run_in_subprocess(interp, str(FIXTURE), "no_such_function",
                               name="missing", timeout=30)
    assert not result.ok
    assert "no_such_function" in result.error


def test_missing_file_faults_cleanly(interp):
    """So is a bad module path."""
    result = run_in_subprocess(interp, "does/not/exist.py", "anything",
                               name="missing", timeout=30)
    assert not result.ok


def test_provenance_reaches_the_dialog_from_a_subprocess(tmp_path):
    """The chain is kernel-side, so it survives the process boundary."""
    seen = {}

    def approve(chain, request, decision):
        """Record what the user would have been shown."""
        seen["chain"] = chain.render()
        return False

    it = Interpreter(approve=approve)
    try:
        run_in_subprocess(it, str(FIXTURE), "attempt_egress",
                          name="fetcher", timeout=30,
                          chain=Chain(root="cron:nightly_index"))
    finally:
        it.shutdown()

    assert seen["chain"] == "cron:nightly_index -> fetcher"


def test_round_trip_cost_versus_in_process(interp, tmp_path, capsys):
    """The number that justifies in-process being the default.

    Concurrency is cheap in both runners; *granularity* is what costs, and it
    costs far more over a pipe than over a queue.
    """
    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")
    module = _load_fixture()
    iterations = 300
    kwargs = {"path": str(target), "iterations": iterations}

    thread = run_in_process(interp, module.bench, name="bench",
                            kwargs=kwargs, timeout=60)
    child = run_in_subprocess(interp, str(FIXTURE), "bench", name="bench",
                              kwargs=kwargs, timeout=60)
    assert thread.ok and child.ok

    thread_us = (thread.data / iterations) * 1_000_000
    child_us = (child.data / iterations) * 1_000_000
    with capsys.disabled():
        print(f"\n  in-process: {thread_us:8.1f} us/request"
              f"\n  subprocess: {child_us:8.1f} us/request"
              f"  ({child_us / thread_us:.1f}x)")

    assert child_us > thread_us


def test_requests_are_ledgered_from_a_subprocess(tmp_path):
    """Both runners feed the same ledger sink."""
    target = tmp_path / "x.txt"
    target.write_text("x", encoding="utf-8")
    rows = []

    it = Interpreter(record=lambda chain, req, dec, res: rows.append(
        (chain.render(), req.type, dec.level, res.ok)))
    try:
        run_in_subprocess(it, str(FIXTURE), "read_and_truncate",
                          name="reader", timeout=30,
                          kwargs={"path": str(target)})
    finally:
        it.shutdown()

    assert rows
    chain, req_type, level, ok = rows[0]
    assert req_type == "fs.read"
    assert chain.endswith("reader")
    assert ok
