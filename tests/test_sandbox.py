"""Sandbox slice 1 — the drive loop, policy, provenance, and the return
contract, plus a round-trip benchmark so the cost is measured rather than
argued about."""

import time

import pytest

from sandbox import Chain, Interpreter, Request, Result, run_in_process
from sandbox.policy import MAX_DEPTH, classify
from sandbox.interpreter import (DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS,
                                 clamp_timeout)
from sandbox.guest.requests import NET_HTTP


@pytest.fixture
def interp():
    """An interpreter that refuses everything unsafe (no approver)."""
    it = Interpreter()
    yield it
    it.shutdown()


# ──────────────────────────────────────────────────────────────────────
# The loop works, and plugin code is plain synchronous Python.
# ──────────────────────────────────────────────────────────────────────

def test_plugin_reads_a_file(interp, tmp_path):
    """A read Request round-trips and the code reads like ordinary Python."""
    target = tmp_path / "notes.md"
    target.write_text("hello sandbox", encoding="utf-8")

    def plugin(sdk, path):
        """Read a file, no yield in sight."""
        r = sdk.fs.read(path)
        if not r:
            return sdk.fail(r.error)
        return sdk.ok(sdk.text.truncate(r.data, 5))

    result = run_in_process(interp, plugin, name="reader",
                            kwargs={"path": str(target)})
    assert result.ok
    assert result.data == "he..."


def test_helpers_can_make_requests_without_being_generators(interp, tmp_path):
    """The point of the thread model: a plain helper can make a Request."""
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")

    def load(sdk, path):
        """An ordinary function - not a generator, no yield from."""
        return sdk.fs.read(path).data

    def plugin(sdk):
        """Call the helper."""
        return sdk.ok(load(sdk, str(tmp_path / "a.txt")))

    assert run_in_process(interp, plugin, name="helper").data == "A"


# ──────────────────────────────────────────────────────────────────────
# Policy: reads broad, egress always gated.
# ──────────────────────────────────────────────────────────────────────

def test_egress_is_refused_when_nobody_can_approve(interp):
    """No approver means refuse - the same default the kernel uses when
    every permission hook abstains."""
    def plugin(sdk):
        """Try to reach the network."""
        return sdk.net.http("https://example.com/collect?data=secret")

    result = run_in_process(interp, plugin, name="exfil")
    assert not result.ok
    assert result.denied


def test_denial_is_an_ordinary_failure_not_a_kill(interp, tmp_path):
    """A refused Request must not end the execution - the code keeps going."""
    (tmp_path / "ok.txt").write_text("still here", encoding="utf-8")

    def plugin(sdk):
        """Get denied, then carry on and succeed."""
        denied = sdk.net.http("https://example.com")
        assert denied.denied
        return sdk.fs.read(str(tmp_path / "ok.txt"))

    result = run_in_process(interp, plugin, name="resilient")
    assert result.ok
    assert result.data == "still here"


def test_approval_sees_the_full_provenance_chain(tmp_path):
    """The dialog must be answerable: it gets the root, not just the leaf."""
    seen = {}

    def approve(chain, request, decision):
        """Record what the user would have been shown."""
        seen["chain"] = chain.render()
        seen["reason"] = decision.reason
        return False

    it = Interpreter(approve=approve)
    try:
        def plugin(sdk):
            """Attempt egress."""
            return sdk.net.http("https://example.com")

        run_in_process(it, plugin, name="fetcher",
                       chain=Chain(root="cron:nightly_index"))
    finally:
        it.shutdown()

    assert seen["chain"] == "cron:nightly_index -> fetcher"
    assert "example.com" in seen["reason"]


def test_get_is_gated_exactly_like_post():
    """A GET with data in the query string is exfiltration too."""
    for method in ("GET", "POST"):
        decision = classify(
            Request(NET_HTTP, {"url": "https://x.com/?d=1", "method": method}),
            Chain())
        assert not decision.safe


def test_deep_call_chains_are_caught():
    """A runaway cycle is stupidity, and it is caught by policy."""
    deep = Chain(links=tuple(f"p{i}" for i in range(MAX_DEPTH + 1)))
    assert not classify(Request("fs.read", {"path": "x"}), deep).safe


# ──────────────────────────────────────────────────────────────────────
# Clamping: declare freely, kernel decides.
# ──────────────────────────────────────────────────────────────────────

def test_declared_timeout_is_clamped():
    """A plugin may ask for a longer leash; it does not get one."""
    assert clamp_timeout(999999) == MAX_TIMEOUT_SECONDS
    assert clamp_timeout(5) == 5
    assert clamp_timeout(None) == DEFAULT_TIMEOUT_SECONDS
    assert clamp_timeout(-1) == DEFAULT_TIMEOUT_SECONDS


def test_timeout_starves_a_runaway(interp):
    """We cannot kill the thread - we stop servicing it, which is enough."""
    def plugin(sdk):
        """Spin forever making Requests."""
        while True:
            sdk.fs.list(".")

    started = time.perf_counter()
    result = run_in_process(interp, plugin, name="runaway", timeout=0.3)
    elapsed = time.perf_counter() - started

    assert not result.ok
    assert "timed out" in result.error
    assert elapsed < 3.0


def test_cancellation_unwinds_rather_than_failing(interp, tmp_path):
    """A cancelled execution is torn down, not merely told 'no'.

    Denial and cancellation look alike and are not. Code that treats a
    cancellation as an ordinary failure carries on and spins forever asking
    questions nobody will answer - which is exactly the hang this prevents.
    """
    from sandbox.guest.channel import Terminated
    from sandbox.guest.sdk import SDK
    from sandbox.interpreter import Execution, InterpreterChannel

    execution = Execution(name="doomed", chain=Chain().push("doomed"))
    sdk = SDK(InterpreterChannel(interp, execution))
    assert sdk.fs.list(str(tmp_path)).ok

    interp.cancel(execution)
    with pytest.raises(Terminated):
        sdk.fs.list(str(tmp_path))


def test_a_denial_is_still_an_ordinary_failure(interp):
    """The other half of the distinction: the user saying no is survivable."""
    def plugin(sdk):
        """Get denied and keep going."""
        denied = sdk.net.http("https://example.com")
        return sdk.ok({"denied": denied.denied, "still_running": True})

    result = run_in_process(interp, plugin, name="denied")
    assert result.ok
    assert result.data == {"denied": True, "still_running": True}


def test_a_starved_loop_actually_stops(interp):
    """The property the escalation depends on: an ignored-failure loop ends.

    Code that never checks its Results used to spin forever against a
    cancelled execution, hammering the gate for the life of the process.
    """
    def plugin(sdk):
        """Loop forever, ignoring every Result."""
        while True:
            sdk.fs.list(".")

    started = time.perf_counter()
    result = run_in_process(interp, plugin, name="ignorer", timeout=0.3)
    assert not result.ok
    assert time.perf_counter() - started < 5.0


# ──────────────────────────────────────────────────────────────────────
# The number that decides whether this design is viable.
# ──────────────────────────────────────────────────────────────────────

def test_round_trip_cost(interp, tmp_path, capsys):
    """Measure per-Request overhead. Concurrency is cheap; granularity costs.

    If this lands far above ~100us the design needs revisiting before
    anything is built on it.
    """
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    path = str(tmp_path / "f.txt")
    iterations = 300

    def plugin(sdk):
        """Make many small Requests."""
        t0 = time.perf_counter()
        for _ in range(iterations):
            sdk.fs.read(path)
        return sdk.ok(time.perf_counter() - t0)

    result = run_in_process(interp, plugin, name="bench", timeout=60)
    assert result.ok
    per_call_us = (result.data / iterations) * 1_000_000
    with capsys.disabled():
        print(f"\n  in-process round trip: {per_call_us:.1f} us/request "
              f"({iterations} requests in {result.data * 1000:.1f} ms)")
    assert per_call_us < 2000
