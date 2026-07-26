"""The chain of provenance.

The chain does two jobs, and the second one is easy to miss: it is the record
of who caused this, *and* it is the call stack, which makes it the cycle
detector too.

Until now every chain in the system was exactly one link deep, so the nesting
this was designed for had never actually happened. These tests nest it.
"""

from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Chain, Interpreter, Request, Sandbox
from sandbox.guest.requests import FS_READ, NET_HTTP
from sandbox.policy import MAX_DEPTH, classify

FIXTURES = Path(__file__).parent / "fixtures"
SCRIPT = FIXTURES / "scratch_script.py"


@pytest.fixture
def sb():
    """A sandbox that refuses everything unsafe."""
    box = Sandbox()
    yield box
    box.shutdown()


def _read(path="x"):
    """A safe Request, for exercising the chain rather than the policy."""
    return Request(FS_READ, {"path": path})


# ──────────────────────────────────────────────────────────────────────
# The chain itself.
# ──────────────────────────────────────────────────────────────────────

def test_a_chain_starts_at_what_caused_the_work():
    """The root is the part that makes a dialog answerable."""
    chain = Chain(root="cron:nightly_index")
    assert chain.render() == "cron:nightly_index"
    assert not chain.attended


def test_a_user_turn_is_attended_and_a_cron_job_is_not():
    """Attendance is a property of the root, not of the leaf."""
    assert Chain(root="user").attended
    assert not Chain(root="cron:nightly").attended
    assert not Chain(root="spawn_subagent:7").attended


def test_pushing_nests_and_keeps_the_root():
    """Descending into a call never loses what caused it."""
    chain = Chain(root="user").push("tool_a").push("service_b")
    assert chain.render() == "user -> tool_a -> service_b"
    assert chain.depth == 2
    assert chain.attended


def test_a_chain_is_immutable():
    """Pushing returns a new chain, so a callee cannot edit its caller's."""
    parent = Chain(root="user").push("tool_a")
    child = parent.push("service_b")
    assert parent.render() == "user -> tool_a"
    assert child.render() == "user -> tool_a -> service_b"
    assert parent.links == ("tool_a",)


# ──────────────────────────────────────────────────────────────────────
# The chain as a safety mechanism.
# ──────────────────────────────────────────────────────────────────────

def test_deep_nesting_is_refused():
    """Runaway nesting is caught by policy, not by exhausting the machine."""
    deep = Chain(links=tuple(f"p{i}" for i in range(MAX_DEPTH + 1)))
    decision = classify(_read(), deep)
    assert not decision.safe
    assert "deeper than" in decision.reason


def test_nesting_just_under_the_cap_is_fine():
    """The cap is a cap, not a discouragement."""
    ok = Chain(links=tuple(f"p{i}" for i in range(MAX_DEPTH)))
    assert classify(_read(), ok).safe


def test_a_cycle_is_refused_even_when_shallow():
    """A tool that reaches itself recurses until something breaks."""
    chain = Chain(root="user").push("tool_a").push("tool_b").push("tool_a")
    decision = classify(_read(), chain)
    assert not decision.safe
    assert "cycle" in decision.reason
    assert "tool_a" in decision.reason


def test_direct_self_recursion_is_a_cycle():
    """The simplest careless mistake."""
    chain = Chain(root="user").push("tool_a").push("tool_a")
    assert not classify(_read(), chain).safe


def test_distinct_names_are_never_a_cycle():
    """Ordinary composition must not trip the detector."""
    chain = Chain(root="user").push("a").push("b").push("c")
    assert classify(_read(), chain).safe


# ──────────────────────────────────────────────────────────────────────
# Nesting for real, through the facade.
# ──────────────────────────────────────────────────────────────────────

def test_a_nested_run_extends_its_parents_chain(sb, tmp_path):
    """What a tool calling a tool will look like."""
    target = tmp_path / "f.txt"
    target.write_text("a b c", encoding="utf-8")
    seen = []
    sb.interpreter._record = lambda chain, req, dec, res: seen.append(
        chain.render())

    parent = Chain(root="user").push("tool_outer")
    sb.run(SCRIPT, "summarize", kwargs={"path": str(target)},
           chain=parent, name="tool_inner")
    unload_box("scratch_script")

    assert seen
    assert seen[0] == "user -> tool_outer -> tool_inner"


def test_the_dialog_sees_the_whole_chain_not_just_the_leaf(tmp_path):
    """'service_web wants HTTP' is unanswerable; the chain is the answer."""
    shown = {}

    def approve(chain, request, decision):
        """Record what a user would be shown, and refuse."""
        shown["chain"] = chain.render()
        shown["reason"] = decision.reason
        return False

    box = Sandbox(interpreter=Interpreter(approve=approve))
    try:
        script = tmp_path / "reach_out.py"
        script.write_text(
            "def go(sdk):\n"
            "    r = sdk.net.http('https://example.invalid/collect')\n"
            "    return sdk.ok(r.denied)\n", encoding="utf-8")
        result = box.run(script, "go",
                         chain=Chain(root="cron:nightly_index").push("task_x"))
    finally:
        box.shutdown()
        unload_box("reach_out")

    assert result.data is True
    assert shown["chain"] == "cron:nightly_index -> task_x -> reach_out"
    assert "example.invalid" in shown["reason"]


def test_a_resident_boxs_requests_carry_its_chain(sb, tmp_path):
    """Provenance survives being loaded once and called many times."""
    target = tmp_path / "f.txt"
    target.write_text("resident", encoding="utf-8")
    seen = []
    sb.interpreter._record = lambda chain, req, dec, res: seen.append(
        chain.render())

    box = sb.open(FIXTURES / "service_counter.py", "Counter",
                  chain=Chain(root="user").push("tool_caller"))
    box.call("read_file", path=str(target))
    box.call("read_file", path=str(target))

    assert len(seen) == 2
    assert all(c == "user -> tool_caller -> counter" for c in seen)


def test_the_guest_never_sees_the_chain(sb, tmp_path):
    """Plugins cannot report their own identity, so they cannot misstate it."""
    script = tmp_path / "probe_chain.py"
    script.write_text(
        "def go(sdk):\n"
        "    found = [a for a in dir(sdk) if 'chain' in a or 'proven' in a]\n"
        "    return sdk.ok(found)\n", encoding="utf-8")
    try:
        result = sb.run(script, "go", chain=Chain(root="user"))
    finally:
        unload_box("probe_chain")
    assert result.data == []
