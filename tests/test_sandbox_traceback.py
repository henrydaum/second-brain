"""Where a plugin broke, reported the same way whichever runner caught it.

An author's question after a crash is "which line of *mine*". Until now the
answer depended on a choice the author does not make and cannot see: the kernel
picks the runner from the tree a file sits in, and each of the four
runner/lifetime combinations reported a raise differently — two error formats,
two codes, and a traceback that was captured on exactly one path and discarded
before reaching any caller.

So the claim under test is parity, not merely presence: one bug, one sentence,
one code, and a stack naming the plugin's own line, four times over.
"""

import importlib.util
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Interpreter, run_in_process
from sandbox.boxes import open_box
from sandbox.guest.codes import ERROR_GUEST_FAULT
from sandbox.guest.faults import clamp, guest_traceback
from sandbox.runner_subprocess import run_in_subprocess

FIXTURE = Path(__file__).parent / "fixtures" / "sandbox_plugin.py"

#: The line ``raises()`` throws from, read out of the fixture rather than
#: written down, so editing that file cannot quietly rot the assertion.
RAISE_LINE = next(
    number
    for number, line in enumerate(
        FIXTURE.read_text(encoding="utf-8").splitlines(), 1)
    if 'raise ValueError("something went wrong")' in line)

#: What every runner must say about it.
SENTENCE = "ValueError: something went wrong"

#: Frames belonging to the machinery. None of these is anything the author
#: wrote, and a stack opening with three of them reads like a kernel bug.
MACHINERY = ("child.py", "runner.py", "boxes.py", "_worker", "_invoke",
             "_run_ephemeral", "_serve_persistent")


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
def four(interp):
    """The same raise, caught by each runner and lifetime in turn.

    ``entry=""`` makes the *module* the resident object and its functions the
    methods, so the persistent pair calls the very same function on the very
    same line as the ephemeral pair — the parity claim is about the reporting,
    and this keeps the code out of the variables.
    """
    module = _load_fixture()
    results, opened = {}, []

    results["in_process/ephemeral"] = run_in_process(
        interp, module.raises, name="raises", timeout=30)
    results["subprocess/ephemeral"] = run_in_subprocess(
        interp, str(FIXTURE), "raises", name="raises", timeout=30)

    for label, isolated in (("in_process", False), ("subprocess", True)):
        box = open_box(interp, str(FIXTURE), entry="",
                       name=f"trace_{label}", isolated=isolated,
                       manage_lifecycle=False)
        opened.append(box)
        results[f"{label}/persistent"] = box.call("raises")

    yield results

    for box in opened:
        box.stop()
        unload_box(box.name)


def test_every_runner_reports_the_same_failure(four):
    """One bug, one sentence, one code — whichever box caught it."""
    assert {r.error for r in four.values()} == {SENTENCE}
    assert {r.code for r in four.values()} == {ERROR_GUEST_FAULT}


def test_every_runner_reports_the_line_the_author_wrote(four):
    """The whole point: a line number in the plugin, not in the kernel."""
    for label, result in four.items():
        assert f"line {RAISE_LINE}, in raises" in result.traceback, label
        assert "sandbox_plugin.py" in result.traceback, label


def test_no_runner_leaks_its_own_frames(four):
    """The machinery above the plugin answers somebody else's question."""
    for label, result in four.items():
        for frame in MACHINERY:
            assert frame not in result.traceback, f"{label} leaked {frame}"


# ──────────────────────────────────────────────────────────────────────
# The formatter's own edges.
# ──────────────────────────────────────────────────────────────────────

def test_an_unraised_exception_has_nothing_to_say():
    """``child.py`` faults with an exception it constructed itself.

    ``format_exc()`` answers "NoneType: None" there — or, worse, a stale stack
    from something handled earlier in the process. Saying nothing beats both.
    """
    assert guest_traceback(ValueError("never raised")) == ""


def test_a_huge_traceback_is_capped_by_its_middle():
    """Both ends carry signal, and the wire has a ceiling."""
    capped = clamp("x" * 9000, limit=1000)
    assert len(capped) < 1200
    assert capped.startswith("x") and capped.endswith("x")
    assert "elided" in capped
