"""Scripts: SDK code the agent runs without it becoming a capability.

The point of a script is that it is the *cheap* thing to reach for. A shell
command is an OS process outside the boundary and is asked about every time; a
script is contained, every effect inside it arrives at the gate on its own, and
so running one costs no dialog. These pin the two halves of that bargain — the
containment that earns it, and the one case that still gets asked about.
"""

from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Chain, Sandbox
from sandbox.guest.box import SUBPROCESS
from sandbox.guest.requests import SCRIPT_RUN, Request
from sandbox.isolation import is_script, required_isolation
from sandbox.policy import SAFE, UNSAFE, classify

SCRIPT = '''\
"""A script: no base class, no declarations, functions that take sdk."""
{extra}

def main(sdk, values=()):
    """Compute something."""
    return sum(values)
'''


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A sandbox_plugins tree with a scripts/ directory in it."""
    root = tmp_path / "sandbox_plugins"
    (root / "scripts").mkdir(parents=True)
    monkeypatch.setattr("paths.SANDBOX_PLUGINS", root)
    monkeypatch.setattr("paths.INSTALLED_PLUGINS", tmp_path / "installed")
    return root


def write(tree, name="tally.py", extra=""):
    """Put a script on disk and answer with its path."""
    target = tree / "scripts" / name
    target.write_text(SCRIPT.format(extra=extra), encoding="utf-8")
    return target


@pytest.fixture(autouse=True)
def clean_boxes():
    """Module caches are per-box; leaking one hides staleness."""
    yield
    for name in ("tally", "heavy", "notascript"):
        unload_box(name)


# ── what makes a script a script ──────────────────────────────────────

def test_a_scripts_directory_is_what_declares_one(tree):
    """No prefix, no base class, no keyword — the directory is the whole of it."""
    assert is_script(write(tree))
    assert not is_script(tree / "tools" / "tool_x.py")
    assert not is_script(tree / "helpers" / "parse_x.py")


def test_nesting_does_not_count(tree):
    """Top level only, matching ``helpers/``."""
    nested = tree / "tools" / "scripts" / "tally.py"
    nested.parent.mkdir(parents=True)
    nested.write_text(SCRIPT.format(extra=""), encoding="utf-8")
    assert not is_script(nested)


def test_a_script_is_subprocessed_wherever_it_lives(tree, tmp_path):
    """The one place the per-tree answer is deliberately not consulted.

    An installed *plugin* that is pure SDK earns in-process execution, because
    somebody approved it at install and it is a declared, registered
    capability. A script is none of those things, and containment is the whole
    of what makes running one cheap — so it is not something an address can buy
    its way out of.
    """
    installed = tmp_path / "installed" / "scripts"
    installed.mkdir(parents=True)
    pure = installed / "tally.py"
    pure.write_text(SCRIPT.format(extra="import json"), encoding="utf-8")

    from sandbox.validator import validate_file

    report = validate_file(pure)
    assert report.unmediated == frozenset()      # would be in-process as a plugin
    assert required_isolation(pure, report) == SUBPROCESS
    assert required_isolation(write(tree), None) == SUBPROCESS


# ── the bargain: contained code runs without a dialog ─────────────────

def _ask(path, **args):
    """Classify one script.run against an ordinary agent chain."""
    return classify(Request(SCRIPT_RUN, {"path": str(path), **args}),
                    Chain(root="user").push("run_script"))


def test_running_a_contained_script_is_not_asked_about(tree):
    """The whole point. Every effect inside it is still classified on its own."""
    decision = _ask(write(tree))
    assert decision.level == SAFE
    assert "contained" in decision.reason


def test_a_foreign_library_is_asked_about_and_named(tree):
    """The one part of a script whose actions do not come back as Requests.

    An installed package importing one is subprocessed and *not* asked, because
    a person approved it at ``plugin.install``. A script was never approved by
    anybody, so this is the only moment there is to ask.
    """
    decision = _ask(write(tree, "heavy.py", extra="import numpy"))
    assert decision.level == UNSAFE
    assert "numpy" in decision.reason


def test_a_script_that_will_not_load_is_not_offered_for_approval(tree):
    """Approving something that cannot run is the worst thing to put in a dialog."""
    broken = write(tree, "heavy.py", extra="import os")
    assert _ask(broken).level == UNSAFE


def test_a_path_outside_a_scripts_directory_is_refused(tree):
    """The containment story rests entirely on where the file lives."""
    stray = tree / "tools" / "notascript.py"
    stray.parent.mkdir(parents=True)
    stray.write_text(SCRIPT.format(extra=""), encoding="utf-8")
    assert _ask(stray).level == UNSAFE


def test_naming_no_script_at_all_does_not_raise(tree):
    """``test_every_request_is_classified`` calls this with empty args."""
    decision = classify(Request(SCRIPT_RUN, {}), Chain())
    assert decision.level == UNSAFE
    assert "unclassified" not in decision.reason


def test_the_verdict_is_re_derived_never_supplied(tree):
    """A caller cannot talk its way past the import walk.

    The Request carries a path and nothing else that matters. Anything a guest
    could say about its own containment — a digest, a report, an ``isolation``
    line — is either absent or ignored, which is the same rule the tree rule
    enforces one level up.
    """
    heavy = write(tree, "heavy.py", extra="import numpy")
    assert _ask(heavy, unmediated=[], digest="whatever").level == UNSAFE


# ── running one ───────────────────────────────────────────────────────

@pytest.fixture
def sb():
    """A sandbox that refuses everything unsafe."""
    made = Sandbox()
    yield made
    made.shutdown()


def test_a_script_runs_and_answers_with_its_return_value(sb, tree):
    """No plugin contract anywhere in the file."""
    result = sb.run(str(write(tree)), "main", kwargs={"values": [1, 2, 3]})
    assert result.ok
    assert result.data == 6


def test_the_script_appears_in_its_own_chain(sb, tree):
    """What makes a ledger row worth reading: who caused this."""
    run = sb.start(str(write(tree)), "main", kwargs={"values": []},
                   chain=Chain(root="user").push("run_script"))
    run.wait()
    assert run.chain.render() == "user -> run_script -> tally"


# ── the caller going away ─────────────────────────────────────────────

SPINNER = '''\
def main(sdk):
    """Never finish."""
    while True:
        pass
'''


def test_cancelling_the_caller_tears_down_the_script(sb, tree, monkeypatch):
    """Cancellation reaches code that is *making* Requests. This makes none.

    The handler starts the script and then blocks, so a cancelled caller would
    otherwise sit here until the child hit its own ceiling — holding a pool
    worker and finishing work nobody is going to read. It is the same shape as
    the frozen-frontend bugs: the thread that would notice is the thread that
    is waiting.
    """
    import threading

    from sandbox import bridge, provenance
    from sandbox.handlers.kernel import _script_run
    from sandbox.interpreter import Execution

    spinner = tree / "scripts" / "tally.py"
    spinner.write_text(SPINNER, encoding="utf-8")
    monkeypatch.setattr("plugins.helpers.plugin_paths.ALLOWED_ROOTS",
                        (tree.parent.resolve(),))
    bridge.configure(sb)

    execution = Execution(name="caller", chain=Chain(root="user"))
    threading.Timer(0.5, lambda: setattr(execution, "cancelled", True)).start()

    with provenance.serving(execution.chain, None, execution):
        result = _script_run(None, {"path": str(spinner)})

    assert not result.ok
    assert "cancelled" in result.error


CALLER = '''\
requests = ["script.run"]

from guest.bases import BaseTool


class Caller(BaseTool):
    """Run a script the way a real tool would."""

    name = "caller"
    description = "x"

    def run(self, sdk, path):
        """Ask the kernel to run a script and hand back what it returned."""
        return sdk.scripts.run(path, values=[10, 20])
'''


def test_a_tool_runs_a_script_through_the_sdk(sb, tree, monkeypatch):
    """The real path, and the one that could have deadlocked.

    A tool is already inside a box when it asks, so the handler answering it
    starts a *second* box and blocks. That is only safe because the two live in
    different pools — the facade's background pool versus the interpreter's
    execution pool — and the chain bounds how deep it can go. Worth an actual
    nested run rather than an argument.
    """
    from sandbox import bridge

    tool = tree / "tools" / "tool_caller.py"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text(CALLER, encoding="utf-8")
    monkeypatch.setattr("plugins.helpers.plugin_paths.ALLOWED_ROOTS",
                        (tree.parent.resolve(),))
    bridge.configure(sb)

    try:
        result = sb.run(str(tool), "Caller", kwargs={"path": str(write(tree))},
                        chain=Chain(root="user"))
    finally:
        unload_box("tool_caller")

    assert result.ok, result.error
    assert result.data == 30


def test_a_handler_with_no_provenance_still_runs(sb, tree, monkeypatch):
    """``abandoned`` answers False when there is nothing to ask.

    Every test that calls a handler directly is in this position, and so is any
    future caller that has not been threaded through the interpreter. Reading
    "carry on" there is the only safe default — the alternative is a handler
    that refuses to work outside a context it cannot see.
    """
    from sandbox import bridge, provenance
    from sandbox.handlers.kernel import _script_run

    monkeypatch.setattr("plugins.helpers.plugin_paths.ALLOWED_ROOTS",
                        (tree.parent.resolve(),))
    bridge.configure(sb)

    assert provenance.current() is None
    result = _script_run(None, {"path": str(write(tree)),
                                "args": {"values": [4, 5]}})
    assert result.ok
    assert result.data == 9
