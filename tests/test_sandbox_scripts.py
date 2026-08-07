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
from tests.support import retarget_trees
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
    """A workspace tree with a scripts/ directory in it."""
    root = retarget_trees(monkeypatch, tmp_path)["workspace"]
    (root / "scripts").mkdir(parents=True)
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


# ── how long a script may take ────────────────────────────────────────

def test_a_script_declares_its_own_deadline(sb, tree):
    """A script has no class, so its declarations are module-level.

    Undeclared it gets the ordinary default, which is deliberately modest and
    still short for the work scripts are actually reached for — a crawl that
    fans out subagents does real computation between waits, and *cumulative*
    CPU across a long run is what breaches a deadline like this. ``box`` is
    already declared this way, so nothing new is being invented here; what was
    missing was anybody knowing it worked, which is why it is pinned rather
    than left to ``_prepare``.

    The clamp is the point of the last case: a plugin may ask for a longer
    leash, it does not get to grant itself one.
    """
    from sandbox.interpreter import (DEFAULT_TIMEOUT_SECONDS,
                                     MAX_TIMEOUT_SECONDS, clamp_timeout)

    def deadline(extra):
        """The seconds the runner would actually enforce."""
        _, _, opts = sb._prepare(str(write(tree, extra=extra)))
        return clamp_timeout(opts["timeout"])

    assert deadline("") == DEFAULT_TIMEOUT_SECONDS
    assert deadline("timeout = 300") == 300.0
    assert deadline("timeout = 5000") == MAX_TIMEOUT_SECONDS


BURNER = '''\
{extra}

def main(sdk):
    """Burn CPU: running time, which is what a deadline measures."""
    import time
    end = time.monotonic() + 3
    while time.monotonic() < end:
        pass
    return "finished"
'''


def test_a_declared_deadline_is_what_the_runner_enforces(sb, tree):
    """Resolution is not enforcement, and only one of them is the promise.

    Declared small so it bites in a second: the default is far longer, so a
    script that dies after 1s can only have died on the number it declared.
    The reverse case — a long declaration outliving the default — is the same
    wiring read from the other end and would cost the suite a minute to say so.
    """
    path = tree / "scripts" / "tally.py"
    path.write_text(BURNER.format(extra="timeout = 1"), encoding="utf-8")

    result = sb.run(str(path), "main", chain=Chain(root="user"))

    assert not result.ok
    assert "timed out after 1.0s" in result.error


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


# ── naming one the way an agent does ──────────────────────────────────
#
# The failure these pin cost a whole Telegram session: an agent wrote a script,
# named it the two obvious ways, and was refused both times — once as "not in a
# scripts/ directory" (the checkout's own ``scripts/``, which is not a tree
# root) and once as "no such script" (a project-root file that never existed).
# The general resolver prefers the project root for a bare relative path, which
# is right for a plugin and wrong for a script.

@pytest.mark.parametrize("named", ["tally.py", "scripts/tally.py"])
def test_a_script_is_found_by_the_names_an_agent_gives_it(tree, named):
    """A bare filename and a ``scripts/``-relative path both resolve."""
    from sandbox.isolation import resolve_script

    expected = write(tree)
    assert resolve_script(named) == expected.resolve()


def test_a_name_that_is_not_a_script_reference_is_left_alone(tree):
    """None means "not mine", so the general resolver still gets its turn."""
    from sandbox.isolation import resolve_script

    write(tree)
    assert resolve_script("tools/tally.py") is None
    assert resolve_script(str(tree / "scripts" / "tally.py")) is None
    assert resolve_script("") is None


def test_a_missing_script_is_not_invented(tree):
    """Resolution is by existence: nothing there means nothing found."""
    from sandbox.isolation import resolve_script

    assert resolve_script("absent.py") is None


@pytest.mark.parametrize("named", ["tally.py", "scripts/tally.py"])
def test_the_policy_resolves_the_name_the_handler_will(tree, named):
    """Both halves must agree, or the dialog is about a path that runs fine.

    Classifying the raw argument while the handler resolved it is how a
    correctly-named script came to be asked about and *then* work — the exact
    complaint scripts exist to remove, one layer down.
    """
    write(tree)
    assert _ask(named).level == SAFE


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

BUSY = '''\
def main(sdk):
    """Never finish, but keep asking the kernel things.

    The difference from ``SPINNER`` is where cancellation lands. A cancelled
    guest learns at its **next Request**, which raises ``Terminated``; code
    that makes none can only be *killed*, and killing waits for the deadline —
    the 60s default, since a script that declares no ``timeout`` gets
    ``interpreter.DEFAULT_TIMEOUT_SECONDS``.

    That kill path has its own test (``test_a_runaway_is_actually_killed``),
    which buys it for 2s by declaring a one-second timeout. Spending another
    sixty here to re-prove it — while the claim under test is that *stopping*
    reaches a detached run — made one test half of the whole suite.
    """
    while True:
        sdk.fs.list(".")
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
    monkeypatch.setattr("plugins.plugin_paths.ALLOWED_ROOTS",
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
    monkeypatch.setattr("plugins.plugin_paths.ALLOWED_ROOTS",
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

    monkeypatch.setattr("plugins.plugin_paths.ALLOWED_ROOTS",
                        (tree.parent.resolve(),))
    bridge.configure(sb)

    assert provenance.current() is None
    result = _script_run(None, {"path": str(write(tree)),
                                "args": {"values": [4, 5]}})
    assert result.ok
    assert result.data == 9


def test_a_bare_name_runs(sb, tree, monkeypatch):
    """End to end on the name the agent actually typed."""
    from sandbox import bridge
    from sandbox.handlers.kernel import _script_run

    write(tree)
    monkeypatch.setattr("plugins.plugin_paths.ALLOWED_ROOTS",
                        (tree.parent.resolve(),))
    bridge.configure(sb)

    result = _script_run(None, {"path": "tally.py",
                                "args": {"values": [1, 2, 3]}})
    assert result.ok, result.error
    assert result.data == 6


# ── detaching one, and coming back for it ─────────────────────────────
#
# ``wait=False`` shipped answering ``{"started": True}`` and dropping the
# ``Run`` on the floor, so a detached script could never be collected,
# cancelled, or even observed to have finished. Both of its siblings already
# had the shape it was missing — ``agent.spawn``/``collect``/``stop`` for a
# background agent, ``proc.start``/``status``/``stop`` for a background
# process — and the argument is the one ``proc.start`` already makes: work that
# outlives the Request that started it needs a handle.


@pytest.fixture
def wired(sb, tree, monkeypatch):
    """The global sandbox pointed at this tree, as a real caller would find it."""
    from sandbox import bridge

    monkeypatch.setattr("plugins.plugin_paths.ALLOWED_ROOTS",
                        (tree.parent.resolve(),))
    bridge.configure(sb)
    return sb


def _as(chain, fn, args):
    """Call a handler with provenance, the way the interpreter does."""
    from sandbox import provenance
    from sandbox.interpreter import Execution

    execution = Execution(name="caller", chain=chain)
    with provenance.serving(chain, None, execution):
        return fn(None, args)


def test_a_detached_script_hands_back_an_id(wired, tree):
    """The handle. Without it there is nothing to name the run by later."""
    from sandbox.handlers.kernel import _script_run

    result = _as(Chain(root="user"), _script_run,
                 {"path": str(write(tree)), "args": {"values": [1]},
                  "wait": False})

    assert result.ok, result.error
    assert result.data["id"]
    # The two keys this answered before there was an id. Kept, because
    # something reading them should not have to change to gain a third.
    assert result.data["started"] is True
    assert result.data["script"] == "tally.py"


def test_several_run_at_once_and_are_collected_together(wired, tree):
    """The fan-out, which is the whole reason to detach one.

    Each is a box of its own, so these genuinely run in parallel — and unlike a
    subagent, none of it involves a model. That is the gap: fanning work out
    used to mean spawning agents even when the work was ordinary code.
    """
    from sandbox.handlers.kernel import _script_collect, _script_run

    chain = Chain(root="user")
    path = str(write(tree))
    for values in ([1], [2, 3], [4, 5, 6]):
        started = _as(chain, _script_run,
                      {"path": path, "args": {"values": values},
                       "wait": False})
        assert started.ok, started.error

    collected = _as(chain, _script_collect, {"ids": None})

    assert collected.ok, collected.error
    assert sorted(r["data"] for r in collected.data) == [1, 5, 15]
    assert {r["state"] for r in collected.data} == {"done"}
    # The same string ``script.run`` answered with. Two Requests about one run
    # must not name it two different ways.
    assert {r["script"] for r in collected.data} == {"tally.py"}


def test_a_report_is_delivered_once(wired, tree):
    """Two collectors both acting on one answer is the worse failure.

    "Did I already handle this?" is not a question the caller can answer from
    outside, so the registry answers it — the same one-shot delivery a subagent
    report has, and for the same reason.
    """
    from sandbox.handlers.kernel import _script_collect, _script_run

    chain = Chain(root="user")
    _as(chain, _script_run, {"path": str(write(tree)),
                             "args": {"values": [7]}, "wait": False})

    first = _as(chain, _script_collect, {"ids": None})
    second = _as(chain, _script_collect, {"ids": None})

    assert [r["data"] for r in first.data] == [7]
    assert second.data == []


def test_polling_leaves_an_unfinished_run_collectable(wired, tree):
    """``timeout=0`` is a look, not a wait.

    A run still going comes back ``running`` and *stays* in the registry, so
    the poll that found it unfinished has not consumed it. Getting this wrong
    would lose the result of every slow run a fan-out checked on early.
    """
    from sandbox.handlers.kernel import _script_collect, _script_run

    slow = tree / "scripts" / "tally.py"
    slow.write_text(
        "import time\n\n\ndef main(sdk):\n"
        '    """Take a moment."""\n'
        "    time.sleep(1.5)\n"
        "    return 'eventually'\n", encoding="utf-8")

    chain = Chain(root="user")
    _as(chain, _script_run, {"path": str(slow), "wait": False})

    polled = _as(chain, _script_collect, {"ids": None, "timeout": 0})
    assert [r["state"] for r in polled.data] == ["running"]
    assert polled.data[0]["data"] is None

    waited = _as(chain, _script_collect, {"ids": None})
    assert [r["data"] for r in waited.data] == ["eventually"]


def test_one_caller_cannot_collect_anothers_runs(wired, tree):
    """Ownership is the chain *root* — what caused the work.

    The root rather than the innermost link, because two scripts started by one
    turn should be collectable together and the root is the only part of a
    chain both share. It is also the part a guest cannot state about itself, so
    a box cannot claim somebody else's results by asking nicely.
    """
    from sandbox.handlers.kernel import _script_collect, _script_run

    mine = Chain(root="user").push("run_script")
    theirs = Chain(root="cron:nightly").push("task_index")
    _as(mine, _script_run, {"path": str(write(tree)),
                            "args": {"values": [9]}, "wait": False})

    assert _as(theirs, _script_collect, {"ids": None}).data == []
    assert [r["data"] for r in _as(mine, _script_collect,
                                   {"ids": None}).data] == [9]


def test_stopping_a_detached_script_reaches_it(wired, tree):
    """Narrows, so it is the safe direction — and it has to actually work.

    A fan-out the caller cannot abandon is one it will not start, which is the
    argument ``proc.stop`` already makes about a dev server.

    The script is ``BUSY`` rather than ``SPINNER`` deliberately — see that
    constant. What is under test here is that ``script.stop`` reaches a
    detached run, not how long the kernel takes to kill code that refuses to
    ask it anything.
    """
    from sandbox.handlers.kernel import _script_collect, _script_run, _script_stop

    spinner = tree / "scripts" / "tally.py"
    spinner.write_text(BUSY, encoding="utf-8")

    chain = Chain(root="user")
    started = _as(chain, _script_run, {"path": str(spinner), "wait": False})
    stopped = _as(chain, _script_stop, {"id": started.data["id"]})

    assert stopped.data is True
    report = _as(chain, _script_collect, {"ids": [started.data["id"]]})
    assert report.data[0]["state"] == "cancelled"


def test_stopping_a_run_that_is_not_yours_does_nothing(wired, tree):
    """Answers False rather than raising: there is no such run, to you."""
    from sandbox.handlers.kernel import _script_run, _script_stop

    started = _as(Chain(root="user"), _script_run,
                  {"path": str(write(tree)), "wait": False})
    refused = _as(Chain(root="cron:nightly"), _script_stop,
                  {"id": started.data["id"]})

    assert refused.data is False


def test_a_waited_run_is_not_kept_for_collection(wired, tree):
    """Nothing to come back for, so nothing is retained.

    Worth pinning as a *negative*: retaining every run would make the registry
    grow with ordinary tool and command work, which is a leak with no symptom
    until the process runs out of memory.
    """
    from sandbox.handlers.kernel import _script_collect, _script_run

    chain = Chain(root="user")
    done = _as(chain, _script_run, {"path": str(write(tree)),
                                    "args": {"values": [1, 2]}})

    assert done.data == 3
    assert _as(chain, _script_collect, {"ids": None}).data == []


FANOUT = '''\
requests = ["script.run", "script.collect"]

from guest.bases import BaseTool


class Fanout(BaseTool):
    """Start several scripts at once and collect them."""

    name = "fanout"
    description = "x"

    def run(self, sdk, path):
        """Three at once, then the answers."""
        ids = [sdk.scripts.run(path, wait=False, values=[n])["id"]
               for n in (1, 2, 3)]
        return sorted(r["data"] for r in sdk.scripts.collect(ids))
'''


def test_the_sdk_reaches_all_of_it_from_inside_a_box(wired, tree):
    """End to end through the SDK, which is the only path that matters.

    The handler tests above call in directly for precision; this is the shape a
    plugin author actually writes, and it is the one that would break if the
    Request names, the argument spelling or the answer shape drifted.
    """
    fanout = tree / "tools" / "tool_fanout.py"
    fanout.parent.mkdir(parents=True, exist_ok=True)
    fanout.write_text(FANOUT, encoding="utf-8")

    try:
        result = wired.run(str(fanout), "Fanout",
                           kwargs={"path": str(write(tree))},
                           chain=Chain(root="user"))
    finally:
        unload_box("tool_fanout")

    assert result.ok, result.error
    assert result.data == [1, 2, 3]
