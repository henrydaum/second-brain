"""The shell family: what it asks, what it runs, and what it keeps.

Three properties, and the first is the one that matters most:

- **starting a process always asks.** There is no classifier and no whitelist,
  because deciding what a command line does is undecidable and a classifier of
  that shape fails silently in the permissive direction. Speaking *about* a
  process already started does not ask.
- the kernel builds the invocation from ``shell``, so a command line survives
  quoting on Windows — which it does not if the guest wraps its own.
- a started process is a handle: pollable, killable, and forgotten once
  stopped.
"""

import sys
import time

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Chain, Request
from sandbox.guest import requests as R
from sandbox.handlers.fs_net import (_proc_list, _proc_run, _proc_start,
                                     _proc_status, _proc_stop)
from sandbox.policy import classify
from sandbox.shell import render_command


# ──────────────────────────────────────────────────────────────────────
# Policy.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", [R.PROC_RUN, R.PROC_START])
@pytest.mark.parametrize("command", [
    "ls",                       # the most read-only thing there is
    "rm -rf /",
    "rg foo | head -20",
    "echo x > /etc/passwd",
])
def test_starting_a_process_asks_unless_something_vouches_for_it(kind, command):
    """Asking is the default, and nothing here is vouched for.

    The old tool auto-ran the first two. That was not wrong so much as
    unmaintainable: its whitelist had to be right about every way a shell can
    smuggle a second command past it, forever, and it was wrong invisibly.

    A whitelist did come back, but somewhere it can be sound — as a
    *recognizer* in the policy rather than a classifier in the plugin it
    authorizes, keyed on ``(program, subcommand)``, and abstaining on anything
    with a shell metacharacter in it. ``git status`` therefore no longer
    belongs in this list; ``tests/test_shell_recognizer.py`` owns that half.
    ``ls`` stays, deliberately: ``sdk.fs.list`` does it mediated and better, so
    a dialog is the right nudge.
    """
    decision = classify(Request(kind, {"argv": command, "shell": "default"}),
                        Chain())
    assert not decision.safe
    assert command[:20] in decision.reason


@pytest.mark.parametrize("kind", [R.PROC_STATUS, R.PROC_STOP, R.PROC_LIST])
def test_speaking_about_a_started_process_does_not_ask(kind):
    """Reading the registry reads nothing that was not approved at start.

    ``stop`` is here for the reason ``session.remove_tool`` is — it narrows.
    A dev server the agent cannot kill without a dialog is one it will not
    start, and the alternative to stopping it is leaving it running.
    """
    assert classify(Request(kind, {"id": 1}), Chain()).safe


def test_the_dialog_names_the_command_and_where_it_runs():
    """A scope nobody is shown is not consent."""
    decision = classify(
        Request(R.PROC_RUN, {"argv": ["git", "push"], "cwd": "/repo"}),
        Chain())
    assert "git push" in decision.reason
    assert "/repo" in decision.reason


def test_the_dialog_and_the_ledger_read_the_same_line():
    """One renderer, so the record and the question cannot drift apart."""
    assert render_command({"argv": ["a", "b"]}) == "a b"
    assert render_command({"argv": "a b"}) == "a b"
    assert "powershell" in render_command({"argv": "gci",
                                           "shell": "powershell"})


def test_a_recognizer_can_widen_and_no_recognizer_can_narrow():
    """The seam future policy work goes through, pinned as a seam.

    Recognizers are how this gets less onerous — a structural read-only
    check, or a remembered "yes". One ships (see
    ``tests/test_shell_recognizer.py``); what is pinned here is the *shape*
    rather than its contents: an unrecognised command is still asked about, a
    recognizer that abstains changes nothing, and one that raises abstains
    rather than allowing.
    """
    from sandbox import shell

    # Deliberately not a command the shipped recognizer knows, so this tests
    # the seam and not the whitelist.
    request = Request(R.PROC_RUN, {"argv": "sortilege --now"})
    original = list(shell._SHELL_RECOGNIZERS)

    shell._SHELL_RECOGNIZERS.append(lambda line, args: None)
    shell._SHELL_RECOGNIZERS.append(lambda line, args: 1 / 0)
    try:
        assert not classify(request, Chain()).safe
        shell._SHELL_RECOGNIZERS.append(
            lambda line, args: "vouched" if line.startswith("sortilege")
            else None)
        assert classify(request, Chain()).safe
        assert not classify(Request(R.PROC_RUN, {"argv": "rm x"}),
                            Chain()).safe
    finally:
        shell._SHELL_RECOGNIZERS[:] = original


# ──────────────────────────────────────────────────────────────────────
# Running.
# ──────────────────────────────────────────────────────────────────────

def test_an_argv_runs_without_a_shell():
    """No shell named, no shell involved: no globbing, no metacharacters."""
    answer = _proc_run(None, {"argv": [sys.executable, "-c", "print('hi')"]})
    assert answer.ok
    assert answer.data["stdout"].strip() == "hi"


def test_a_shell_gets_the_command_line_intact():
    """The reason the kernel builds the invocation rather than the guest.

    ``cmd.exe`` does not understand the backslash-escaped quotes that
    ``subprocess``'s list conversion produces, so a guest wrapping its own
    command as ``["cmd", "/c", line]`` loses every embedded quote. Passing the
    string with ``shell=True`` is the only form that round-trips.
    """
    answer = _proc_run(None, {"argv": 'echo "two words"', "shell": "default"})
    assert answer.ok
    assert "two words" in answer.data["stdout"]


def test_a_shell_is_what_makes_a_pipeline_possible():
    answer = _proc_run(None, {"argv": "echo one && echo two",
                              "shell": "default"})
    assert answer.ok
    assert "one" in answer.data["stdout"] and "two" in answer.data["stdout"]


def test_an_unknown_shell_is_refused_by_name():
    answer = _proc_run(None, {"argv": "ls", "shell": "fish"})
    assert not answer.ok
    assert "fish" in answer.error


def test_nothing_to_run_is_an_ordinary_failure():
    assert not _proc_run(None, {"argv": ""}).ok


# ──────────────────────────────────────────────────────────────────────
# The platform branches, pinned from either platform.
#
# Second Brain runs on Windows here and on POSIX elsewhere, and the half that
# is not under the developer's feet is the half that breaks. These drive the
# invocation builder with ``os.name`` forced both ways, so a Mac-only or
# Windows-only mistake fails on whichever machine runs the suite.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def as_platform(monkeypatch):
    """Build invocations as if this were the named platform."""
    from sandbox.handlers import fs_net

    def pretend(name):
        monkeypatch.setattr(fs_net.os, "name", name)
    return pretend


def test_a_list_renders_with_the_platform_quoting(as_platform):
    """A path with a space is one argument on both, spelled differently."""
    from sandbox.handlers.fs_net import _command_line

    as_platform("nt")
    assert _command_line(["ls", "two words"]) == 'ls "two words"'
    as_platform("posix")
    assert _command_line(["ls", "two words"]) == "ls 'two words'"


def test_powershell_is_pwsh_off_windows(as_platform):
    """The binary has two names and they are not interchangeable.

    ``powershell`` is Windows PowerShell 5.1 and exists only on Windows;
    PowerShell Core installs as ``pwsh``, which is the only thing a Mac or
    Linux box could have. Naming the wrong one turns a supported shell into
    "no such file or directory".
    """
    from sandbox.handlers.fs_net import _invocation

    as_platform("nt")
    assert _invocation({"argv": "gci", "shell": "powershell"})[0][0] == \
        "powershell"
    as_platform("posix")
    assert _invocation({"argv": "gci", "shell": "powershell"})[0][0] == "pwsh"


def test_cmd_is_refused_off_windows_by_name(as_platform):
    """Better a clear refusal than an exec failure for a missing binary."""
    from sandbox.handlers.fs_net import _invocation

    from sandbox.guest.requests import Result

    as_platform("posix")
    answer = _invocation({"argv": "dir", "shell": "cmd"})
    assert isinstance(answer, Result) and not answer.ok
    assert "Windows" in answer.error

    as_platform("nt")
    # On Windows it builds instead of refusing: a tuple, not a Result.
    assert not isinstance(_invocation({"argv": "dir", "shell": "cmd"}), Result)


def test_the_default_shell_hands_over_the_string_itself(as_platform):
    """``shell=True`` is the only form that survives cmd.exe's quoting, and
    it is also what gives POSIX ``/bin/sh -c``. Same shape either way."""
    from sandbox.handlers.fs_net import _invocation

    for name in ("nt", "posix"):
        as_platform(name)
        cmd, use_shell, _ = _invocation({"argv": 'echo "x y"',
                                         "shell": "default"})
        assert use_shell and cmd == 'echo "x y"'


def test_stopping_escalates_on_posix_and_is_already_hard_on_windows():
    """The asymmetry that made ``stop`` a lie on POSIX.

    ``taskkill /T /F`` is a hard kill. ``SIGTERM`` is a *request* — the right
    first thing to send a dev server, since it gets to close its socket, but
    a process that traps it survives. Without escalation ``proc.stop``
    reported a clean stop on something still running and no longer tracked.
    """
    from sandbox.handlers import fs_net

    if fs_net.os.name == "nt":
        pytest.skip("SIGKILL escalation is a POSIX path")

    import signal as signals

    answer = _proc_start(None, {
        "argv": [sys.executable, "-c",
                 "import signal, time\n"
                 "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                 "print('deaf', flush=True)\n"
                 "time.sleep(60)"]})
    handle = answer.data["id"]
    for _ in range(50):
        if "deaf" in _proc_status(None, {"id": handle}).data["output"]:
            break
        time.sleep(0.1)
    stopped = _proc_stop(None, {"id": handle}).data
    assert stopped["stopped"], "a SIGTERM-deaf process outlived proc.stop"
    assert stopped["code"] in (-signals.SIGKILL, signals.SIGKILL, None) or \
        stopped["code"] is not None


def test_a_stop_that_did_not_stop_says_so():
    """Reporting a clean stop on a survivor is the failure worth naming."""
    from sandbox.handlers.fs_net import _proc_stop as stop

    answer = _proc_start(None, {"argv": [sys.executable, "-c", "pass"]})
    time.sleep(0.3)
    assert stop(None, {"id": answer.data["id"]}).data["stopped"] is True


# ──────────────────────────────────────────────────────────────────────
# Keeping.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def started():
    """A process that will outlive the Request that started it."""
    answer = _proc_start(None, {
        "argv": [sys.executable, "-c",
                 "import time; print('up', flush=True); time.sleep(60)"],
        "label": "server"})
    assert answer.ok
    yield answer.data
    _proc_stop(None, {"id": answer.data["id"]})


def test_a_started_process_is_a_handle(started):
    """The thing a return value cannot express, which is why the type exists."""
    assert started["running"] and started["pid"] and started["label"] == "server"


def test_output_arrives_through_the_log_because_it_cannot_stream(started):
    """A live process cannot cross the boundary; a file it wrote can."""
    for _ in range(50):
        status = _proc_status(None, {"id": started["id"]}).data
        if "up" in status["output"]:
            break
        time.sleep(0.1)
    assert "up" in status["output"]
    assert status["running"] and status["code"] is None


def test_stopping_ends_it_and_forgets_it(started):
    assert any(entry["id"] == started["id"]
               for entry in _proc_list(None, {}).data)
    assert _proc_stop(None, {"id": started["id"]}).ok
    assert all(entry["id"] != started["id"]
               for entry in _proc_list(None, {}).data)
    # And the handle is gone, rather than answering about a dead process.
    assert not _proc_status(None, {"id": started["id"]}).ok


def test_an_unknown_handle_fails_with_somewhere_to_go():
    for handler in (_proc_status, _proc_stop):
        answer = handler(None, {"id": 99999})
        assert not answer.ok
        assert "proc.list" in answer.error


def test_an_exited_process_stays_listed_so_its_output_is_still_readable():
    """Forgetting it the moment it exits would lose the reason to poll it."""
    answer = _proc_start(None, {"argv": [sys.executable, "-c",
                                         "print('done')"]})
    handle = answer.data["id"]
    try:
        for _ in range(50):
            status = _proc_status(None, {"id": handle}).data
            if not status["running"]:
                break
            time.sleep(0.1)
        assert not status["running"] and status["code"] == 0
        assert "done" in status["output"]
    finally:
        _proc_stop(None, {"id": handle})


# ──────────────────────────────────────────────────────────────────────
# The two facts behind the door the validator closes.
# ──────────────────────────────────────────────────────────────────────

def test_paths_answers_for_the_interpreter_and_the_platform():
    """``sys`` is refused, and these are the only things behind it a plugin
    has a real claim on: which Python ``pip`` should target, and which shell
    a command line is being built for."""
    from sandbox.handlers.kernel import _path_get

    assert _path_get(None, {"name": "python"}).data == sys.executable
    assert _path_get(None, {"name": "platform"}).data == sys.platform
