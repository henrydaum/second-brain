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
from sandbox.policy import classify, render_command


# ──────────────────────────────────────────────────────────────────────
# Policy.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", [R.PROC_RUN, R.PROC_START])
@pytest.mark.parametrize("command", [
    "ls",                       # the most read-only thing there is
    "git status",               # and the second
    "rm -rf /",
    "rg foo | head -20",
    "echo x > /etc/passwd",
])
def test_starting_a_process_always_asks(kind, command):
    """No whitelist, no exceptions, no argument about what `ls` does.

    The old tool auto-ran the first two. That was not wrong so much as
    unmaintainable: the whitelist has to be right about every way a shell can
    smuggle a second command past it, forever, and it is wrong invisibly.
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


def test_a_recognizer_can_widen_and_an_empty_list_cannot():
    """The seam future policy work goes through, pinned as a seam.

    Recognizers are how this gets less onerous — a structural read-only
    check, or a remembered "yes". The list ships empty, so the default is
    still "ask", and a recognizer that raises abstains rather than allowing.
    """
    from sandbox import policy

    assert policy._SHELL_RECOGNIZERS == []
    request = Request(R.PROC_RUN, {"argv": "ls"})

    policy._SHELL_RECOGNIZERS.append(lambda line, args: None)
    policy._SHELL_RECOGNIZERS.append(lambda line, args: 1 / 0)
    try:
        assert not classify(request, Chain()).safe
        policy._SHELL_RECOGNIZERS.append(
            lambda line, args: "read-only" if line == "ls" else None)
        assert classify(request, Chain()).safe
        assert not classify(Request(R.PROC_RUN, {"argv": "rm x"}),
                            Chain()).safe
    finally:
        policy._SHELL_RECOGNIZERS.clear()


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
