"""Contracts for the store tools used by the benchmark workflow."""

from __future__ import annotations

from types import SimpleNamespace

import sandbox  # noqa: F401 - installs the guest package alias
from guest.requests import Result

from tests.support import store_source


def _tool(relative: str, class_name: str):
    """Load a dependency-free store tool without importing the store tree."""
    namespace = {"__name__": f"test_{class_name.lower()}"}
    source = store_source(relative)
    exec(compile(source, relative, "exec"), namespace)
    return namespace[class_name]()


class _Paths:
    def get(self, name):
        return {"project": "C:/project", "scripts": "C:/scripts"}[name]


class _Files:
    def list(self, *args, **kwargs):
        return {"entries": []}


class _SDK:
    """The narrow result/process surface these contract tests exercise."""

    def __init__(self, *, done=None, status=None, mode="ask"):
        self.paths = _Paths()
        self.fs = _Files()
        self.session = SimpleNamespace(get=lambda: {"mode": mode})
        self.proc = SimpleNamespace(
            run=lambda *args, **kwargs: dict(done or {}),
            status=lambda *args, **kwargs: dict(status or {}),
        )

    def ok(self, data=None, llm_summary=""):
        return Result(data=data, llm_summary=llm_summary)

    def fail(self, error, retryable=False):
        return Result.failure(error, retryable=retryable)


def test_run_command_nonzero_foreground_exit_is_a_tool_failure():
    tool = _tool("tools/tool_run_command.py", "RunCommand")
    sdk = _SDK(done={"stdout": "partial", "stderr": "bad input", "code": 7})

    result = tool._foreground(sdk, "thing", "C:/project", "default", 60)

    assert not result.ok
    assert "partial" in result.error
    assert "bad input" in result.error
    assert "exit code 7" in result.error
    assert "cwd: C:/project" in result.error


def test_run_command_zero_foreground_exit_stays_successful():
    tool = _tool("tools/tool_run_command.py", "RunCommand")
    result = tool._foreground(
        _SDK(done={"stdout": "done", "stderr": "", "code": 0}),
        "thing", "C:/project", "default", 60)

    assert result.ok
    assert result.data["code"] == 0


def test_completed_background_failure_is_a_tool_failure():
    tool = _tool("tools/tool_run_command.py", "RunCommand")
    status = {"id": 3, "running": False, "code": 9, "command": "thing",
              "output": "trace", "log": "C:/logs/3.log", "label": ""}
    result = tool._one(_SDK(status=status), "check", 3)

    assert not result.ok
    assert "code 9" in result.error
    assert "trace" in result.error
    assert "C:/logs/3.log" in result.error


def test_running_background_check_stays_successful():
    tool = _tool("tools/tool_run_command.py", "RunCommand")
    status = {"id": 3, "running": True, "code": None, "command": "thing",
              "output": "working", "log": "C:/logs/3.log", "label": ""}
    assert tool._one(_SDK(status=status), "check", 3).ok


def test_shell_and_script_prompts_change_strategy_with_mode():
    shell = _tool("tools/tool_run_command.py", "RunCommand")
    script = _tool("tools/tool_run_script.py", "RunScript")
    assert "approval is refused" in shell.agent_prompt(_SDK(mode="lockdown"))
    assert "raw `.py`" in shell.agent_prompt(_SDK(mode="yolo"))
    assert "valid contained script still runs" in script.agent_prompt(
        _SDK(mode="lockdown"))
    assert "do not `import sdk`" in script.agent_prompt(_SDK(mode="ask"))


def test_glob_contributes_lockdown_guidance_only_in_lockdown():
    tool = _tool("tools/tool_glob.py", "GlobFiles")
    assert "directory discovery" in tool.agent_prompt(_SDK(mode="lockdown"))
    assert tool.agent_prompt(_SDK(mode="yolo")) == ""


def test_read_file_owns_lockdown_file_guidance():
    source = store_source("tools/tool_read_file.py")
    assert "def agent_prompt(self, sdk)" in source
    assert "use read_file for its contents" in source
