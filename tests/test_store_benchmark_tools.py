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


def test_the_mode_only_prompts_declare_the_session_cue():
    """Three tools whose prompt reads the mode and nothing else.

    Left on the default rung they recompute on every ``fs.write`` in the
    process — a fresh box each, to re-read something that changes once a
    conversation. This is the case ``prompt_cues`` was built for, so it is
    worth stating that they actually claim it.
    """
    from sandbox.validator import validate_file

    for relative in ("tools/tool_glob.py", "tools/tool_read_file.py",
                     "tools/tool_run_command.py"):
        source = store_source(relative)
        assert 'agent_prompt_refresh = "session"' in source, relative
        assert "sdk.session.get()" in source, relative


def test_the_live_listing_prompts_stay_on_the_default_rung():
    """And these must not: their text is a directory and a table list.

    Declaring anything rarer would leave the agent reading a listing that
    predates the file it just wrote — the exact failure the write rung exists
    to prevent, so the absence is deliberate and pinned.
    """
    for relative in ("tools/tool_run_script.py", "tools/tool_sql_query.py"):
        source = store_source(relative)
        assert "agent_prompt_refresh" not in source.replace("# ", ""), relative


def test_grep_completes_the_lockdown_trio():
    """glob and read_file already steer off the shell; grep did not.

    In lockdown the agent was told to use glob for listing and read_file for
    contents, and nothing about content *search* — the one of the three whose
    shell alternative is most tempting and most likely to be refused.
    """
    tool = _tool("tools/tool_grep.py", "Grep")
    guidance = tool.agent_prompt(_SDK(mode="lockdown"))
    assert "grep for content search" in guidance
    assert tool.agent_prompt(_SDK(mode="yolo")) == ""


def _hybrid(indexed, scope=("hybrid_search", "lexical_search",
                            "semantic_search")):
    """Hybrid search plus an SDK answering a corpus size and a tool scope."""
    namespace = {"__name__": "test_hybridsearch"}
    source = store_source("tools/tool_hybrid_search.py").replace(
        "from .tool_lexical_search import _search_summary",
        "_search_summary = None")
    exec(compile(source, "tool_hybrid_search.py", "exec"), namespace)

    sdk = _SDK()
    sdk.Failed = Exception
    sdk.tools = SimpleNamespace(list=lambda details=False: list(scope))
    sdk.db = SimpleNamespace(
        query=lambda sql: [{"n": indexed}] if indexed is not None
        else (_ for _ in ()).throw(Exception("no such table")))
    return namespace["HybridSearch"](), sdk


def test_search_guidance_names_an_empty_corpus():
    """An empty result reads exactly like the fact not existing.

    Both sub-tools comment on this failure in their own source; the prompt
    asserted "three retrieval tools search the indexed corpus" regardless, so
    an agent searched three ways over nothing and concluded the information
    was not there.
    """
    tool, sdk = _hybrid(0)
    guidance = tool.agent_prompt(sdk)
    assert "Nothing is indexed yet" in guidance
    assert "not a missing fact" in guidance
    assert "hybrid_search:" not in guidance


def test_search_guidance_counts_what_is_indexed():
    tool, sdk = _hybrid(412)
    assert "412 documents indexed" in tool.agent_prompt(sdk)


def test_search_guidance_names_only_the_tools_in_scope():
    """Installing lexical_search alone is legitimate, and used to say nothing.

    The guidance lives on hybrid_search, which pulls the other two as
    dependencies — one-directional, so a lighter install got no guidance and a
    partial one got told about tools it did not have.
    """
    tool, sdk = _hybrid(412, scope=("hybrid_search", "lexical_search"))
    guidance = tool.agent_prompt(sdk)
    assert "- lexical_search:" in guidance
    assert "- semantic_search:" not in guidance


def test_search_guidance_does_not_claim_an_empty_corpus_it_cannot_verify():
    """A missing table is not proof of nothing indexed — the same wrong
    conclusion in the other direction, and the reason ``_indexed`` answers
    None rather than 0."""
    tool, sdk = _hybrid(None)
    guidance = tool.agent_prompt(sdk)
    assert "Nothing is indexed yet" not in guidance
    assert "Search covers your sync_directories" in guidance
