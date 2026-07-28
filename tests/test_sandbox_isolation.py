"""Isolation is decided by provenance, and code gets no vote.

``isolation = "subprocess"`` used to be a declaration read off the file, which
made the code being contained the authority on its own containment. These pin
the replacement: the tree a file lives in decides, and a file cannot move
itself between trees by writing anything.
"""

from pathlib import Path

import pytest

from paths import INSTALLED_PLUGINS, ROOT_DIR, SANDBOX_PLUGINS
from sandbox.guest.box import IN_PROCESS, SUBPROCESS
from sandbox.isolation import (INSTALLED, KERNEL, SANDBOX, UNKNOWN,
                               required_isolation, tree_of)
from sandbox.policy import SAFE, UNSAFE, Chain, classify
from sandbox.guest.requests import (FS_DELETE, FS_MOVE, FS_WRITE, Request)
from sandbox.validator import validate_file

TOOL = '''\
from guest.bases import BaseTool
requests = []
{extra}

class SampleTool(BaseTool):
    name = "sample"
    description = "x"

    def run(self, sdk):
        return 1
'''


def _write(path: Path, extra: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TOOL.format(extra=extra), encoding="utf-8")
    return path


# ── the tree decides ──────────────────────────────────────────────────

@pytest.mark.parametrize("root, tree, expected", [
    (SANDBOX_PLUGINS / "tools", SANDBOX, SUBPROCESS),
    (ROOT_DIR / "plugins" / "tools", KERNEL, IN_PROCESS),
])
def test_the_tree_decides_isolation(root, tree, expected):
    """Agent-authored code is contained; the kernel's own is not.

    Neither reads anything out of the file, so neither can be talked out of.
    """
    source = root / "tool_sample.py"
    assert tree_of(source) == tree
    assert required_isolation(source, None) == expected


def test_an_unknown_path_fails_closed():
    """"I do not know what this is" has exactly one safe answer."""
    assert tree_of("C:/somewhere/else/tool_x.py") == UNKNOWN
    assert required_isolation("C:/somewhere/else/tool_x.py", None) == SUBPROCESS


def test_a_file_cannot_declare_its_way_out_of_a_subprocess(tmp_path,
                                                           monkeypatch):
    """The headline. This is the vulnerability the rework closes.

    An agent authoring a plugin could previously author its own escape from
    the process boundary by leaving a line out — or putting one in.
    """
    tree = tmp_path / "sandbox_plugins"
    monkeypatch.setattr("paths.SANDBOX_PLUGINS", tree)

    source = _write(tree / "tools" / "tool_sample.py",
                    'isolation = "in_process"')
    assert required_isolation(source, validate_file(source)) == SUBPROCESS


def test_isolation_is_not_a_declaration_any_more(tmp_path, monkeypatch):
    """Reading it at all would be at best ignored, at worst believed.

    Asserted against the collector rather than against a list of keys: the
    collector takes every literal class attribute, so "not in the allowlist"
    was never what kept ``isolation`` out — dropping it explicitly is.
    """
    tree = tmp_path / "sandbox_plugins"
    monkeypatch.setattr("paths.SANDBOX_PLUGINS", tree)
    source = _write(tree / "tools" / "tool_sample.py",
                    'isolation = "in_process"')
    report = validate_file(source)

    assert "isolation" not in report.declarations
    # And the author is told once, at the line, rather than left to wonder.
    assert any("isolation" in f.message for f in report.findings)
    assert "isolation" not in validate_file(
        Path("plugins/commands/command_update.py")).declarations


# ── installed packages are judged on what they import ─────────────────

def _installed(tmp_path, monkeypatch, extra):
    monkeypatch.setattr("paths.INSTALLED_PLUGINS", tmp_path / "installed")
    source = _write(tmp_path / "installed" / "tools" / "tool_sample.py", extra)
    return source, validate_file(source)


def test_a_pure_store_package_runs_in_process(tmp_path, monkeypatch):
    """Pure computation over the SDK is as inspectable as kernel code."""
    source, report = _installed(tmp_path, monkeypatch, "import json, re")
    assert tree_of(source) == INSTALLED
    assert report.unmediated == frozenset()
    assert required_isolation(source, report) == IN_PROCESS


@pytest.mark.parametrize("imports, found", [
    ("import fitz", "fitz"),          # a foreign library
    ("import zipfile", "zipfile"),    # stdlib that does its own path I/O
])
def test_a_store_package_reaching_past_the_sdk_is_isolated(
        tmp_path, monkeypatch, imports, found):
    """The case the security contract names: a component nobody can validate."""
    source, report = _installed(tmp_path, monkeypatch, imports)
    assert found in report.unmediated
    assert required_isolation(source, report) == SUBPROCESS


def test_the_foreign_check_is_computed_not_declared(tmp_path, monkeypatch):
    """``dependencies_pip`` would reintroduce the same bug one level down.

    A package that imports a foreign library while declaring no dependencies
    is still isolated, because the answer comes from the import walk.
    """
    source, report = _installed(tmp_path, monkeypatch,
                                "dependencies_pip = []\nimport fitz")
    assert required_isolation(source, report) == SUBPROCESS


def test_an_unreadable_report_isolates(tmp_path, monkeypatch):
    """No verdict about the contents is not the same as a clean verdict."""
    monkeypatch.setattr("paths.INSTALLED_PLUGINS", tmp_path / "installed")
    source = tmp_path / "installed" / "tools" / "tool_x.py"
    assert required_isolation(source, None) == SUBPROCESS


# ── what the boundary buys: free authorship in the agent's own tree ───

def _chain():
    return Chain(root="user").push("authoring_tool")


@pytest.mark.parametrize("kind, args", [
    (FS_WRITE, {"path": str(SANDBOX_PLUGINS / "tools" / "tool_new.py")}),
    (FS_DELETE, {"path": str(SANDBOX_PLUGINS / "tools" / "tool_old.py")}),
    (FS_MOVE, {"src": str(SANDBOX_PLUGINS / "tools" / "a.py"),
               "dst": str(SANDBOX_PLUGINS / "tools" / "b.py")}),
])
def test_the_agent_writes_its_own_plugins_without_asking(kind, args):
    """This is what the subprocess buys.

    Code under this tree is contained before it runs, so a dialog per edit
    would interrupt a dozen times to approve something that cannot act
    unmediated anyway.
    """
    assert classify(Request(kind, args), _chain()).level == SAFE


def test_the_grant_is_that_tree_and_no_other():
    """Free authorship is scoped to where containment applies."""
    outside = str(ROOT_DIR / "plugins" / "tools" / "tool_kernel.py")
    assert classify(Request(FS_WRITE, {"path": outside}),
                    _chain()).level == UNSAFE


def test_writing_code_freely_does_not_widen_what_it_may_do():
    """The LibOS invariant, stated as a test.

    A plugin may change what it can *ask* for without a dialog. What it may
    *affect* is unchanged: its Requests are classified like anybody else's,
    and it inherits nothing from having been written without one.
    """
    from sandbox.guest.requests import NET_HTTP, PROC_RUN
    authored = Chain(root="user").push("tool_the_agent_just_wrote")

    assert classify(Request(NET_HTTP, {"url": "https://x.invalid"}),
                    authored).level == UNSAFE
    assert classify(Request(PROC_RUN, {"argv": ["git", "push"]}),
                    authored).level == UNSAFE


def test_the_ledger_can_tell_authoring_from_scratch():
    """Two grants share a branch; reading back what happened should not guess."""
    reason = classify(Request(FS_WRITE, {
        "path": str(SANDBOX_PLUGINS / "tools" / "tool_new.py")}),
        _chain()).reason
    assert "plugin tree" in reason
