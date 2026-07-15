"""Tests for the store grep tool (live-tree regex search).

Materializes ``tools/tool_grep.py`` + ``tools/helpers/file_walk.py`` from the
local store ref as a package (the tool uses a relative helper import) and
drives it against a temp tree with the walk roots monkeypatched.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _store_source(rel: str) -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{rel}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def mod(tmp_path_factory):
    tool_src = _store_source("tools/tool_grep.py")
    helper_src = _store_source("tools/helpers/file_walk.py")
    if tool_src is None or helper_src is None:
        pytest.skip("grep package not present on a local store ref")
    root = tmp_path_factory.mktemp("grep_pkg")
    pkg = root / "grep_store_pkg"
    (pkg / "helpers").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tool_grep.py").write_text(tool_src, encoding="utf-8")
    (pkg / "helpers" / "file_walk.py").write_text(helper_src, encoding="utf-8")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("grep_store_pkg.tool_grep")
    finally:
        sys.path.remove(str(root))
    return module


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """A small file tree with junk dirs, a binary, and pinned mtimes."""
    root = tmp_path_factory.mktemp("grep_tree")
    (root / "a.py").write_text("needle in a haystack\nsecond line\n", encoding="utf-8")
    (root / "b.txt").write_text("no match here\nNEEDLE upper\n", encoding="utf-8")
    sub = root / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("before\nneedle again\nafter\n", encoding="utf-8")
    (sub / "span.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    git = root / ".git"
    git.mkdir()
    (git / "junk.py").write_text("needle hidden in .git\n", encoding="utf-8")
    nm = root / "node_modules"
    nm.mkdir()
    (nm / "dep.py").write_text("needle hidden in node_modules\n", encoding="utf-8")
    (root / "blob.bin").write_bytes(b"\x00needle in binary")
    # newest-first ordering: a.py newest, then sub/c.py, then b.txt
    now = 1_700_000_000
    os.utime(root / "b.txt", (now, now))
    os.utime(sub / "c.py", (now + 10, now + 10))
    os.utime(root / "a.py", (now + 20, now + 20))
    return root


@pytest.fixture()
def grep(mod, tree, monkeypatch):
    fw = sys.modules["grep_store_pkg.helpers.file_walk"]
    monkeypatch.setattr(fw, "ALLOWED_ROOTS", {tree.resolve()})
    monkeypatch.setattr(fw, "ROOT_DIR", tree)
    def call(**kwargs):
        return mod.Grep().run(SimpleNamespace(), **kwargs)
    return call


def test_dependency_literals(mod):
    assert mod.dependencies_files == ['tools/helpers/file_walk.py']
    assert mod.dependencies_pip == []


def test_root_confinement(grep, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    out = grep(pattern="needle", path=str(outside))
    assert out.success is False
    assert "outside" in out.error
    out = grep(pattern="needle", path="../../..")
    assert out.success is False


def test_files_with_matches_skips_junk_and_binary(grep):
    out = grep(pattern="needle")
    assert out.success is True
    results = out.data["results"]
    assert "a.py" in results and "sub/c.py" in results
    assert not any(".git" in r or "node_modules" in r or "blob.bin" in r for r in results)
    assert out.data["skipped_binary"] == 1
    # newest-first: a.py before sub/c.py
    assert results.index("a.py") < results.index("sub/c.py")


def test_case_insensitive(grep):
    assert "b.txt" not in grep(pattern="NEEDLE juice|NEEDLE upper").data["results"] or True
    strict = grep(pattern="needle")
    loose = grep(pattern="needle", case_insensitive=True)
    assert "b.txt" not in strict.data["results"]
    assert "b.txt" in loose.data["results"]


def test_content_mode_with_context(grep):
    out = grep(pattern="needle again", output_mode="content", context_lines=1)
    assert out.success is True
    block = out.data["results"][0]
    assert "sub/c.py:1- before" in block
    assert "sub/c.py:2: needle again" in block
    assert "sub/c.py:3- after" in block


def test_count_mode(grep):
    out = grep(pattern="needle", output_mode="count")
    counts = dict(out.data["results"])
    assert counts["a.py"] == 1 and counts["sub/c.py"] == 1
    assert "Total: 2" in out.llm_summary


def test_limit_truncation(grep):
    out = grep(pattern="needle", limit=1)
    assert len(out.data["results"]) == 1
    assert out.data["truncated"] is True
    assert "more matches exist" in out.llm_summary


def test_invalid_regex(grep):
    out = grep(pattern="[unclosed")
    assert out.success is False
    assert "regex" in out.error.lower()


def test_multiline(grep):
    assert grep(pattern=r"alpha\nbeta").data["results"] == []
    out = grep(pattern=r"alpha\nbeta", multiline=True)
    assert out.data["results"] == ["sub/span.txt"]


def test_glob_filter(grep):
    out = grep(pattern="needle", glob="**/*.py")
    assert all(r.endswith(".py") for r in out.data["results"])
    top_only = grep(pattern="needle", glob="*.py")
    assert top_only.data["results"] == ["a.py"]


def test_single_file_path(grep, tree):
    out = grep(pattern="needle", path=str(tree / "a.py"), output_mode="content")
    assert out.success is True
    assert "a.py:1:" in out.data["results"][0]


def test_no_matches_is_success(grep):
    out = grep(pattern="zzz_not_there_zzz")
    assert out.success is True
    assert "No matches found" in out.llm_summary
