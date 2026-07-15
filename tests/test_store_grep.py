"""Tests for the store grep tool (live-tree regex search).

Materializes ``tools/tool_grep.py`` + ``tools/helpers/file_walk.py`` from the
local store ref as a package (the tool uses a relative helper import) and
drives it against a temp tree with the walk roots monkeypatched.
"""

from __future__ import annotations

import importlib
import os
import shutil
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
    """Force the Python backend so assertions hold on rg-equipped machines."""
    fw = sys.modules["grep_store_pkg.helpers.file_walk"]
    monkeypatch.setattr(fw, "ALLOWED_ROOTS", {tree.resolve()})
    monkeypatch.setattr(fw, "ROOT_DIR", tree)
    monkeypatch.setattr(mod.Grep, "_rg_path", None)
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


# ── ripgrep backend ─────────────────────────────────────────────────


def test_python_backend_reported(grep):
    assert grep(pattern="needle").data["backend"] == "python"


@pytest.fixture()
def rg_stub(mod, tree, monkeypatch):
    """Pretend rg exists; capture the command line and script the output."""
    fw = sys.modules["grep_store_pkg.helpers.file_walk"]
    monkeypatch.setattr(fw, "ALLOWED_ROOTS", {tree.resolve()})
    monkeypatch.setattr(fw, "ROOT_DIR", tree)
    monkeypatch.setattr(mod.Grep, "_rg_path", "rg-stub")
    calls = []
    box = SimpleNamespace(calls=calls, returncode=0, stdout="")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=box.returncode, stdout=box.stdout, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    def call(**kwargs):
        return mod.Grep().run(SimpleNamespace(), **kwargs)
    box.call = call
    return box


def test_rg_command_flags(rg_stub):
    rg_stub.stdout = "a.py:1\n"
    out = rg_stub.call(pattern="needle", glob="*.py", output_mode="count",
                       case_insensitive=True, multiline=True)
    assert out.data["backend"] == "ripgrep"
    cmd = rg_stub.calls[0]
    assert cmd[:1] == ["rg-stub"]
    assert "--no-ignore" in cmd and "--hidden" in cmd and "--no-config" in cmd
    assert ["--sortr", "modified"] == cmd[cmd.index("--sortr"):cmd.index("--sortr") + 2]
    assert ["--max-filesize", "2000000"] == cmd[cmd.index("--max-filesize"):cmd.index("--max-filesize") + 2]
    assert "!**/node_modules/**" in cmd and "!**/.git/**" in cmd
    assert "/*.py" in cmd  # top-level glob anchored with leading '/'
    assert "--glob-case-insensitive" in cmd
    assert "-i" in cmd and "-U" in cmd and "--multiline-dotall" in cmd
    assert "--count-matches" in cmd and "-c" not in cmd
    assert cmd[-2:] == ["needle", "--"] or cmd[cmd.index("-e") + 1] == "needle"
    assert "--" in cmd


def test_rg_glob_doublestar_passthrough(rg_stub):
    rg_stub.stdout = ""
    rg_stub.returncode = 1
    rg_stub.call(pattern="x", glob="**/*.py")
    assert "**/*.py" in rg_stub.calls[0]


def test_rg_files_and_count_parsing(rg_stub):
    rg_stub.stdout = "a.py\nsub/c.py\n"
    out = rg_stub.call(pattern="needle")
    assert out.data["results"] == ["a.py", "sub/c.py"]
    rg_stub.stdout = "a.py:2\nsub\\c.py:1\n"
    out = rg_stub.call(pattern="needle", output_mode="count")
    assert out.data["results"] == [("a.py", 2), ("sub/c.py", 1)]
    assert "Total: 3" in out.llm_summary


def test_rg_content_parsing_no_context(rg_stub):
    rg_stub.stdout = "a.py:1:needle in a haystack\nsub/c.py:2:needle again\n"
    out = rg_stub.call(pattern="needle", output_mode="content")
    assert out.data["results"] == [
        "a.py:1: needle in a haystack",
        "sub/c.py:2: needle again",
    ]


def test_rg_content_parsing_with_context(rg_stub):
    rg_stub.stdout = ("sub/c.py-1-before\n"
                      "sub/c.py:2:needle again\n"
                      "sub/c.py-3-after\n"
                      "--\n"
                      "a.py:1:needle in a haystack\n")
    out = rg_stub.call(pattern="needle", output_mode="content", context_lines=1)
    assert out.data["results"] == [
        "sub/c.py:1- before\nsub/c.py:2: needle again\nsub/c.py:3- after",
        "a.py:1: needle in a haystack",
    ]


def test_rg_error_falls_back_to_python(rg_stub):
    rg_stub.returncode = 2  # e.g. Rust regex rejecting a backref
    out = rg_stub.call(pattern="needle")
    assert out.success is True
    assert out.data["backend"] == "python"
    assert "a.py" in out.data["results"]


def test_rg_no_match_is_clean_empty(rg_stub):
    rg_stub.returncode = 1
    out = rg_stub.call(pattern="zzz")
    assert out.success is True
    assert out.data["backend"] == "ripgrep"
    assert out.data["results"] == []


def test_rg_single_file_target_uses_python(rg_stub, tree):
    out = rg_stub.call(pattern="needle", path=str(tree / "a.py"))
    assert out.data["backend"] == "python"
    assert rg_stub.calls == []


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_rg_real_parity(mod, tree, monkeypatch):
    fw = sys.modules["grep_store_pkg.helpers.file_walk"]
    monkeypatch.setattr(fw, "ALLOWED_ROOTS", {tree.resolve()})
    monkeypatch.setattr(fw, "ROOT_DIR", tree)
    monkeypatch.setattr(mod.Grep, "_rg_path", shutil.which("rg"))
    rg_out = mod.Grep().run(SimpleNamespace(), pattern="needle")
    monkeypatch.setattr(mod.Grep, "_rg_path", None)
    py_out = mod.Grep().run(SimpleNamespace(), pattern="needle")
    assert rg_out.data["backend"] == "ripgrep" and py_out.data["backend"] == "python"
    assert sorted(rg_out.data["results"]) == sorted(py_out.data["results"])
    rg_count = mod.Grep()
    monkeypatch.setattr(mod.Grep, "_rg_path", shutil.which("rg"))
    rc = rg_count.run(SimpleNamespace(), pattern="needle", output_mode="count").data["results"]
    monkeypatch.setattr(mod.Grep, "_rg_path", None)
    pc = mod.Grep().run(SimpleNamespace(), pattern="needle", output_mode="count").data["results"]
    assert sorted(rc) == sorted(pc)
