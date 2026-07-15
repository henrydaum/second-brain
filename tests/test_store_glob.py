"""Tests for the store glob tool (filename search)."""

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
    tool_src = _store_source("tools/tool_glob.py")
    helper_src = _store_source("tools/helpers/file_walk.py")
    if tool_src is None or helper_src is None:
        pytest.skip("glob package not present on a local store ref")
    root = tmp_path_factory.mktemp("glob_pkg")
    pkg = root / "glob_store_pkg"
    (pkg / "helpers").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tool_glob.py").write_text(tool_src, encoding="utf-8")
    (pkg / "helpers" / "file_walk.py").write_text(helper_src, encoding="utf-8")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("glob_store_pkg.tool_glob")
    finally:
        sys.path.remove(str(root))
    return module


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("glob_tree")
    (root / "top.py").write_text("x", encoding="utf-8")
    (root / "top.txt").write_text("x", encoding="utf-8")
    deep = root / "src" / "inner"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("x", encoding="utf-8")
    junk = root / "__pycache__"
    junk.mkdir()
    (junk / "cached.py").write_text("x", encoding="utf-8")
    now = 1_700_000_000
    os.utime(root / "top.py", (now, now))
    os.utime(deep / "deep.py", (now + 10, now + 10))
    return root


@pytest.fixture()
def glob(mod, tree, monkeypatch):
    fw = sys.modules["glob_store_pkg.helpers.file_walk"]
    monkeypatch.setattr(fw, "ALLOWED_ROOTS", {tree.resolve()})
    monkeypatch.setattr(fw, "ROOT_DIR", tree)
    def call(**kwargs):
        return mod.GlobFiles().run(SimpleNamespace(), **kwargs)
    return call


def test_dependency_literals(mod):
    assert mod.dependencies_files == ['tools/helpers/file_walk.py']
    assert mod.dependencies_pip == []


def test_root_confinement(glob, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    out = glob(pattern="*.py", path=str(outside))
    assert out.success is False and "outside" in out.error


def test_star_is_top_level_only(glob):
    assert glob(pattern="*.py").data["results"] == ["top.py"]


def test_doublestar_matches_any_depth(glob):
    out = glob(pattern="**/*.py")
    results = out.data["results"]
    assert set(results) == {"top.py", "src/inner/deep.py"}
    # newest-first: deep.py has the later mtime
    assert results[0] == "src/inner/deep.py"
    assert not any("__pycache__" in r for r in results)


def test_symlinked_dir_not_walked(glob, tree):
    target = tree.parent / "sym_target"
    target.mkdir(exist_ok=True)
    (target / "linked.py").write_text("x", encoding="utf-8")
    link = tree / "sym"
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation not permitted on this system")
    try:
        assert "sym/linked.py" not in glob(pattern="**/*.py").data["results"]
    finally:
        link.unlink()


def test_limit_truncation(glob):
    out = glob(pattern="**/*.py", limit=1)
    assert len(out.data["results"]) == 1
    assert out.data["truncated"] is True
    assert "more exist" in out.llm_summary


def test_no_matches_is_success(glob):
    out = glob(pattern="*.nope")
    assert out.success is True
    assert "No files matched" in out.llm_summary
