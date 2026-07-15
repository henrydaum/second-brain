"""Tests for the store edit_file tool: read-before-edit enforcement and
self-correcting replace errors.

read_file + edit_file + the file_reads helper are materialized as one
package so both tools share the same helper module (and its session bag).
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
def mods(tmp_path_factory):
    sources = {rel: _store_source(rel) for rel in (
        "tools/tool_edit_file.py", "tools/tool_read_file.py", "tools/helpers/file_reads.py")}
    if any(v is None for v in sources.values()):
        pytest.skip("edit_file package not present on a local store ref")
    root = tmp_path_factory.mktemp("edit_pkg")
    pkg = root / "edit_store_pkg"
    (pkg / "helpers").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helpers" / "__init__.py").write_text("", encoding="utf-8")
    for rel, src in sources.items():
        (pkg / Path(rel).relative_to("tools")).write_text(src, encoding="utf-8")
    sys.path.insert(0, str(root))
    try:
        edit = importlib.import_module("edit_store_pkg.tool_edit_file")
        read = importlib.import_module("edit_store_pkg.tool_read_file")
        reads = importlib.import_module("edit_store_pkg.helpers.file_reads")
    finally:
        sys.path.remove(str(root))
    return SimpleNamespace(edit=edit, read=read, reads=reads)


@pytest.fixture()
def env(mods, tmp_path, monkeypatch):
    """Approving context with a real session bag; edits confined to tmp_path."""
    monkeypatch.setattr(mods.edit, "ROOTS", (tmp_path.resolve(),))
    monkeypatch.setattr(mods.edit, "ROOT_DIR", tmp_path)
    session = SimpleNamespace(plugin_state={})
    context = SimpleNamespace(
        approve_command=lambda cmd, why: True,
        approval_denial_reason="",
        runtime=SimpleNamespace(sessions={"k": session}),
        session_key="k",
    )
    target = tmp_path / "sample.py"
    target.write_text("alpha\nbeta\ngamma\nbeta\n", encoding="utf-8")

    def edit(**kwargs):
        kwargs.setdefault("justification", "test")
        return mods.edit.EditFile().run(context, **kwargs)

    def read(path=None):
        return mods.read.ReadFile().run(context, path=str(path or target))

    return SimpleNamespace(edit=edit, read=read, context=context,
                           session=session, target=target, tmp=tmp_path, mods=mods)


def test_dependency_literals(mods):
    assert mods.edit.dependencies_files == ['tools/helpers/file_reads.py']
    assert mods.read.dependencies_files == ['tools/helpers/file_reads.py']
    assert mods.reads.dependencies_files == []


def test_replace_unread_fails(env):
    out = env.edit(operation="replace", path=str(env.target), old_text="alpha", new_text="omega")
    assert out.success is False
    assert "read_file" in out.error
    assert "alpha" in env.target.read_text(encoding="utf-8")


def test_read_then_replace_succeeds_and_refreshes(env):
    env.read()
    out = env.edit(operation="replace", path=str(env.target), old_text="alpha", new_text="omega")
    assert out.success is True
    # the edit itself refreshed the recorded mtime — an immediate second edit works
    out = env.edit(operation="replace", path=str(env.target), old_text="omega", new_text="alpha")
    assert out.success is True


def _touch(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))


def test_stale_with_unique_match_succeeds(env):
    env.read()
    _touch(env.target)
    out = env.edit(operation="replace", path=str(env.target), old_text="alpha", new_text="omega")
    assert out.success is True


def test_stale_with_ambiguous_match_fails(env):
    env.read()
    _touch(env.target)
    out = env.edit(operation="replace", path=str(env.target), old_text="beta", new_text="delta", replace_all=True)
    assert out.success is False
    assert "re-read" in out.error


def test_create_exempt_and_records(env):
    new = env.tmp / "brand_new.txt"
    out = env.edit(operation="create", path=str(new), content="hello\n")
    assert out.success is True
    # recorded by the create — an immediate edit needs no read_file call
    out = env.edit(operation="replace", path=str(new), old_text="hello", new_text="bye")
    assert out.success is True


def test_overwrite_append_delete_enforced(env):
    for op, extra in (("overwrite", {"content": "x"}), ("append", {"content": "x"}), ("delete", {})):
        out = env.edit(operation=op, path=str(env.target), **extra)
        assert out.success is False and "read_file" in out.error, op
    env.read()
    assert env.edit(operation="append", path=str(env.target), content="tail\n").success is True
    assert env.edit(operation="delete", path=str(env.target)).success is True
    assert env.mods.reads._key(env.target) not in env.session.plugin_state["file_reads"]


def test_not_found_quotes_closest_match(env):
    env.read()
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="alpha\nbetta\ngamma", new_text="x")
    assert out.success is False
    assert "Closest match (lines 1-3)" in out.error
    assert "alpha\nbeta\ngamma" in out.error


def test_not_found_garbage_stays_plain(env):
    env.read()
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="zzz qqq totally unrelated stuff", new_text="x")
    assert out.success is False
    assert "old_text was not found." in out.error
    assert "Closest match" not in out.error


def test_line_number_contamination_hint(env):
    env.read()
    out = env.edit(operation="replace", path=str(env.target),
                   old_text="1: alpha\n2: beta", new_text="x")
    assert out.success is False
    assert "line-number" in out.error
    # a file genuinely containing '1:' lines with matching old_text: no false refusal
    numbered = env.tmp / "numbered.txt"
    numbered.write_text("1: real content\n2: more\n", encoding="utf-8")
    env.read(numbered)
    out = env.edit(operation="replace", path=str(numbered), old_text="1: real content", new_text="x")
    assert out.success is True


def test_ambiguous_error_lists_lines(env):
    env.read()
    out = env.edit(operation="replace", path=str(env.target), old_text="beta", new_text="x")
    assert out.success is False
    assert "lines 2, 4" in out.error


def test_missing_session_skips_enforcement(env):
    env.context.runtime.sessions.clear()
    out = env.edit(operation="replace", path=str(env.target), old_text="alpha", new_text="omega")
    assert out.success is True


def test_cap_eviction(env, mods):
    for i in range(mods.reads.MAX_ENTRIES + 1):
        f = env.tmp / f"f{i}.txt"
        f.write_text("x", encoding="utf-8")
        mods.reads.record_read(env.context, f)
    bag = env.session.plugin_state["file_reads"]
    assert len(bag) == mods.reads.MAX_ENTRIES
    assert mods.reads._key(env.tmp / "f0.txt") not in bag
