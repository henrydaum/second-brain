"""Tests for the store todo tool (per-conversation checklist).

Runs against a real Database so the FK cascade and ensure_output_table
paths are exercised, with a SimpleNamespace session carrying the cid.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_REL = "tools/tool_todo.py"


def _store_source() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def mod(tmp_path_factory):
    src = _store_source()
    if src is None:
        pytest.skip("todo package not present on a local store ref")
    path = tmp_path_factory.mktemp("todo_pkg") / "tool_todo.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("tool_todo_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def env(mod, tmp_path):
    from pipeline.database import Database
    db = Database(str(tmp_path / "todo.db"))
    cid = db.create_conversation("test convo")
    context = SimpleNamespace(
        db=db,
        runtime=SimpleNamespace(sessions={"k": SimpleNamespace(conversation_id=cid)}),
        session_key="k",
        user_id=1,
    )
    def call(**kwargs):
        return mod.Todo().run(context, **kwargs)
    return SimpleNamespace(call=call, db=db, cid=cid, context=context)


def _count(db, cid):
    with db.lock:
        return db.conn.execute(
            "SELECT COUNT(*) FROM todos WHERE conversation_id = ?", (cid,)).fetchone()[0]


def test_dependency_literals(mod):
    assert mod.dependencies_files == []
    assert mod.dependencies_pip == []


def test_add_and_bulk_add(env):
    out = env.call(operation="add", content="first step")
    assert out.success is True
    assert "- [ ] #1 first step" in out.llm_summary
    out = env.call(operation="add", items=["second", "third"])
    assert _count(env.db, env.cid) == 3
    assert "3 open" in out.llm_summary


def test_update_complete_and_reword(env):
    env.call(operation="add", items=["a", "b"])
    out = env.call(operation="update", todo_id=1, status="in_progress")
    assert "**a** (in progress)" in out.llm_summary
    out = env.call(operation="complete", todo_id=1)
    assert "- [x] #1 a" in out.llm_summary
    out = env.call(operation="update", todo_id=2, content="b reworded")
    assert "#2 b reworded" in out.llm_summary


def test_remove_and_unknown_id(env):
    env.call(operation="add", content="only")
    assert env.call(operation="remove", todo_id=1).success is True
    assert _count(env.db, env.cid) == 0
    out = env.call(operation="remove", todo_id=99)
    assert out.success is False and "#99" in out.error


def test_per_conversation_isolation(env):
    env.call(operation="add", content="mine")
    other_cid = env.db.create_conversation("other convo")
    env.context.runtime.sessions["k"].conversation_id = other_cid
    out = env.call(operation="list")
    assert "(empty)" in out.llm_summary
    # ids from another conversation are invisible
    assert env.call(operation="complete", todo_id=1).success is False


def test_cap(env, mod):
    items = [f"item {i}" for i in range(mod.MAX_TODOS)]
    assert env.call(operation="add", items=items).success is True
    out = env.call(operation="add", content="one too many")
    assert out.success is False and "cap" in out.error.lower()


def test_no_conversation_is_graceful(mod, env):
    context = SimpleNamespace(db=env.db, runtime=SimpleNamespace(sessions={}), session_key=None, user_id=1)
    out = mod.Todo().run(context, operation="list")
    assert out.success is False
    assert "conversation" in out.error


def test_cascade_on_conversation_delete(env):
    env.call(operation="add", items=["a", "b"])
    assert _count(env.db, env.cid) == 2
    env.db.delete_conversation(env.cid)
    assert _count(env.db, env.cid) == 0
