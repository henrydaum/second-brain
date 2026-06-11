"""Tests for the store MCP server frontend (``frontends/frontend_mcp_server.py``).

The package lives on the ``store`` branch, not in the kernel tree, so the
module is materialized from the local store ref via ``git show`` and loaded
with importlib — mirroring what ``/packages install`` would copy. Skips
cleanly when no store ref is available (e.g. shallow CI checkouts). The
``mcp`` pip package is only needed by ``start()``; the session/identity and
tool-body behavior under test here is plain kernel plumbing.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.database import DEFAULT_USER_ID, Database

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401
from runtime.conversation_runtime import ConversationRuntime

_REPO = Path(__file__).resolve().parents[1]
_STORE_REL = "frontends/frontend_mcp_server.py"


def _store_module_source() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_STORE_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def frontend_cls(tmp_path_factory):
    source = _store_module_source()
    if source is None:
        pytest.skip(f"{_STORE_REL} not present on a local store ref")
    path = tmp_path_factory.mktemp("mcp_frontend") / "frontend_mcp_server.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("frontend_mcp_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MCPServerFrontend


def _bound_frontend(frontend_cls, tmp_path):
    db = Database(str(tmp_path / "mcp.db"))
    rt = ConversationRuntime(db=db, services={}, config={})
    fe = frontend_cls()
    fe.bind(rt, None, {})
    return fe, rt, db


def test_declares_store_contract(frontend_cls):
    assert frontend_cls.name == "mcp_server"
    assert frontend_cls.user_binding == "per_user"
    src = _store_module_source()
    assert 'dependencies_pip = ["mcp"]' in src  # module-level literal for AST parsing


def test_default_client_acts_as_operator_and_unattended(frontend_cls, tmp_path):
    fe, rt, _db = _bound_frontend(frontend_cls, tmp_path)

    key = fe._session_for("default")

    assert key == "mcp:default"
    assert rt.session_user_id(key) == DEFAULT_USER_ID
    assert rt.is_attended(key) is False  # MCP clients are agents, not humans


def test_named_client_gets_isolated_user(frontend_cls, tmp_path):
    fe, rt, db = _bound_frontend(frontend_cls, tmp_path)

    key = fe._session_for("claude-code")

    assert key == "mcp:claude-code"
    uid = rt.session_user_id(key)
    assert uid != DEFAULT_USER_ID
    assert db.get_user_by_external("mcp_server", "claude-code")["id"] == uid


def test_browse_tools_respect_conversation_ownership(frontend_cls, tmp_path):
    fe, rt, db = _bound_frontend(frontend_cls, tmp_path)
    mine = db.create_conversation("mine", user_id=DEFAULT_USER_ID)
    db.save_message(mine, "user", "hello")
    db.save_message(mine, "assistant", "hi!")
    other = db.upsert_user("web", "someone-else")
    theirs = db.create_conversation("theirs", user_id=other)

    listing = fe._list_conversations("default")
    assert f"{mine}: mine" in listing
    assert "theirs" not in listing

    assert "[user] hello" in fe._read_conversation(mine, "default")
    assert fe._read_conversation(theirs, "default") == "No such conversation."


def test_ask_returns_friendly_error_without_llm(frontend_cls, tmp_path):
    """No LLM configured: the turn fails, but ask returns text instead of
    raising, never leaves the session mid-phase, and stays unattended."""
    fe, rt, db = _bound_frontend(frontend_cls, tmp_path)
    cid = db.create_conversation("chat", user_id=DEFAULT_USER_ID)
    rt.load_conversation("mcp:default", cid)

    reply = fe._ask("hello?", "default")

    assert isinstance(reply, str) and reply
    assert rt.is_attended("mcp:default") is False
    # The MCP submission must not have stolen the global active-session slot.
    assert rt.active_session_key != "mcp:default"
