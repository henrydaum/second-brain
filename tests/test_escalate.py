"""Tests for the escalation stack: the turn-restart primitive, per-enact
model recording, and the store Escalate package
(``services/service_escalate.py``, loaded off the local store ref).

The kernel pieces are generic — a ``model_call`` escort retargets
``request.llm`` per call (contract pinned in tests/test_hooks_moments.py), a
tool asks for the turn to be re-driven (``session.restart_turn``), and every
agent enact records which model actually made it (ledger ``data_json.llm``).
The store package composes all three into the weak-model cascade.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from pipeline.database import Database
from runtime import runtime_config as _cfg
from runtime.conversation_runtime import ConversationRuntime
from runtime.hooks import HookContext, ModelRequest
from runtime.ledger import record_enact

_REPO = Path(__file__).resolve().parents[1]
_STORE_REL = "services/service_escalate.py"


# ─────────────────────────────────────────────────────────────────────────
# Shared fakes
# ─────────────────────────────────────────────────────────────────────────

def _response(content="", tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        is_error=False,
        prompt_tokens=0,
    )


class _FakeLLM:
    """Scripted LLM service: pops one queued response per call."""

    context_size = 0
    is_llm_backend = True

    def __init__(self, model_name, responses=None):
        self.model_name = model_name
        self.loaded = True
        self._responses = list(responses or [])
        self.calls = []

    def load(self):
        self.loaded = True
        return True

    def chat_with_tools(self, messages, tools, attachments=None):
        self.calls.append(list(messages))
        if not self._responses:
            return _response(content=f"{self.model_name} says done.")
        return self._responses.pop(0)


def _runtime(tmp_path, services=None, config=None):
    db = Database(str(tmp_path / "escalate.db"))
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db, services=services or {}, config=config or {})
    session = rt.load_conversation("s", cid)
    return rt, session, db


# ─────────────────────────────────────────────────────────────────────────
# Kernel A: the turn-restart primitive
# (Per-call brain swapping — the model_call escort — is pinned in
# tests/test_hooks_moments.py; active_llm is profile resolution only.)
# ─────────────────────────────────────────────────────────────────────────

def test_restart_turn_is_not_persisted(tmp_path):
    rt, session, _ = _runtime(tmp_path)
    session.restart_turn = True
    assert "restart_turn" not in session.to_marker()


def test_restart_turn_re_drives_once(tmp_path):
    rt, session, _ = _runtime(tmp_path)
    turns = []

    def fake_drive(sess, out, allow_restart=True):
        turns.append(len(turns) + 1)
        if len(turns) == 1:
            sess.restart_turn = True  # what the escalate tool does
        out.messages.append(f"reply {len(turns)}")
        sess.cs.set_priority("user")
        sess.busy = False
        return out

    rt._drive_agent_turn = fake_drive
    out = rt.handle_action("s", "send_text", "hello")

    assert turns == [1, 2]
    assert not session.restart_turn
    assert "reply 1" in out.messages and "reply 2" in out.messages


def test_restart_turn_ping_pong_is_bounded(tmp_path):
    rt, session, _ = _runtime(tmp_path)
    turns = []

    def hostile_drive(sess, out, allow_restart=True):
        turns.append(1)
        sess.restart_turn = True  # requests a restart every single time
        sess.cs.set_priority("user")
        sess.busy = False
        return out

    rt._drive_agent_turn = hostile_drive
    rt.handle_action("s", "send_text", "hello")

    assert len(turns) <= 5  # the drives cap keeps a restart-happy plugin finite


def test_restart_skips_end_turn_and_keeps_agent_priority(tmp_path):
    """A real drive: the tool sets restart_turn; the loop must exit without
    enacting end_turn (agent keeps priority for the re-drive), and the
    re-driven loop — its calls retargeted by a model_call escort — finishes
    the turn."""

    def escalate_handler(cs, actor, args):
        session.restart_turn = True
        rt.update_session_plugin_state("s", "esc", {"pending": True})
        from plugins.BaseTool import ToolResult
        return ToolResult(llm_summary="Escalating.")

    weak = _FakeLLM("weak", [
        _response(tool_calls=[{"id": "c1", "name": "escalate", "arguments": "{}"}]),
    ])
    strong = _FakeLLM("strong", [_response(content="Strong answer.")])

    rt, session, db = _runtime(tmp_path, services={"llm": weak, "strong": strong})

    def escort(ctx, request, proceed):
        if (ctx.session.plugin_state.get("esc") or {}).get("pending"):
            request.llm = strong
        return proceed(request)

    rt.hooks.add("model_call", escort)

    # Wire a registry exposing the escalate tool spec through the state machine.
    from state_machine.conversation import CallableSpec

    class _Reg:
        max_tool_calls = 5
        tools = {}

        def get_all_schemas(self):
            return [{"type": "function",
                     "function": {"name": "escalate", "parameters": {}}}]

    rt.tool_registry = _Reg()

    real_build_loop = _cfg.build_loop

    def build_loop_with_tool(runtime, session_key=None):
        loop = real_build_loop(runtime, session_key)
        # Hand the participant a live handler for the escalate spec.
        sess = runtime.sessions[session_key]
        agent = next(p for p in sess.cs.participants.values() if p.kind == "agent")
        agent.tools = {"escalate": CallableSpec("escalate", handler=escalate_handler)}
        sess.cs.cache["agent_scoped_tool_names"] = ["escalate"]
        return loop

    import runtime.conversation_runtime as _crt
    original = _crt._cfg.build_loop
    _crt._cfg.build_loop = build_loop_with_tool
    try:
        out = rt.handle_action("s", "send_text", "do something hard")
    finally:
        _crt._cfg.build_loop = original

    # One call per brain: the weak drive's escalate round, then the re-driven
    # turn's call retargeted onto the strong model by the escort.
    assert [len(weak.calls), len(strong.calls)] == [1, 1]
    assert "Strong answer." in out.messages
    assert session.cs.turn_priority == "user"
    # The weak partial turn stayed in history: assistant tool-call, tool result,
    # then the strong answer.
    roles = [m["role"] for m in session.history]
    assert "tool" in roles
    assert session.history[-1] == {"role": "assistant", "content": "Strong answer."}
    # Exactly one end_turn was recorded (the strong drive's).
    rows = db.conn.execute(
        "SELECT COUNT(*) FROM action_ledger WHERE action_type='end_turn' AND origin='agent_enact'"
    ).fetchone()
    assert rows[0] == 1
    # The ledger shows the flip: weak enacts, then strong enacts.
    models = [json.loads(r[0])["llm"] for r in db.conn.execute(
        "SELECT data_json FROM action_ledger WHERE origin='agent_enact' ORDER BY id").fetchall()]
    assert "weak" in models and "strong" in models


def test_crashed_drive_voids_restart_and_returns_priority(tmp_path):
    weak = _FakeLLM("weak")
    rt, session, _ = _runtime(tmp_path, services={"llm": weak})

    def exploding_build_loop(runtime, session_key=None):
        runtime.sessions[session_key].restart_turn = True
        raise RuntimeError("boom")

    import runtime.conversation_runtime as _crt
    original = _crt._cfg.build_loop
    _crt._cfg.build_loop = exploding_build_loop
    try:
        out = rt.handle_action("s", "send_text", "hello")
    finally:
        _crt._cfg.build_loop = original

    assert not out.ok
    assert not session.restart_turn
    assert session.cs.turn_priority == "user"


# ─────────────────────────────────────────────────────────────────────────
# Kernel C: ledger records the model
# ─────────────────────────────────────────────────────────────────────────

def test_record_enact_data_lands_in_data_json(tmp_path):
    db = Database(str(tmp_path / "ledger.db"))
    ok = SimpleNamespace(ok=True, error=None, data={})
    record_enact(db, origin="agent_enact", session_key="s", conversation_id=None,
                 user_id=None, actor_id="agent", action_type="send_text",
                 content="hi", result=ok, data={"llm": "weak"})
    row = db.conn.execute("SELECT data_json FROM action_ledger").fetchone()
    assert json.loads(row[0]) == {"llm": "weak"}


def test_record_enact_tolerates_stub_db_with_data():
    record_enact(None, origin="agent_enact", session_key="s", conversation_id=None,
                 user_id=None, actor_id="agent", action_type="send_text",
                 content="hi", result=None, data={"llm": "x"})  # must not raise


def test_agent_enacts_record_model_name(tmp_path):
    weak = _FakeLLM("weak", [_response(content="hello back")])
    rt, session, db = _runtime(tmp_path, services={"llm": weak})
    rt.handle_action("s", "send_text", "hi")
    rows = db.conn.execute(
        "SELECT data_json FROM action_ledger WHERE origin='agent_enact'").fetchall()
    assert rows
    for (data_json,) in rows:
        assert json.loads(data_json)["llm"] == "weak"


# ─────────────────────────────────────────────────────────────────────────
# The store Escalate package
# ─────────────────────────────────────────────────────────────────────────

def _store_module_source() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_STORE_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


@pytest.fixture(scope="module")
def escalate_module(tmp_path_factory):
    source = _store_module_source()
    if source is None:
        pytest.skip(f"{_STORE_REL} not present on a local store ref")
    path = tmp_path_factory.mktemp("escalate_pkg") / "service_escalate.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("service_escalate_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _escalate_runtime(tmp_path, escalate_module, weak_responses=None, strong_responses=None,
                      config=None):
    weak = _FakeLLM("weak", weak_responses)
    strong = _FakeLLM("strong", strong_responses)
    config = {"escalate_strong_llm": "strong",
              "llm_profiles": {"weak": {}, "strong": {}},
              "default_llm_profile": "weak", **(config or {})}
    services = {"llm": weak, "weak": weak, "strong": strong}
    svc = escalate_module.build_services(config)["escalate"]
    services["escalate"] = svc
    rt, session, db = _runtime(tmp_path, services=services, config=config)
    svc.bind_runtime(runtime=rt)
    svc.load()
    return rt, session, db, svc, weak, strong


class _SchemaRegistry:
    """Minimal ToolRegistry stand-in that scope shapers can clone."""

    max_tool_calls = 5

    def __init__(self, db, config, services):
        self.db, self.config, self.services = db, config, services
        self.orchestrator = None
        self.runtime = None
        self.tools = {}
        self.visible_tool_names = set()

    def get_all_schemas(self):
        return [
            {"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
            for name, t in self.tools.items() if name in self.visible_tool_names
        ]


def test_declares_store_contract(escalate_module):
    src = _store_module_source()
    assert "dependencies_files = []" in src
    assert "dependencies_pip = []" in src
    assert escalate_module.EscalateService.lifecycle == "extension"
    assert escalate_module.EscalateTool.max_calls == 1
    assert escalate_module.EscalateTool.background_safe is False


def test_shaper_offers_tool_only_to_weak_sessions(tmp_path, escalate_module):
    rt, session, db, svc, weak, strong = _escalate_runtime(tmp_path, escalate_module)
    base = _SchemaRegistry(db, rt.config, rt.services)

    shaped = rt.hooks.shape_scope(session, base)
    assert "escalate" in shaped.tools and "escalate" in shaped.visible_tool_names

    # Point the session's profile at the strong model: tool disappears.
    rt.config["agent_profiles"] = {"pro": {"llm": "strong"}}
    session.profile_override = "pro"
    shaped = rt.hooks.shape_scope(session, base)
    assert "escalate" not in shaped.tools


def test_shaper_abstains_when_unconfigured_or_missing(tmp_path, escalate_module):
    rt, session, db, svc, weak, strong = _escalate_runtime(tmp_path, escalate_module)
    base = _SchemaRegistry(db, rt.config, rt.services)

    rt.config["escalate_strong_llm"] = ""
    assert rt.hooks.shape_scope(session, base) is base

    rt.config["escalate_strong_llm"] = "not_a_service"
    assert rt.hooks.shape_scope(session, base) is base


def test_unresolvable_strong_name_warns_once(tmp_path, escalate_module, caplog):
    rt, session, db, svc, weak, strong = _escalate_runtime(tmp_path, escalate_module)
    rt.config["escalate_strong_llm"] = "typo_profile"

    with caplog.at_level("WARNING", logger="Escalate"):
        assert svc._strong_service() is None
        assert svc._strong_service() is None  # same bad name: no second warning
    warnings = [r for r in caplog.records if "typo_profile" in r.getMessage()]
    assert len(warnings) == 1

    # A different bad name re-warns; a good name clears the latch.
    caplog.clear()
    rt.config["escalate_strong_llm"] = "other_typo"
    with caplog.at_level("WARNING", logger="Escalate"):
        assert svc._strong_service() is None
    assert any("other_typo" in r.getMessage() for r in caplog.records)
    rt.config["escalate_strong_llm"] = "strong"
    assert svc._strong_service() is rt.services.get("strong")


def test_tool_sets_pending_and_restart(tmp_path, escalate_module):
    rt, session, db, svc, weak, strong = _escalate_runtime(tmp_path, escalate_module)
    tool = escalate_module.EscalateTool(svc)
    ctx = SimpleNamespace(runtime=rt, session_key="s")

    result = tool.run(ctx, reason="too hard for me")

    assert result.success
    assert "strong" in result.llm_summary
    assert session.restart_turn is True
    assert escalate_module.escalation_pending(session)


def test_tool_sets_handoff_note_cleared_after_strong_turn(tmp_path, escalate_module):
    rt, session, db, svc, weak, strong = _escalate_runtime(tmp_path, escalate_module)
    key = escalate_module.PROMPT_KEY
    tool = escalate_module.EscalateTool(svc)

    tool.run(SimpleNamespace(runtime=rt, session_key="s"), reason="hard proof")
    note = session.system_prompt_extras.get(key)
    assert note and "Escalated turn" in note and "hard proof" in note

    finish_ctx = HookContext(session=session, runtime=rt, moment="turn_finish")
    # The weak drive's turn_finish must NOT clear it (strong turn hasn't run).
    svc._turn_finish(finish_ctx, None)
    assert key in session.system_prompt_extras

    # Once the escort serves a strong call, turn_finish clears it.
    call_ctx = HookContext(session=session, runtime=rt, moment="model_call")
    request = ModelRequest(llm=weak, messages=[], tools=None)
    svc._model_call(call_ctx, request, lambda req: SimpleNamespace(content="ok"))
    assert request.llm is strong
    svc._turn_finish(finish_ctx, None)
    assert key not in session.system_prompt_extras


def test_tool_fails_cleanly_when_strong_missing(tmp_path, escalate_module):
    rt, session, db, svc, weak, strong = _escalate_runtime(tmp_path, escalate_module)
    rt.config["escalate_strong_llm"] = "gone"
    tool = escalate_module.EscalateTool(svc)

    result = tool.run(SimpleNamespace(runtime=rt, session_key="s"))

    assert not result.success
    assert not session.restart_turn


def test_full_escalation_round_trip(tmp_path, escalate_module):
    """User message → weak model calls escalate → strong model retakes and
    finishes the turn → pending cleared → next turn resolves weak again."""
    rt, session, db, svc, weak, strong = _escalate_runtime(
        tmp_path, escalate_module,
        weak_responses=[
            _response(tool_calls=[{"id": "c1", "name": "escalate",
                                   "arguments": '{"reason": "beyond me"}'}]),
            _response(content="weak handles the follow-up"),
        ],
        strong_responses=[_response(content="Strong answer.")],
    )
    registry = _SchemaRegistry(db, rt.config, rt.services)
    registry.runtime = rt  # the shaped ToolRegistry clone inherits this
    rt.tool_registry = registry

    out = rt.handle_action("s", "send_text", "prove something hard")

    assert "Strong answer." in out.messages
    assert session.history[-1] == {"role": "assistant", "content": "Strong answer."}
    # Weak attempt is preserved: an escalate tool result precedes the answer.
    tool_rows = [m for m in session.history if m["role"] == "tool"]
    assert any("Escalating this turn to strong" in (m.get("content") or "") for m in tool_rows)
    # Pending was cleared by the finalizer: next turn resolves weak again.
    assert not escalate_module.escalation_pending(session)
    assert _cfg.active_llm(rt, session) is weak
    # Ledger shows the model flip inside one exchange.
    models = [json.loads(r[0])["llm"] for r in db.conn.execute(
        "SELECT data_json FROM action_ledger WHERE origin='agent_enact' ORDER BY id").fetchall()]
    assert "weak" in models and "strong" in models

    out2 = rt.handle_action("s", "send_text", "and something easy")
    assert "weak handles the follow-up" in out2.messages


def test_unload_removes_all_hooks(tmp_path, escalate_module):
    rt, session, db, svc, weak, strong = _escalate_runtime(tmp_path, escalate_module)
    base = _SchemaRegistry(db, rt.config, rt.services)
    assert "escalate" in rt.hooks.shape_scope(session, base).tools

    svc.unload()

    assert rt.hooks.shape_scope(session, base) is base
    # With the escort gone, a pending escalation no longer retargets calls.
    session.plugin_state.setdefault("escalate", {})["pending"] = True
    request = ModelRequest(llm=weak, messages=[], tools=None)
    handler = rt.hooks.wrap_model_call(session, rt, lambda req: req.llm)
    assert handler(request) is weak
