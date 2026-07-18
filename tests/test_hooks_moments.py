"""Contract tests for the unified hook system (``runtime/hooks.py``).

The six moments — turn_start, shape_scope, vet_permission, model_call,
end_turn, turn_finish — share one contract: every hook receives
``(ctx, payload)``, returns None to abstain, and can never break a turn by
raising. Escorts (``model_call``) additionally receive ``proceed`` and own
the round trip; doormen (``end_turn``) return verdicts the loop obeys under
a hard fire budget. These tests pin that contract with a real
``ConversationState`` + ``ConversationLoop`` and fake LLMs (no network).
"""

from types import SimpleNamespace

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from plugins.BaseTool import ToolResult
from state_machine.conversation import CallableSpec, ConversationState, Participant
from state_machine.conversation_phases import BASE_PHASE
from runtime.conversation_loop import ConversationLoop
from runtime.hooks import (
    END_TURN,
    MODEL_CALL,
    SHAPE_SCOPE,
    VET_PERMISSION,
    Allow,
    HookRegistry,
    ModelRequest,
    PermissionVerdict,
    Redrive,
    RequireTool,
    SendBack,
    TurnEnding,
)
from runtime.session import RuntimeSession


def _response(content="", tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        is_error=False,
        prompt_tokens=0,
    )


class _FakeLLM:
    """Returns queued responses; records every call with its kwargs."""

    context_size = 0  # disables proactive compaction
    model_name = "fake"

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def chat_with_tools(self, messages, tools, attachments=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, "kwargs": kwargs})
        return self._responses.pop(0) if self._responses else _response(content="done.")


class _ToolChoiceLLM(_FakeLLM):
    supports_tool_choice = True


class _FakeRegistry:
    def __init__(self, schemas, max_tool_calls=5):
        self._schemas = schemas
        self.max_tool_calls = max_tool_calls
        self.tools = {}  # empty -> no per-tool budget enforcement

    def get_all_schemas(self):
        return self._schemas


def _rig(tools=None, schemas=None, llm=None, max_tool_calls=5):
    """Build a loop wired to a minimal runtime with a live HookRegistry."""
    cs = ConversationState(
        [Participant("user", "user"), Participant("agent", "agent", tools=tools or {})],
        "agent",
        BASE_PHASE,
        {"session_key": "s", "agent_scoped_tool_names": list((tools or {}).keys())},
    )
    session = RuntimeSession("s", cs)
    hooks = HookRegistry()
    runtime = SimpleNamespace(
        sessions={"s": session}, hooks=hooks, services={},
        push_message=lambda *a, **k: None,
    )
    llm = llm or _FakeLLM()
    loop = ConversationLoop(
        llm, _FakeRegistry(schemas or [], max_tool_calls), {}, "You are a helpful agent.",
        runtime=runtime, session_key="s",
    )
    return loop, cs, session, hooks, llm, runtime


def _echo_tools(record):
    def handler(cs, actor, args):
        record.append(args)
        return ToolResult(llm_summary="echoed", data={"ok": True})
    spec = {"type": "function", "function": {"name": "echo", "parameters": {}}}
    return {"echo": CallableSpec("echo", handler=handler)}, [spec]


# ──────────────────────────────────────────────────────────────────────
# model_call — the escort doorway
# ──────────────────────────────────────────────────────────────────────

def test_escort_swaps_the_brain_per_call():
    loop, cs, session, hooks, weak, _ = _rig()
    strong = _FakeLLM([_response(content="from the strong brain")])

    def escort(ctx, request, proceed):
        request.llm = strong
        return proceed(request)

    hooks.add(MODEL_CALL, escort)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "from the strong brain"
    assert weak.calls == []  # the default brain never took the call
    assert strong.calls, "the swapped brain took the call"


def test_escort_can_inspect_and_retry():
    llm = _FakeLLM([_response(content="weak answer"), _response(content="better answer")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)

    def escort(ctx, request, proceed):
        response = proceed(request)
        if "weak" in (response.content or ""):
            return proceed(ModelRequest(
                llm=request.llm,
                messages=[*request.messages, {"role": "user", "content": "try harder"}],
                tools=request.tools,
            ))
        return response

    hooks.add(MODEL_CALL, escort)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "better answer"
    assert len(llm.calls) == 2
    assert llm.calls[1]["messages"][-1]["content"] == "try harder"


def test_raising_escort_before_dialing_is_transparent():
    llm = _FakeLLM([_response(content="fine.")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)

    def escort(ctx, request, proceed):
        raise RuntimeError("escort exploded before dialing")

    hooks.add(MODEL_CALL, escort)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "fine."
    assert len(llm.calls) == 1  # the call still happened, exactly once


def test_raising_escort_after_dialing_keeps_the_response():
    llm = _FakeLLM([_response(content="fine.")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)

    def escort(ctx, request, proceed):
        proceed(request)
        raise RuntimeError("escort exploded after dialing")

    hooks.add(MODEL_CALL, escort)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "fine."
    assert len(llm.calls) == 1  # the fetched response was used, never re-fetched


def test_escort_abstaining_with_none_uses_its_fetched_response():
    llm = _FakeLLM([_response(content="fine.")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)

    def escort(ctx, request, proceed):
        proceed(request)
        return None  # abstain after dialing

    hooks.add(MODEL_CALL, escort)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "fine."
    assert len(llm.calls) == 1


# ──────────────────────────────────────────────────────────────────────
# end_turn — the doorman doorway
# ──────────────────────────────────────────────────────────────────────

def test_doorman_sendback_re_asks_once_and_records_the_note():
    llm = _FakeLLM([_response(content="first answer"), _response(content="second answer")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)

    def doorman(ctx, ending):
        if ending.doorman_fires == 0:
            return SendBack("Please also include a haiku.")
        return None

    hooks.add(END_TURN, doorman)
    history = [{"role": "user", "content": "hi"}]
    reply, new_messages, _ = loop.drive(cs, "agent", history)

    assert reply == "second answer"
    assert len(llm.calls) == 2
    assert {"role": "user", "content": "Please also include a haiku."} in new_messages
    assert cs.turn_priority == "user"  # the turn still ended


def test_doorman_ephemeral_note_reaches_model_but_not_history():
    llm = _FakeLLM([_response(content="first"), _response(content="second")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)

    def doorman(ctx, ending):
        return SendBack("secret nudge", ephemeral=True) if ending.doorman_fires == 0 else None

    hooks.add(END_TURN, doorman)
    history = [{"role": "user", "content": "hi"}]
    reply, new_messages, _ = loop.drive(cs, "agent", history)

    assert reply == "second"
    assert llm.calls[1]["messages"][-1]["content"] == "secret nudge"
    assert all(m.get("content") != "secret nudge" for m in history)


def test_doorman_fire_budget_frees_a_trapped_agent():
    llm = _FakeLLM([_response(content=f"answer {i}") for i in range(10)])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)

    hooks.add(END_TURN, lambda ctx, ending: SendBack("not good enough"))
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    # LIMIT send-backs → LIMIT+1 model calls, then the doorman is ignored.
    assert len(llm.calls) == ConversationLoop.DOORMAN_FIRE_LIMIT + 1
    assert cs.turn_priority == "user"


def test_doorman_require_tool_forces_the_call_when_supported():
    ran = []
    tools, schemas = _echo_tools(ran)
    llm = _ToolChoiceLLM([
        _response(content="I'm done."),
        _response(tool_calls=[{"id": "tc1", "name": "echo", "arguments": '{"via": "forced"}'}]),
        _response(content="Echo sent."),
    ])
    loop, cs, session, hooks, _, _ = _rig(tools=tools, schemas=schemas, llm=llm)

    def doorman(ctx, ending):
        return RequireTool("echo") if not ran else None

    hooks.add(END_TURN, doorman)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert ran == [{"via": "forced"}]
    forced_call = llm.calls[1]
    assert forced_call["kwargs"]["tool_choice"] == {"type": "function", "function": {"name": "echo"}}
    assert [s["function"]["name"] for s in forced_call["tools"]] == ["echo"]
    assert reply == "Echo sent."


def test_doorman_require_tool_degrades_to_a_note_without_backend_support():
    ran = []
    tools, schemas = _echo_tools(ran)
    llm = _FakeLLM([  # no supports_tool_choice
        _response(content="I'm done."),
        _response(tool_calls=[{"id": "tc1", "name": "echo", "arguments": "{}"}]),
        _response(content="Echo sent."),
    ])
    loop, cs, session, hooks, _, _ = _rig(tools=tools, schemas=schemas, llm=llm)
    hooks.add(END_TURN, lambda ctx, ending: RequireTool("echo") if not ran else None)

    history = [{"role": "user", "content": "hi"}]
    reply, _, _ = loop.drive(cs, "agent", history)

    followup = llm.calls[1]
    assert "tool_choice" not in followup["kwargs"]  # never forwarded unsupported
    assert "echo" in followup["messages"][-1]["content"]  # prompt-level instruction
    assert all("Before finishing" not in (m.get("content") or "") for m in history)
    assert ran, "the tool still ran via the instruction"
    assert reply == "Echo sent."


def test_doorman_redrive_exits_without_end_turn():
    llm = _FakeLLM([_response(content="half done")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)
    hooks.add(END_TURN, lambda ctx, ending: Redrive())

    loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert session.restart_turn is True
    assert cs.turn_priority == "agent"  # no end_turn: the re-drive finishes the turn


def test_budget_exhaustion_doorman_can_replace_the_wrapup_note():
    ran = []
    tools, schemas = _echo_tools(ran)
    tool_call = [{"id": "tc", "name": "echo", "arguments": "{}"}]
    # max_tool_calls=1 → max_iterations=8: eight tool-call rounds exhaust the
    # loop, then the budget-exhaustion doorman runs the wrap-up call.
    llm = _FakeLLM([*(_response(tool_calls=list(tool_call)) for _ in range(8)),
                    _response(content="brief.")])
    loop, cs, session, hooks, _, _ = _rig(tools=tools, schemas=schemas, llm=llm, max_tool_calls=1)

    def doorman(ctx, ending):
        return SendBack("Wrap up in one word.") if ending.reason == "budget_exhausted" else None

    hooks.add(END_TURN, doorman)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "brief."
    summary_call = llm.calls[-1]
    assert summary_call["messages"][-1]["content"] == "Wrap up in one word."
    assert summary_call["tools"] is None  # wrap-up is text-only
    assert cs.turn_priority == "user"


# ──────────────────────────────────────────────────────────────────────
# pending_agent_actions — the queued-action drain
# ──────────────────────────────────────────────────────────────────────

def test_queued_agent_action_runs_before_the_model_is_consulted(monkeypatch):
    ran = []
    tools, schemas = _echo_tools(ran)
    llm = _FakeLLM([_response(content="all set.")])
    loop, cs, session, hooks, _, _ = _rig(tools=tools, schemas=schemas, llm=llm)
    session.pending_agent_actions.append(
        {"name": "echo", "args": {"who": "queued"}, "forced_by": "test_hook"})

    ledgered = []
    monkeypatch.setattr("runtime.conversation_loop.record_enact",
                        lambda *a, **k: ledgered.append(k))

    reply, new_messages, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert ran == [{"who": "queued"}]
    assert session.pending_agent_actions == []
    tool_row = next(m for m in new_messages if m.get("role") == "tool")
    assert tool_row["tool_call_id"].startswith("tc_hook_")
    stamp = next(k for k in ledgered if k.get("action_type") == "call_tool")
    assert stamp["data"]["hook"] == "test_hook"
    assert reply == "all set."


# ──────────────────────────────────────────────────────────────────────
# Registry contract (registration, abstain, removal, adjusters)
# ──────────────────────────────────────────────────────────────────────

def test_every_doorway_threads_a_real_runtime_into_ctx(tmp_path):
    """The uniform contract promises ctx.runtime at EVERY doorway (the
    template's vet_permission example dereferences ctx.runtime.db). A driven
    turn must reach all six moments with ctx.runtime set — never None."""
    from pipeline.database import Database
    from runtime.conversation_runtime import ConversationRuntime
    from runtime.hooks import (
        END_TURN, MODEL_CALL, SHAPE_SCOPE, TURN_FINISH, TURN_START,
        VET_PERMISSION, SendBack,
    )

    class _ServiceLLM(_FakeLLM):
        loaded = True
        is_llm_backend = True

    db = Database(str(tmp_path / "ctx.db"))
    cid = db.create_conversation("x")
    llm = _ServiceLLM([_response(content="one"), _response(content="two")])
    rt = ConversationRuntime(db=db, services={"llm": llm}, config={},
                             tool_registry=_FakeRegistry([]))
    rt.load_conversation("s", cid)

    seen = {}

    def record(moment):
        def hook(ctx, *_):
            seen[moment] = ctx.runtime
            # end_turn: send back exactly once so the turn still ends.
            if moment == END_TURN:
                return SendBack("more") if _[0].doorman_fires == 0 else None
            return None
        return hook

    rt.hooks.add(TURN_START, record(TURN_START))
    rt.hooks.add(SHAPE_SCOPE, lambda ctx, reg: seen.__setitem__(SHAPE_SCOPE, ctx.runtime) or reg)
    rt.hooks.add(VET_PERMISSION, record(VET_PERMISSION))
    rt.hooks.add(MODEL_CALL, lambda ctx, req, proceed: (seen.__setitem__(MODEL_CALL, ctx.runtime), proceed(req))[1])
    rt.hooks.add(END_TURN, record(END_TURN))
    rt.hooks.add(TURN_FINISH, record(TURN_FINISH))

    rt.handle_action("s", "send_text", "hello")

    # vet_permission only fires when a tool asks to run; the other five are
    # reached by any driven turn. Assert none of the reached ones saw None.
    for moment in (TURN_START, SHAPE_SCOPE, MODEL_CALL, END_TURN, TURN_FINISH):
        assert moment in seen, f"{moment} doorway was never reached"
        assert seen[moment] is rt, f"{moment} handed ctx.runtime={seen[moment]!r}, expected the runtime"


def test_add_rejects_unknown_moment():
    hooks = HookRegistry()
    try:
        hooks.add("bogus_moment", lambda ctx, payload: None)
    except ValueError as e:
        assert "bogus_moment" in str(e)
    else:
        raise AssertionError("unknown moment must be rejected loudly")


def test_remove_unregisters_a_new_api_hook():
    llm = _FakeLLM([_response(content="one"), _response(content="two")])
    loop, cs, session, hooks, _, _ = _rig(llm=llm)
    doorman = lambda ctx, ending: SendBack("again") if ending.doorman_fires == 0 else None  # noqa: E731

    hooks.add(END_TURN, doorman)
    hooks.remove(doorman)
    reply, _, _ = loop.drive(cs, "agent", [{"role": "user", "content": "hi"}])

    assert reply == "one"
    assert len(llm.calls) == 1  # the removed doorman never fired


def test_shape_scope_folds_and_skips_raisers():
    hooks = HookRegistry()
    session = SimpleNamespace(key="s")

    def boom(ctx, registry):
        raise RuntimeError("shaper down")

    hooks.add(SHAPE_SCOPE, boom)
    hooks.add(SHAPE_SCOPE, lambda ctx, registry: registry + ["extra"])

    assert hooks.shape_scope(session, ["base"]) == ["base", "extra"]


def test_vet_permission_new_api_receives_the_query():
    hooks = HookRegistry()
    seen = {}

    def gate(ctx, query):
        seen["tool"], seen["command"] = query.tool_name, query.command
        return PermissionVerdict(False, "not on my watch")

    hooks.add(VET_PERMISSION, gate)
    verdict = hooks.vet_permission(SimpleNamespace(key="s"), "shell", "rm -rf /")

    assert seen == {"tool": "shell", "command": "rm -rf /"}
    assert verdict.allow is False
    assert verdict.reason == "not on my watch"


def test_turn_finish_receives_the_outcome():
    from runtime.hooks import TURN_FINISH, TurnOutcome
    hooks = HookRegistry()
    seen = []
    hooks.add(TURN_FINISH, lambda ctx, outcome: seen.append((ctx.moment, outcome)))

    hooks.finish_turn(SimpleNamespace(key="s"), TurnOutcome(ok=True, final_text="bye"))

    assert seen[0][0] == TURN_FINISH
    assert seen[0][1].final_text == "bye"


def test_turn_finish_fires_once_per_logical_turn_across_redrive(tmp_path):
    """A Redrive splits one logical turn into two drives; the turn_finish
    observers fire once, from the drive that actually ends the turn."""
    from pipeline.database import Database
    from runtime.conversation_runtime import ConversationRuntime
    from runtime.hooks import TURN_FINISH

    class _ServiceLLM(_FakeLLM):
        loaded = True
        is_llm_backend = True

    db = Database(str(tmp_path / "moments.db"))
    cid = db.create_conversation("x")
    llm = _ServiceLLM([_response(content="half"), _response(content="whole")])
    rt = ConversationRuntime(db=db, services={"llm": llm}, config={})
    rt.load_conversation("s", cid)

    fired = []
    redriven = {"done": False}

    def doorman(ctx, ending):
        if not redriven["done"]:
            redriven["done"] = True
            return Redrive()
        return None

    rt.hooks.add(END_TURN, doorman)
    rt.hooks.add(TURN_FINISH, lambda ctx, outcome: fired.append(outcome))
    out = rt.handle_action("s", "send_text", "hello")

    assert out.ok
    assert len(llm.calls) == 2   # two drives, one model call each
    assert len(fired) == 1       # one logical turn → one finish
    assert fired[0].ok is True
    assert fired[0].final_text == "whole"
