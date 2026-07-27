"""Sandboxed conversation compactor coverage."""

from types import SimpleNamespace

from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.guest.requests import AGENT_COMPLETE
from sandbox.handlers import HANDLERS


def test_agent_complete_selects_the_session_llm(monkeypatch):
    """A resident service still compacts with the session's active profile."""
    selected = object()
    fallback = object()
    session = object()
    runtime = SimpleNamespace(
        sessions={"chat": session},
        services={"llm": fallback},
    )
    ctx = SimpleNamespace(runtime=runtime, services=runtime.services)
    seen = []

    class Brain:
        def chat_with_tools(self, messages):
            seen.append(messages)
            return SimpleNamespace(
                is_error=False,
                content="summary",
                tool_calls=[],
                error=None,
            )

    brain = Brain()
    monkeypatch.setattr(
        "runtime.runtime_config.active_llm",
        lambda actual_runtime, actual_session: (
            brain
            if actual_runtime is runtime and actual_session is session
            else selected
        ),
    )

    result = HANDLERS[AGENT_COMPLETE](
        ctx,
        {
            "session_key": "chat",
            "messages": [{"role": "user", "content": "history"}],
        },
    )

    assert result.ok
    assert result.data["content"] == "summary"
    assert seen == [[{"role": "user", "content": "history"}]]


def test_compactor_runs_through_the_sandbox(monkeypatch):
    """The real service exports one serializable compaction call."""
    sandbox = Sandbox()
    configure(sandbox)
    module = adapt("plugins/services/service_compactor.py")
    service = module.build_services({})["compactor"]
    seen = []

    def complete(_ctx, args):
        seen.append(args)
        from sandbox.guest.requests import Result

        return Result(data={"content": "  compacted  ", "tool_calls": []})

    monkeypatch.setitem(HANDLERS, AGENT_COMPLETE, complete)
    try:
        assert service.load()
        assert service.compact(
            session_key="chat",
            transcript="USER: hello",
        ) == "compacted"
        assert seen[0]["session_key"] == "chat"
        assert seen[0]["messages"][1]["content"] == "USER: hello"
    finally:
        service.unload()
        configure(None)
        sandbox.shutdown()
