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
        # The native backend contract, in full: ``as_brain`` wraps anything
        # exposing this in a ``NativeBrain``, which calls it with tools and an
        # attachment bundle. A double taking ``messages`` alone passed only
        # because the handler used to call it directly — which meant the
        # handler spoke a language no real brain answered to.
        def chat_with_tools(self, messages, tools=None, attachments=None):
            seen.append(messages)
            return SimpleNamespace(
                is_error=False,
                content="summary",
                tool_calls=[],
                error=None,
                error_code=None,
                prompt_tokens=None,
                cached_prompt_tokens=None,
            )

    brain = Brain()
    # ``monkeypatch.setattr`` on a dotted string imports the module first, and
    # ``runtime.runtime_config`` cannot *be* imported first: it and
    # ``runtime.persistence`` import each other, so whichever is asked for
    # initially fails on a partially initialized partner. ``runtime.bootstrap``
    # is an entry point that resolves the cycle. This test passed only when
    # some earlier test in the session had already pulled runtime in — a pass
    # by luck, and it went green or red depending on which files pytest
    # collected alongside it.
    # ``from … import`` and not ``import runtime.bootstrap``: the latter binds
    # the name ``runtime`` in this scope, shadowing the fake runtime above.
    from runtime import bootstrap  # noqa: F401

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


# ── naming a model, rather than holding one ───────────────────────────
#
# ``ModelRequest.llm`` is a name the kernel resolves, and ``agent.complete``
# works the same way for the same reason: a box cannot hold a live model, so a
# background chore that wants a cheap one has to be able to say which.

def test_agent_complete_resolves_a_named_profile(monkeypatch):
    """An explicit profile wins over whatever the session drives with."""
    from sandbox.guest.llm import LLMResponse

    placed = []

    class Cheap:
        name = "cheap"

        def chat(self, request, on_delta=None):
            placed.append(request)
            return LLMResponse(content="six words or fewer")

    monkeypatch.setattr("llm.registry.usable_brain",
                        lambda name: Cheap() if name == "cheap" else None)

    result = HANDLERS[AGENT_COMPLETE](
        SimpleNamespace(config={}),
        {"profile": "cheap", "messages": [{"role": "user", "content": "hi"}]})

    assert result.ok
    assert result.data["content"] == "six words or fewer"
    assert result.data["llm"] == "cheap"
    assert placed[0].messages == [{"role": "user", "content": "hi"}]


def test_a_named_profile_that_does_not_exist_says_so(monkeypatch):
    """Falling back to the default would title conversations with the
    expensive model and never mention it."""
    monkeypatch.setattr("llm.registry.usable_brain", lambda name: None)

    result = HANDLERS[AGENT_COMPLETE](
        SimpleNamespace(config={}), {"profile": "gone", "prompt": "hi"})

    assert not result.ok
    assert "gone" in result.error


def test_no_profile_and_no_session_uses_the_default_brain(monkeypatch):
    """The fallback used to be ``services["llm"]`` — a service that stopped
    existing when the LLM moved kernel-side, so this path was simply dead."""
    from sandbox.guest.llm import LLMResponse

    class Default:
        name = "default"

        def chat(self, request, on_delta=None):
            return LLMResponse(content="from the default")

    monkeypatch.setattr("llm.default_brain", lambda config: Default())

    result = HANDLERS[AGENT_COMPLETE](SimpleNamespace(config={}),
                                      {"prompt": "hi"})

    assert result.ok
    assert result.data["content"] == "from the default"


def test_a_prompt_becomes_one_user_message(monkeypatch):
    """``prompt`` is the convenience shape; ``messages`` is the real one."""
    from sandbox.guest.llm import LLMResponse

    seen = []

    class Brain:
        name = "b"

        def chat(self, request, on_delta=None):
            seen.append(request.messages)
            return LLMResponse(content="ok")

    monkeypatch.setattr("llm.default_brain", lambda config: Brain())
    HANDLERS[AGENT_COMPLETE](SimpleNamespace(config={}), {"prompt": "hello"})

    assert seen == [[{"role": "user", "content": "hello"}]]
