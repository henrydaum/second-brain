"""Tests for agent tool scoping (``agent_scope`` + ``ToolRegistry``).

An agent profile whitelists/blacklists tools. Blacklisted tools stay callable
as dependencies of visible tools but are hidden from the LLM's schema list, and
the agent cannot invoke a hidden tool directly through the state machine.
"""

from agent.tool_registry import DEFAULT_TOOL_MAX_CALLS, ToolRegistry
from plugins.native.tool import BaseTool, ToolResult
from runtime.agent_scope import load_scope, registry_with_tools, scoped_registry
from state_machine.conversation import CallableSpec, ConversationState, Participant
from state_machine.conversation_phases import PHASE_AWAITING_INPUT


class _Lexical(BaseTool):
    """Lexical."""
    name = "lexical_search"
    description = "Hidden keyword helper."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}
    max_calls = 9

    def run(self, context, **kwargs):
        """Run lexical."""
        return ToolResult(data={"tool": self.name, **kwargs})


class _Semantic(_Lexical):
    """Semantic."""
    name = "semantic_search"
    description = "Hidden semantic helper."


class _Hybrid(_Lexical):
    """Hybrid."""
    name = "hybrid_search"
    description = "Visible composite search."
    max_calls = 2

    def run(self, context, **kwargs):
        """Run hybrid."""
        return ToolResult(data={
            "lex": context.call_tool("lexical_search", **kwargs).data,
            "sem": context.call_tool("semantic_search", **kwargs).data,
        })


class _Injected(_Lexical):
    """Injected."""
    name = "injected_tool"
    description = "Session-scoped injected tool."


def _registry():
    """Internal helper to handle registry."""
    registry = ToolRegistry(None, {})
    for tool in (_Hybrid(), _Lexical(), _Semantic()):
        registry.register(tool)
    return registry


def _config():
    """Internal helper to handle config."""
    return {"active_agent_profile": "default", "agent_profiles": {"default": {
        "whitelist_or_blacklist_tools": "blacklist",
        "tools_list": ["lexical_search", "semantic_search"],
    }}}


def test_blacklisted_dependencies_stay_callable_but_schema_hidden():
    """Verify blacklisted dependencies stay callable but schema hidden."""
    registry = scoped_registry(_registry(), load_scope("default", _config()))

    assert set(registry.tools) == {"hybrid_search", "lexical_search", "semantic_search"}
    assert [s["function"]["name"] for s in registry.get_all_schemas()] == ["hybrid_search"]
    assert registry.get_schema("lexical_search") is None
    assert registry.max_tool_calls == 2
    assert registry.call("hybrid_search", query="Buddhism").data == {
        "lex": {"tool": "lexical_search", "query": "Buddhism"},
        "sem": {"tool": "semantic_search", "query": "Buddhism"},
    }


def test_agent_cannot_directly_call_blacklisted_dependency():
    """Verify agent cannot directly call blacklisted dependency."""
    registry = scoped_registry(_registry(), load_scope("default", _config()))
    specs = {s["function"]["name"]: CallableSpec(s["function"]["name"]) for s in registry.get_all_schemas()}
    cs = ConversationState(
        [Participant("agent", "agent", tools=specs)],
        "agent",
        PHASE_AWAITING_INPUT,
        {"agent_scoped_tool_names": ["lexical_search", "semantic_search"]},
    )

    result = cs.enact("call_tool", {"name": "lexical_search", "args": {"query": "Buddhism"}}, "agent")

    assert not result.ok
    assert result.error.code == "unknown_tool"
    assert result.message == "Tool not in agent scope: 'lexical_search'."


def test_registry_with_tools_clones_and_exposes_injected_tools():
    """Verify registry_with_tools preserves registry wiring and exposes new tools."""
    registry = _registry()
    registry.orchestrator = object()
    registry.runtime = object()
    registry.visible_tool_names = {"hybrid_search"}

    cloned = registry_with_tools(registry, [_Injected()])

    assert cloned is not registry
    assert cloned.db is registry.db
    assert cloned.config is registry.config
    assert cloned.services is registry.services
    assert cloned.orchestrator is registry.orchestrator
    assert cloned.runtime is registry.runtime
    assert set(cloned.tools) == {"hybrid_search", "lexical_search", "semantic_search", "injected_tool"}
    assert cloned.visible_tool_names == {"hybrid_search", "injected_tool"}
    assert registry.visible_tool_names == {"hybrid_search"}


def test_registry_with_tools_replaces_existing_tool():
    """Verify injected tools replace by name."""
    registry = _registry()
    registry.visible_tool_names = {"hybrid_search"}
    replacement = _Injected()
    replacement.name = "hybrid_search"

    cloned = registry_with_tools(registry, [replacement], visible=False)

    assert cloned.tools["hybrid_search"] is replacement
    assert cloned.visible_tool_names == {"hybrid_search"}


def test_registry_with_tools_leaves_uncloneable_registries_unchanged():
    """Verify stub registries are left untouched."""
    registry = object()

    assert registry_with_tools(registry, [_Injected()]) is registry


def test_a_tool_cannot_declare_itself_unrunnable_in_the_background():
    """The registry no longer gates on ``background_safe``.

    Pinned as a negative because the failure would be silent in the dangerous
    direction: a store tool still carrying the declaration must not quietly
    keep an authority the kernel stopped honouring, and a reader finding the
    attribute in an old plugin should find this test rather than guess. What
    refuses unattended work now is policy — an unattended chain approves
    nothing unsafe, so ``sdk.ui.ask`` fails there on its own.
    """
    from types import SimpleNamespace

    class Interactive(_Lexical):
        """A tool that would once have been refused outright."""
        name = "interactive_tool"
        background_safe = False

    registry = _registry()
    registry.register(Interactive())
    registry.runtime = SimpleNamespace(is_attended=lambda key: False)

    result = registry.call("interactive_tool",
                           _session_key="spawn_subagent:7", query="x")

    assert result.success


class _Unbounded(_Lexical):
    """A tool that says nothing about how often it may be called."""
    name = "unbounded_tool"
    max_calls = None


def test_an_undeclared_tool_takes_the_configured_default():
    """``max_calls`` is unset on almost every tool, so the setting is the budget.

    Pinned because the default moved out of the base class and into
    ``default_tool_max_calls``: a regression that restored a class attribute
    would look entirely healthy — the tool still runs, it just stops early on
    a long turn, and the model reports a wall it was never meant to hit.
    """
    registry = ToolRegistry(None, {"default_tool_max_calls": 40})
    registry.register(_Unbounded())

    assert registry.call_limit(registry.tools["unbounded_tool"]) == 40
    assert registry.max_tool_calls == 40


def test_a_declared_limit_still_wins_over_the_setting():
    """The declaration is what a bounded tool has left, so it must beat config."""
    registry = ToolRegistry(None, {"default_tool_max_calls": 40})
    registry.register(_Hybrid())          # declares max_calls = 2

    assert registry.call_limit(registry.tools["hybrid_search"]) == 2


def test_a_registry_with_no_config_falls_back_rather_than_failing():
    """A registry built without settings still answers a number.

    The fallback matters because the budget is read on the hot path of every
    tool call: raising here would end the turn, and returning 0 would refuse
    every tool in a session whose config never loaded.
    """
    registry = ToolRegistry(None, {})
    registry.register(_Unbounded())

    assert registry.call_limit(registry.tools["unbounded_tool"]) == DEFAULT_TOOL_MAX_CALLS
    assert DEFAULT_TOOL_MAX_CALLS > 1
