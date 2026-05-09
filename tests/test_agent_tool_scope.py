from agent.tool_registry import ToolRegistry
from plugins.BaseTool import BaseTool, ToolResult
from runtime.agent_scope import load_scope, scoped_registry
from state_machine.conversation import CallableSpec, ConversationState, Participant
from state_machine.conversation_phases import PHASE_AWAITING_INPUT


class _Lexical(BaseTool):
    name = "lexical_search"
    description = "Hidden keyword helper."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}}
    max_calls = 9

    def run(self, context, **kwargs):
        return ToolResult(data={"tool": self.name, **kwargs})


class _Semantic(_Lexical):
    name = "semantic_search"
    description = "Hidden semantic helper."


class _Hybrid(_Lexical):
    name = "hybrid_search"
    description = "Visible composite search."
    max_calls = 2

    def run(self, context, **kwargs):
        return ToolResult(data={
            "lex": context.call_tool("lexical_search", **kwargs).data,
            "sem": context.call_tool("semantic_search", **kwargs).data,
        })


def _registry():
    registry = ToolRegistry(None, {"tool_timeout": 10})
    for tool in (_Hybrid(), _Lexical(), _Semantic()):
        registry.register(tool)
    return registry


def _config():
    return {"active_agent_profile": "default", "agent_profiles": {"default": {
        "whitelist_or_blacklist_tools": "blacklist",
        "tools_list": ["lexical_search", "semantic_search"],
    }}}


def test_blacklisted_dependencies_stay_callable_but_schema_hidden():
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
