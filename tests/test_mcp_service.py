"""Tests for the MCP client service.

These exercise the synchronous surface — tool wrapping, result mapping,
registration, and unload — without a live event loop or the ``mcp`` package,
mirroring the fake-dependency style of test_litellm_service.py.
"""
from types import SimpleNamespace

import pytest

from plugins.services.service_mcp import (
    MCPService,
    MCPTool,
    build_services,
    qualified_tool_name,
    result_to_text,
)
from plugins.BaseTool import ToolResult


# ── fakes ────────────────────────────────────────────────────────────

class _FakeRegistry:
    """Minimal stand-in for ToolRegistry."""

    def __init__(self):
        self.tools = {}

    def register(self, tool):
        self.tools[tool.name] = tool

    def unregister(self, name):
        self.tools.pop(name, None)


def _mcp_tool(name="search", description="Search the web", schema=None):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema or {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )


def _call_result(text=None, blocks=None, is_error=False, structured=None):
    content = blocks
    if content is None:
        content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(content=content, isError=is_error, structuredContent=structured)


# ── name + result helpers ────────────────────────────────────────────

def test_qualified_tool_name_namespaces_and_sanitizes():
    assert qualified_tool_name("github", "create_issue") == "mcp__github__create_issue"
    # Illegal characters collapse to underscores.
    assert qualified_tool_name("my server", "weird/name") == "mcp__my_server__weird_name"


def test_qualified_tool_name_capped_at_64():
    name = qualified_tool_name("server", "x" * 100)
    assert len(name) <= 64


def test_result_to_text_joins_text_and_labels_other_blocks():
    result = _call_result(blocks=[
        SimpleNamespace(type="text", text="line one"),
        SimpleNamespace(type="image"),
        SimpleNamespace(type="text", text="line two"),
    ])
    text = result_to_text(result)
    assert "line one" in text
    assert "line two" in text
    assert "[image content returned by MCP tool]" in text


# ── MCPTool wrapping ─────────────────────────────────────────────────

def test_mcptool_builds_schema_from_mcp_definition():
    tool = MCPTool(SimpleNamespace(), "github", _mcp_tool())
    assert tool.name == "mcp__github__search"
    assert "github" in tool.description
    assert tool.requires_services == ["mcp"]
    schema = tool.to_schema()
    assert schema["function"]["name"] == "mcp__github__search"
    assert schema["function"]["parameters"]["required"] == ["q"]


def test_mcptool_defaults_empty_schema_when_missing():
    bad = SimpleNamespace(name="t", description="", inputSchema=None)
    tool = MCPTool(SimpleNamespace(), "srv", bad)
    assert tool.parameters == {"type": "object", "properties": {}}


def test_mcptool_run_success_maps_text_to_summary():
    service = SimpleNamespace(
        call_mcp_tool=lambda server, name, args: _call_result(text="hello world"),
    )
    tool = MCPTool(service, "srv", _mcp_tool())
    result = tool.run(None, q="x")
    assert isinstance(result, ToolResult)
    assert result.success
    assert result.llm_summary == "hello world"


def test_mcptool_run_passes_args_through():
    seen = {}

    def call(server, name, args):
        seen.update({"server": server, "name": name, "args": args})
        return _call_result(text="ok")

    tool = MCPTool(SimpleNamespace(call_mcp_tool=call), "srv", _mcp_tool())
    tool.run(None, q="hello", limit=5)
    assert seen == {"server": "srv", "name": "search", "args": {"q": "hello", "limit": 5}}


def test_mcptool_run_error_result_becomes_failure():
    service = SimpleNamespace(
        call_mcp_tool=lambda server, name, args: _call_result(text="boom", is_error=True),
    )
    tool = MCPTool(service, "srv", _mcp_tool())
    result = tool.run(None, q="x")
    assert not result.success
    assert "boom" in result.error


def test_mcptool_run_surfaces_structured_content():
    service = SimpleNamespace(
        call_mcp_tool=lambda server, name, args: _call_result(text="t", structured={"answer": 42}),
    )
    tool = MCPTool(service, "srv", _mcp_tool())
    result = tool.run(None, q="x")
    assert result.data == {"answer": 42}


def test_mcptool_run_exception_becomes_failure():
    def boom(server, name, args):
        raise RuntimeError("connection dropped")

    tool = MCPTool(SimpleNamespace(call_mcp_tool=boom), "srv", _mcp_tool())
    result = tool.run(None, q="x")
    assert not result.success
    assert "connection dropped" in result.error


# ── registration + lifecycle ─────────────────────────────────────────

def test_register_tools_is_idempotent_and_populates_registry():
    service = MCPService({})
    registry = _FakeRegistry()
    service._tool_registry = registry
    service._server_tools = {"srv": [_mcp_tool("a"), _mcp_tool("b")]}

    service._register_tools()
    assert set(registry.tools) == {"mcp__srv__a", "mcp__srv__b"}

    # Second call must not double-register or error.
    service._register_tools()
    assert set(registry.tools) == {"mcp__srv__a", "mcp__srv__b"}
    assert service._registered == {"mcp__srv__a", "mcp__srv__b"}


def test_register_tools_noop_without_registry():
    service = MCPService({})
    service._server_tools = {"srv": [_mcp_tool("a")]}
    service._register_tools()  # no registry bound — must not raise
    assert service._registered == set()


def test_bind_runtime_registers_when_loaded_first():
    """Autoload order: _load() connects before the registry is bound."""
    service = MCPService({})
    service.loaded = True
    service._server_tools = {"srv": [_mcp_tool("a")]}
    registry = _FakeRegistry()

    service.bind_runtime(tool_registry=registry)
    assert "mcp__srv__a" in registry.tools


def test_unload_unregisters_tools_without_a_loop():
    service = MCPService({})
    registry = _FakeRegistry()
    service._tool_registry = registry
    service._server_tools = {"srv": [_mcp_tool("a")]}
    service._register_tools()
    service.loaded = True
    assert registry.tools

    service.unload()
    assert registry.tools == {}
    assert service._registered == set()
    assert service.loaded is False
    assert service._tools_registered is False


# ── config handling ──────────────────────────────────────────────────

def test_active_servers_filters_disabled_and_malformed():
    service = MCPService({"mcp_servers": {
        "good": {"command": "x"},
        "off": {"command": "y", "disabled": True},
        "bad": "not-a-dict",
    }})
    assert set(service._active_servers()) == {"good"}


def test_load_with_no_servers_is_idle_success():
    service = MCPService({"mcp_servers": {}})
    assert service._load() is True
    assert service.loaded is True
    assert service._loop is None  # never started a loop


def test_build_services_exposes_mcp_key():
    services = build_services({})
    assert "mcp" in services
    assert isinstance(services["mcp"], MCPService)


# ── OAuth classification ─────────────────────────────────────────────

def test_is_oauth_true_for_http_without_headers():
    svc = MCPService({})
    assert svc._is_oauth({"url": "https://x/mcp"}) is True
    assert svc._is_oauth({"transport": "http", "url": "https://x"}) is True


def test_is_oauth_false_for_stdio_headers_or_opt_out():
    svc = MCPService({})
    assert svc._is_oauth({"command": "npx"}) is False
    assert svc._is_oauth({"url": "https://x", "headers": {"Authorization": "Bearer t"}}) is False
    assert svc._is_oauth({"url": "https://x", "oauth": False}) is False
