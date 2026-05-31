"""Tests for the /mcp slash command."""
from types import SimpleNamespace

import pytest

from plugins.commands.command_mcp import (
    McpCommand,
    _build_spec,
    _parse_args,
    _parse_json_dict,
)


class _FakeMcp:
    """Stand-in for the loaded mcp service."""

    def __init__(self, tools=None, load_ok=True):
        self.loaded = True
        self.loads = 0
        self.unloads = 0
        self._tools = tools or {}      # server -> [qualified tool names]
        self._load_ok = load_ok

    def load(self):
        self.loads += 1
        self.loaded = bool(self._load_ok)
        return self._load_ok

    def unload(self):
        self.unloads += 1
        self.loaded = False

    def is_connected(self, server):
        return server in self._tools

    def registered_tools(self, server=None):
        if server is None:
            return sorted(t for v in self._tools.values() for t in v)
        return list(self._tools.get(server, []))


def _ctx(servers=None, mcp=None):
    services = {"mcp": mcp} if mcp is not None else {}
    return SimpleNamespace(config={"mcp_servers": servers or {}}, services=services)


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    """Never touch plugin_config.json during tests."""
    saved = {}
    monkeypatch.setattr("plugins.commands.command_mcp._save", lambda config: saved.update(config))
    return saved


# ── form ─────────────────────────────────────────────────────────────

def test_form_lists_servers_plus_add():
    ctx = _ctx({"github": {"command": "x"}})
    steps = McpCommand().form({}, ctx)
    assert steps[0].name == "server"
    assert steps[0].enum == ["github", "add"]


def test_form_add_stdio_branch_asks_for_command():
    ctx = _ctx()
    steps = McpCommand().form({"server": "add", "transport": "stdio"}, ctx)
    names = [s.name for s in steps]
    assert "command" in names and "cmd_args" in names and "env" in names
    assert "url" not in names


def test_form_add_http_branch_asks_for_url():
    ctx = _ctx()
    steps = McpCommand().form({"server": "add", "transport": "http"}, ctx)
    names = [s.name for s in steps]
    assert "url" in names and "headers" in names
    assert "command" not in names


def test_form_existing_server_offers_actions_with_disable_label():
    ctx = _ctx({"github": {"command": "x"}})
    steps = McpCommand().form({"server": "github"}, ctx)
    action = steps[-1]
    assert action.name == "action"
    assert action.enum == ["tools", "toggle", "reconnect", "remove"]
    # Enabled server -> the toggle reads "Disable".
    assert action.enum_labels[1] == "Disable"


def test_form_disabled_server_shows_enable_label():
    ctx = _ctx({"github": {"command": "x", "disabled": True}})
    steps = McpCommand().form({"server": "github"}, ctx)
    assert steps[-1].enum_labels[1] == "Enable"


# ── run: add ─────────────────────────────────────────────────────────

def test_run_add_stdio_persists_spec_and_reloads():
    mcp = _FakeMcp()
    ctx = _ctx(mcp=mcp)
    result = McpCommand().run(
        {"server": "add", "new_name": "github", "transport": "stdio",
         "command": "npx", "cmd_args": "-y @modelcontextprotocol/server-github", "env": ""},
        ctx,
    )
    assert ctx.config["mcp_servers"]["github"] == {
        "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
    }
    assert mcp.unloads == 1 and mcp.loads == 1   # reloaded
    assert result.startswith("Added MCP server 'github'.")


def test_run_add_http_sets_transport_and_url():
    mcp = _FakeMcp()
    ctx = _ctx(mcp=mcp)
    McpCommand().run(
        {"server": "add", "new_name": "remote", "transport": "http",
         "url": "https://example.com/mcp", "headers": '{"Authorization": "Bearer t"}'},
        ctx,
    )
    spec = ctx.config["mcp_servers"]["remote"]
    assert spec["transport"] == "http"
    assert spec["url"] == "https://example.com/mcp"
    assert spec["headers"] == {"Authorization": "Bearer t"}


def test_run_add_reports_tool_count_when_connected():
    mcp = _FakeMcp(tools={"github": ["mcp__github__a", "mcp__github__b"]})
    ctx = _ctx(mcp=mcp)
    result = McpCommand().run(
        {"server": "add", "new_name": "github", "transport": "stdio", "command": "npx"},
        ctx,
    )
    assert "2 tool(s)" in result


def test_run_add_duplicate_name_rejected():
    ctx = _ctx({"github": {"command": "x"}}, mcp=_FakeMcp())
    result = McpCommand().run(
        {"server": "add", "new_name": "github", "transport": "stdio", "command": "npx"}, ctx,
    )
    assert "already exists" in result
    # Spec untouched.
    assert ctx.config["mcp_servers"]["github"] == {"command": "x"}


def test_run_add_missing_command_errors():
    ctx = _ctx(mcp=_FakeMcp())
    result = McpCommand().run(
        {"server": "add", "new_name": "x", "transport": "stdio", "command": ""}, ctx,
    )
    assert "command is required" in result.lower()
    assert "x" not in ctx.config["mcp_servers"]


def test_run_add_surfaces_load_failure():
    ctx = _ctx(mcp=_FakeMcp(load_ok=False))
    result = McpCommand().run(
        {"server": "add", "new_name": "x", "transport": "stdio", "command": "npx"}, ctx,
    )
    assert "failed to load" in result.lower()
    # Spec was still saved.
    assert "x" in ctx.config["mcp_servers"]


# ── run: existing-server actions ─────────────────────────────────────

def test_run_remove_deletes_and_reloads():
    mcp = _FakeMcp()
    ctx = _ctx({"github": {"command": "x"}}, mcp=mcp)
    result = McpCommand().run({"server": "github", "action": "remove"}, ctx)
    assert "github" not in ctx.config["mcp_servers"]
    assert mcp.loads == 1
    assert "Removed" in result


def test_run_toggle_flips_disabled():
    ctx = _ctx({"github": {"command": "x"}}, mcp=_FakeMcp())
    McpCommand().run({"server": "github", "action": "toggle"}, ctx)
    assert ctx.config["mcp_servers"]["github"]["disabled"] is True
    McpCommand().run({"server": "github", "action": "toggle"}, ctx)
    assert ctx.config["mcp_servers"]["github"]["disabled"] is False


def test_run_tools_lists_registered():
    mcp = _FakeMcp(tools={"github": ["mcp__github__a", "mcp__github__b"]})
    ctx = _ctx({"github": {"command": "x"}}, mcp=mcp)
    result = McpCommand().run({"server": "github", "action": "tools"}, ctx)
    assert "mcp__github__a" in result and "mcp__github__b" in result


def test_run_no_server_lists_configured():
    ctx = _ctx({"github": {"command": "x"}}, mcp=_FakeMcp(tools={"github": ["mcp__github__a"]}))
    result = McpCommand().run({}, ctx)
    assert "MCP servers:" in result
    assert "github" in result


def test_run_unknown_server_errors():
    ctx = _ctx({"github": {"command": "x"}}, mcp=_FakeMcp())
    assert McpCommand().run({"server": "ghost", "action": "tools"}, ctx) == "Unknown server."


# ── spec parsing units ───────────────────────────────────────────────

def test_build_spec_stdio_with_env():
    spec, err = _build_spec({"transport": "stdio", "command": "python",
                             "cmd_args": "-m server", "env": '{"K": "v"}'})
    assert err is None
    assert spec == {"command": "python", "args": ["-m", "server"], "env": {"K": "v"}}


def test_build_spec_http_requires_url():
    spec, err = _build_spec({"transport": "http", "url": ""})
    assert spec is None and "URL is required" in err


def test_build_spec_rejects_bad_json_env():
    spec, err = _build_spec({"transport": "stdio", "command": "x", "env": "{not json}"})
    assert spec is None and "env" in err


def test_parse_args_accepts_json_array_and_whitespace():
    assert _parse_args("-y foo bar") == ["-y", "foo", "bar"]
    assert _parse_args('["-y", "foo"]') == ["-y", "foo"]
    assert _parse_args("") == []


def test_parse_json_dict_blank_is_empty():
    assert _parse_json_dict("", "env") == ({}, None)
