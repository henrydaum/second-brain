"""Structured tool SDK and sandboxed ``/tools`` coverage."""

from types import SimpleNamespace

from agent.tool_registry import ToolRegistry
from pipeline.database import Database
from plugins import plugin_discovery
from plugins.BaseTool import BaseTool, ToolResult
from sandbox import Sandbox
from sandbox.handlers.kernel import _tool_list


class DemoTool(BaseTool):
    name = "demo"
    description = "Return a useful demo value."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to echo.",
            },
            "limit": {
                "type": "integer",
                "default": 2,
            },
        },
        "required": ["query"],
    }
    requires_services = ["search"]
    config_settings = [
        (
            "Demo mode",
            "demo_mode_tools_test",
            "Controls the demo.",
            False,
            {"type": "boolean", "scope": "user"},
        ),
        (
            "Hidden token",
            "demo_token_tools_test",
            "Never expose this.",
            "",
            {"hidden": True},
        ),
    ]

    def run(self, context, query, limit=2):
        if query == "fail":
            return ToolResult.failed("demo broke")
        if query == "table":
            return ToolResult(data={
                "columns": ["Value"],
                "rows": [["one"], ["two"]],
            })
        return ToolResult(
            data={"query": query, "limit": limit},
            llm_summary=f"echoed {query}",
        )


def _context(tmp_path):
    db = Database(str(tmp_path / "tools.db"))
    config = {"skip_permissions": []}
    runtime = SimpleNamespace(
        config=config,
        sessions={
            "chat": SimpleNamespace(
                conversation_id=None, busy=False, cs=None, user_id=1)
        },
        refresh_session_specs=lambda: None,
        is_attended=lambda _key: True,
    )
    registry = ToolRegistry(db, config, {"search": SimpleNamespace(loaded=True)})
    registry.runtime = runtime
    registry.register(DemoTool())
    context = SimpleNamespace(
        db=db,
        config=dict(config),
        runtime=runtime,
        services=registry.services,
        tool_registry=registry,
        call_tool=registry.call,
        command_registry=object(),
        session_key="chat",
        user_id=1,
        current_tool_name=None,
    )
    return context, registry


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_tools.py",
            "ToolsCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_tool_list_details_are_structured_and_hide_settings(
        tmp_path, monkeypatch):
    context, _ = _context(tmp_path)
    plugin_discovery._collect_config_settings(
        DemoTool(), plugin_type="tool")
    try:
        result = _tool_list(context, {"details": True})
    finally:
        _remove_test_setting()

    assert result.ok
    assert result.data[0]["name"] == "demo"
    assert result.data[0]["parameters"] == DemoTool.parameters
    assert result.data[0]["requires_services"] == ["search"]
    assert [item["key"] for item in result.data[0]["config_settings"]] == [
        "demo_mode_tools_test"]


def test_tools_form_uses_static_schema_helper_and_quicklink(tmp_path):
    context, _ = _context(tmp_path)

    selected = _run(
        context, {"tool_name": "demo"}, method="form")
    calling = _run(
        context,
        {"tool_name": "demo", "action": "call"},
        method="form",
    )
    editing = _run(
        context,
        {
            "tool_name": "demo",
            "action": "edit_setting:demo_mode_tools_test",
        },
        method="form",
    )

    assert selected.data[1]["enum"] == [
        "call", "toggle_skip_permissions",
        "edit_setting:demo_mode_tools_test",
    ]
    assert [step["name"] for step in calling.data] == [
        "tool_name", "action", "query", "limit"]
    assert calling.data[2]["prompt"] == "Enter a query.\nWhat to echo."
    assert calling.data[3]["prompt_when_missing"] is True
    assert editing.data[-1]["type"] == "boolean"


def test_tools_list_output_matches_native_wire_format(tmp_path):
    context, _ = _context(tmp_path)

    result = _run(context, {})

    assert result.data == (
        "Tools:\n\n"
        "| Tool | Args | Description |\n"
        "| --- | --- | --- |\n"
        "| demo | query*, limit | "
        "Return a useful demo value. (needs: search) |"
    )


def test_tools_call_preserves_summary_and_user_initiated(tmp_path):
    context, registry = _context(tmp_path)
    seen = []
    original = registry.call

    def call(name, **kwargs):
        seen.append(dict(kwargs))
        return original(name, **kwargs)

    context.call_tool = call
    result = _run(
        context,
        {
            "tool_name": "demo",
            "action": "call",
            "query": "hello",
            "limit": 3,
        },
    )

    assert result.data == "Done: echoed hello"
    assert seen == [{
        "query": "hello",
        "limit": 3,
        "_user_initiated": True,
    }]


def test_tools_call_formats_table_and_failure_results(tmp_path):
    context, _ = _context(tmp_path)

    table = _run(
        context,
        {"tool_name": "demo", "action": "call", "query": "table"},
    )
    failed = _run(
        context,
        {"tool_name": "demo", "action": "call", "query": "fail"},
    )

    assert table.data == (
        "| Value |\n"
        "| --- |\n"
        "| one |\n"
        "| two |"
    )
    assert failed.data == "Failed: demo broke"


def test_tool_setting_quicklink_uses_config_write_scope(
        tmp_path, monkeypatch):
    context, _ = _context(tmp_path)
    plugin_discovery._collect_config_settings(
        DemoTool(), plugin_type="tool")
    try:
        result = _run(
            context,
            {
                "tool_name": "demo",
                "action": "edit_setting:demo_mode_tools_test",
                "value": True,
            },
            approve=lambda *_: True,
        )
    finally:
        _remove_test_setting()

    assert result.data == "Set demo_mode_tools_test = true"
    assert context.db.get_user_config(1)["demo_mode_tools_test"] is True
    assert "demo_mode_tools_test" not in context.runtime.config


def test_tools_toggle_skip_permissions_is_user_scoped(
        tmp_path, monkeypatch):
    context, _ = _context(tmp_path)

    enabled = _run(
        context,
        {"tool_name": "demo", "action": "toggle_skip_permissions"},
        approve=lambda *_: True,
    )
    context.config["skip_permissions"] = ["demo"]
    disabled = _run(
        context,
        {"tool_name": "demo", "action": "toggle_skip_permissions"},
        approve=lambda *_: True,
    )

    assert enabled.data == "Skip permissions enabled for demo."
    assert disabled.data == "Skip permissions disabled for demo."
    assert context.db.get_user_config(1)["skip_permissions"] == []


def _remove_test_setting():
    key = "demo_mode_tools_test"
    plugin_discovery._plugin_settings[:] = [
        entry for entry in plugin_discovery._plugin_settings
        if entry[1] != key
    ]
    plugin_discovery._plugin_settings_keys.discard(key)
    plugin_discovery._plugin_setting_types.pop(key, None)
    plugin_discovery._setting_to_plugins.pop(key, None)
