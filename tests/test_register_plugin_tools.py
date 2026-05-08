from pathlib import Path
from types import SimpleNamespace

from plugins.tools.tool_register_plugin import RegisterPlugin, _PLUGIN_CONFIG
from plugins.tools.tool_unregister_plugin import UnregisterPlugin


def _ctx(**kwargs):
    defaults = {"tool_registry": object(), "orchestrator": object(), "services": {}, "config": {}}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_register_plugin_schema_is_load_only():
    props = RegisterPlugin.parameters["properties"]

    assert set(props) == {"plugin_type", "file_name"}
    assert RegisterPlugin.parameters["required"] == ["plugin_type", "file_name"]


def test_register_plugin_loads_valid_sandbox_file(monkeypatch):
    calls = []
    sandbox = Path(".codex_register_plugin_test")
    path = sandbox / "tool_demo.py"
    try:
        sandbox.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        monkeypatch.setitem(_PLUGIN_CONFIG, "tool", (sandbox, "tool_", "sandbox_tools_{stem}"))
        monkeypatch.setattr("plugins.plugin_discovery.load_single_plugin", lambda *a, **k: calls.append((a, k)) or ("demo", None))
        monkeypatch.setattr("config.config_manager.reconcile_plugin_config", lambda *_: None)

        result = RegisterPlugin().run(_ctx(), plugin_type="tool", file_name="tool_demo.py")

        assert result.success
        assert calls and calls[0][0][:2] == ("tool", path)
        assert "Registered tool 'demo'" in result.llm_summary
    finally:
        path.unlink(missing_ok=True)
        sandbox.rmdir()


def test_register_plugin_validates_type_and_file_name():
    tool = RegisterPlugin()

    assert "Invalid plugin_type" in tool.run(_ctx(), plugin_type="nope", file_name="x.py").error
    assert "Tool files must start" in tool.run(_ctx(), plugin_type="tool", file_name="demo.py").error
    assert "File name must end" in tool.run(_ctx(), plugin_type="tool", file_name="tool_demo.txt").error


def test_unregister_plugin_calls_unload(monkeypatch):
    calls = []
    monkeypatch.setattr("plugins.plugin_discovery.unload_plugin", lambda *a, **k: calls.append((a, k)))

    result = UnregisterPlugin().run(_ctx(), plugin_type="tool", plugin_name="demo")

    assert result.success
    assert calls and calls[0][0][:2] == ("tool", "demo")
    assert "Unregistered tool 'demo'" in result.llm_summary


def test_unregister_plugin_validates_inputs_and_handles():
    tool = UnregisterPlugin()

    assert "Invalid plugin_type" in tool.run(_ctx(), plugin_type="nope", plugin_name="demo").error
    assert "plugin_name is required" in tool.run(_ctx(), plugin_type="tool", plugin_name="").error
    assert "No tool registry" in tool.run(_ctx(tool_registry=None), plugin_type="tool", plugin_name="demo").error
