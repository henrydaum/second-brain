"""Regression tests for test plugin tool."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.BaseCommand import BaseCommand
from plugins.BaseFrontend import BaseFrontend
from plugins.BaseService import BaseService
from plugins.BaseTask import BaseTask
from plugins.BaseTool import BaseTool
from plugins.tools.tool_test_plugin import TestPlugin, _diagnose


def _ctx():
    """Internal helper to handle ctx."""
    return SimpleNamespace(config={"plugin_test_timeout": 5})


def _patch_plugin_dir(monkeypatch, plugin_type, directory):
    """Internal helper to handle patch plugin dir."""
    import plugins.helpers.plugin_paths as paths

    config = dict(paths.PLUGIN_CONFIG)
    built, sandbox, prefix, namespaces = config[plugin_type]
    config[plugin_type] = (Path(directory).resolve(), sandbox, prefix, namespaces)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)


def test_test_plugin_schema_is_path_only():
    """Verify test plugin schema is path only."""
    props = TestPlugin.parameters["properties"]

    assert set(props) == {"plugin_path"}
    assert TestPlugin.parameters["required"] == ["plugin_path"]


def test_test_plugin_validates_bad_extension_before_load(monkeypatch):
    """Verify test plugin validates bad extension before load."""
    calls = []
    monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))

    result = TestPlugin().run(_ctx(), plugin_path="plugins/tools/tool_demo.txt")

    assert not result.success
    assert "File name must end" in result.error
    assert not calls


def test_test_plugin_validates_folder_mismatch(monkeypatch):
    """Verify test plugin validates folder mismatch."""
    calls = []
    monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))

    result = TestPlugin().run(_ctx(), plugin_path="plugins/tasks/tool_wrong.py")

    assert not result.success
    assert "files must start" in result.error
    assert not calls


def test_test_plugin_reports_load_and_pytest_success(monkeypatch):
    """Verify test plugin reports load and pytest success."""
    root_dir = Path(".codex_test_plugin_tools")
    path = root_dir / "tool_demo.py"
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, "tool", root_dir)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", _load_fake_tool)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.unload_plugin", lambda *a, **k: None)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="1 passed", stderr=""))

        result = TestPlugin().run(_ctx(), plugin_path=str(path))

        assert result.success
        assert "Load check: ok: demo" in result.llm_summary
        assert "Diagnostics:" in result.llm_summary
        assert "Pytest regression suite: passed" in result.llm_summary
        assert "pytest checks whether the app still works" in result.llm_summary
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_test_plugin_reports_pytest_failure(monkeypatch):
    """Verify test plugin reports pytest failure."""
    root_dir = Path(".codex_test_plugin_tools")
    path = root_dir / "tool_demo.py"
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, "tool", root_dir)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", _load_fake_tool)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.unload_plugin", lambda *a, **k: None)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="FAILED test_x", stderr=""))

        result = TestPlugin().run(_ctx(), plugin_path=str(path))

        assert not result.success
        assert result.error == "Plugin test failed."
        assert "Pytest regression suite: failed" in result.llm_summary
        assert "FAILED test_x" in result.llm_summary
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()


def test_test_plugin_reports_contract_suggestions(monkeypatch):
    """Verify test plugin reports contract suggestions."""
    root_dir = Path(".codex_test_plugin_tools")
    path = root_dir / "tool_demo.py"
    try:
        root_dir.mkdir(exist_ok=True)
        path.write_text("x", encoding="utf-8")
        _patch_plugin_dir(monkeypatch, "tool", root_dir)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", _load_bad_tool)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.unload_plugin", lambda *a, **k: None)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="1 passed", stderr=""))

        result = TestPlugin().run(_ctx(), plugin_path=str(path))

        assert not result.success
        assert "error: parameters" in result.llm_summary
        assert "Suggestion:" in result.llm_summary
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()


@pytest.mark.parametrize(
    ("plugin_type", "obj", "expected"),
    [
        ("tool", lambda: _BadTool(), "parameters"),
        ("task", lambda: _BadTask(), "run"),
        ("service", lambda: _BadService(), "shared"),
        ("command", lambda: _BadCommand(), "run"),
        ("frontend", lambda: _BadFrontend(), "capabilities"),
    ],
)
def test_diagnostics_cover_each_plugin_type(plugin_type, obj, expected):
    """Verify diagnostics cover each plugin type."""
    item = obj()
    item._source_path = "x.py"
    state = SimpleNamespace(
        tool_registry=SimpleNamespace(tools={"demo": item} if plugin_type == "tool" else {}),
        orchestrator=SimpleNamespace(tasks={"demo": item} if plugin_type == "task" else {}),
        command_registry=SimpleNamespace(_commands={"demo": item} if plugin_type == "command" else {}),
        services={"demo": item} if plugin_type == "service" else {},
        frontend_manager=SimpleNamespace(adapters={"demo": item} if plugin_type == "frontend" else {}),
    )

    diagnostics = _diagnose(plugin_type, "demo", state)

    assert any(d["check"] == expected and d["suggestion"] for d in diagnostics)


class _GoodTool(BaseTool):
    """Good tool."""
    name = "demo"
    description = "Demo tool."
    parameters = {"type": "object", "properties": {}}

    def run(self, context, **kwargs):
        """Execute the test test plugin tool tool."""
        pass


class _BadTool(_GoodTool):
    """Bad tool."""
    parameters = {"type": "string"}


class _BadTask(BaseTask):
    """Bad task."""
    name = "demo"


class _BadService(BaseService):
    """Bad service."""
    model_name = "Demo"
    shared = "yes"

    def _load(self):
        """Internal helper to load bad service."""
        return True

    def unload(self):
        """Handle unload."""
        pass


class _BadCommand(BaseCommand):
    """Bad command."""
    name = "demo"


class _BadFrontend(BaseFrontend):
    """Bad frontend."""
    name = "demo"
    capabilities = {}


def _load_fake_tool(*args, **kwargs):
    """Internal helper to load fake tool."""
    tool = _GoodTool()
    tool._source_path = str(args[1].resolve())
    kwargs["tool_registry"].register(tool)
    return "demo", None


def _load_bad_tool(*args, **kwargs):
    """Internal helper to load bad tool."""
    tool = _BadTool()
    tool._source_path = str(args[1].resolve())
    kwargs["tool_registry"].register(tool)
    return "demo", None
