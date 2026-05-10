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
    return SimpleNamespace(config={"plugin_test_timeout": 5})


def _patch_plugin_dir(monkeypatch, plugin_type, directory):
    import plugins.helpers.plugin_paths as paths

    config = dict(paths.PLUGIN_CONFIG)
    built, sandbox, prefix, namespaces = config[plugin_type]
    config[plugin_type] = (Path(directory).resolve(), sandbox, prefix, namespaces)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)


def test_test_plugin_schema_is_path_only():
    props = TestPlugin.parameters["properties"]

    assert set(props) == {"plugin_path"}
    assert TestPlugin.parameters["required"] == ["plugin_path"]


def test_test_plugin_validates_bad_extension_before_load(monkeypatch):
    calls = []
    monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))

    result = TestPlugin().run(_ctx(), plugin_path="plugins/tools/tool_demo.txt")

    assert not result.success
    assert "File name must end" in result.error
    assert not calls


def test_test_plugin_validates_folder_mismatch(monkeypatch):
    calls = []
    monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", lambda *a, **k: calls.append(a) or ("demo", None))

    result = TestPlugin().run(_ctx(), plugin_path="plugins/tasks/tool_wrong.py")

    assert not result.success
    assert "files must start" in result.error
    assert not calls


def test_test_plugin_reports_load_and_pytest_success(monkeypatch):
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
    name = "demo"
    description = "Demo tool."
    parameters = {"type": "object", "properties": {}}

    def run(self, context, **kwargs):
        pass


class _BadTool(_GoodTool):
    parameters = {"type": "string"}


class _BadTask(BaseTask):
    name = "demo"


class _BadService(BaseService):
    model_name = "Demo"
    shared = "yes"

    def _load(self):
        return True

    def unload(self):
        pass


class _BadCommand(BaseCommand):
    name = "demo"


class _BadFrontend(BaseFrontend):
    name = "demo"
    capabilities = {}


def _load_fake_tool(*args, **kwargs):
    tool = _GoodTool()
    tool._source_path = str(args[1].resolve())
    kwargs["tool_registry"].register(tool)
    return "demo", None


def _load_bad_tool(*args, **kwargs):
    tool = _BadTool()
    tool._source_path = str(args[1].resolve())
    kwargs["tool_registry"].register(tool)
    return "demo", None
