from pathlib import Path
from types import SimpleNamespace

from plugins.tools.tool_test_plugin import TestPlugin


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
        monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", lambda *a, **k: ("demo", None))
        monkeypatch.setattr("plugins.tools.tool_test_plugin.unload_plugin", lambda *a, **k: None)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="1 passed", stderr=""))

        result = TestPlugin().run(_ctx(), plugin_path=str(path))

        assert result.success
        assert "Load check: ok: demo" in result.llm_summary
        assert "Pytest: passed" in result.llm_summary
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
        monkeypatch.setattr("plugins.tools.tool_test_plugin.load_single_plugin", lambda *a, **k: ("demo", None))
        monkeypatch.setattr("plugins.tools.tool_test_plugin.unload_plugin", lambda *a, **k: None)
        monkeypatch.setattr("plugins.tools.tool_test_plugin.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="FAILED test_x", stderr=""))

        result = TestPlugin().run(_ctx(), plugin_path=str(path))

        assert not result.success
        assert result.error == "Plugin test failed."
        assert "Pytest: failed" in result.llm_summary
        assert "FAILED test_x" in result.llm_summary
    finally:
        path.unlink(missing_ok=True)
        root_dir.rmdir()
