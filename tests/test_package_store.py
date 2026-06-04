"""Tests for package store install/uninstall V1."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from plugins.commands.helpers import package_manager
from plugins.commands.command_packages import PackagesCommand, _cleanup_choices
from plugins.commands.helpers.store_backend import GitStoreBackend, StoreBackendError
from plugins import plugin_discovery


class _ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.unloaded = []

    def register(self, tool):
        self.tools[tool.name] = tool

    def unregister(self, name):
        self.unloaded.append(name)
        self.tools.pop(name, None)


class _Context:
    def __init__(self, root_dir, registry):
        self.root_dir = root_dir
        self.tool_registry = registry
        self.orchestrator = None
        self.services = {}
        self.config = {}
        self.command_registry = None
        self.runtime = None
        self.request_user_input = None
        self.db = None


class _Backend:
    def __init__(self, manifests, files):
        self.manifests = manifests
        self.files = files

    def get_manifest(self, package_id):
        if package_id not in self.manifests:
            raise package_manager.PackageError(f"missing manifest: {package_id}")
        return self.manifests[package_id]

    def get_manifest_bytes(self, package_id):
        return json.dumps(self.get_manifest(package_id), sort_keys=True).encode()

    def get_file_bytes(self, package_id, rel_path):
        try:
            return self.files[(package_id, rel_path)]
        except KeyError:
            raise package_manager.PackageError(f"missing file: {package_id}/{rel_path}")


def _patch_install_root(monkeypatch, tmp_path):
    import plugins.helpers.plugin_paths as paths

    installed = tmp_path / "installed_plugins"
    receipts = tmp_path / "packages" / "receipts"
    roots = (paths.PluginRoot("installed", installed, "installed_plugins"),)
    config = dict(paths.PLUGIN_CONFIG)
    config["tool"] = (paths.PluginDir(roots[0], "tool", "tools", "tool_"),)
    config["task"] = (paths.PluginDir(roots[0], "task", "tasks", "task_"),)
    monkeypatch.setattr(paths, "PLUGIN_ROOTS", roots)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)
    monkeypatch.setattr(package_manager, "INSTALLED_PLUGINS", installed)
    monkeypatch.setattr(package_manager, "RECEIPTS_DIR", receipts)
    monkeypatch.setattr(plugin_discovery, "PLUGIN_ROOTS", roots)
    return installed, receipts


class _Db:
    _validate_identifier = staticmethod(lambda name: None)

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.lock = threading.Lock()


def _tool_source(summary='"echo ok"'):
    return (
        "from plugins.BaseTool import BaseTool, ToolResult\n"
        "from .helpers.echo_format import fmt\n\n"
        "class EchoTool(BaseTool):\n"
        "    name = 'echo'\n"
        "    description = 'Echo test tool.'\n"
        "    parameters = {}\n"
        "    def run(self, context, **kwargs):\n"
        f"        return ToolResult(llm_summary=fmt({summary}))\n"
    ).encode()


def test_manifest_validation_rejects_bad_paths():
    manifest = {"id": "bad", "requires": [], "files": ["../nope.py"]}
    with pytest.raises(package_manager.PackageError):
        package_manager._validate_manifest(manifest)
    manifest = {"id": "bad", "requires": [], "files": ["unknown/file.py"]}
    with pytest.raises(package_manager.PackageError):
        package_manager._validate_manifest(manifest)


def test_store_backend_reads_store_branch_without_checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "main"], cwd=repo, check=True, stdout=subprocess.PIPE)
    main_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    subprocess.run(["git", "checkout", "-b", "store"], cwd=repo, check=True, stdout=subprocess.PIPE)
    (repo / "packages" / "echo-tool").mkdir(parents=True)
    (repo / "packages" / "index.json").write_text('{"packages":[{"id":"echo-tool"}]}', encoding="utf-8")
    (repo / "packages" / "echo-tool" / "manifest.json").write_text('{"id":"echo-tool","files":[],"requires":[]}', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "store"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "checkout", main_branch], cwd=repo, check=True, stdout=subprocess.PIPE)

    backend = GitStoreBackend(repo, ref="store")

    assert backend.get_index()[0]["id"] == "echo-tool"
    assert backend.get_manifest("echo-tool")["id"] == "echo-tool"
    assert not (repo / "packages").exists()


def test_install_copies_loads_and_writes_receipt(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {"echo-tool": {"id": "echo-tool", "name": "Echo", "description": "", "requires": [], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]}},
        {
            ("echo-tool", "tools/tool_echo.py"): _tool_source(),
            ("echo-tool", "tools/helpers/echo_format.py"): b"def fmt(value):\n    return value\n",
        },
    )
    registry = _ToolRegistry()
    context = _Context(tmp_path, registry)
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    monkeypatch.setattr(plugin_discovery, "load_single_plugin", lambda *a, **k: pytest.fail("package install should not register plugins"))

    result = package_manager.install_package(tmp_path, "echo-tool", context)

    assert result.ok
    assert registry.tools == {}
    receipt = package_manager.installed_packages()[0]
    assert receipt["id"] == "echo-tool"
    assert receipt["entrypoints"][0]["path"] == "tools/tool_echo.py"
    assert receipt["entrypoints"][0]["type"] == "tool"


def test_install_pip_installs_missing_imports_in_current_python(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {"service-litellm": {"id": "service-litellm", "requires": [], "files": ["services/service_litellm.py"], "entrypoints": []}},
        {("service-litellm", "services/service_litellm.py"): b"import pathlib\nimport litellm\nfrom plugins.services.service_llm import BaseLLM\n"},
    )
    calls = []
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    monkeypatch.setattr(package_manager.importlib.util, "find_spec", lambda name: None if name == "litellm" else object())
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.install_package(tmp_path, "service-litellm", _Context(tmp_path, _ToolRegistry()))

    assert calls[0][0] == [sys.executable, "-m", "pip", "install", "litellm"]
    assert "Installed Python package(s): litellm" in result.lines
    assert package_manager.installed_packages()[0]["pip_packages"] == ["litellm"]


def test_install_pip_failure_aborts_package_install(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {"bad": {"id": "bad", "requires": [], "files": ["tools/tool_bad.py"], "entrypoints": []}},
        {("bad", "tools/tool_bad.py"): b"import definitely_missing_package\n"},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    monkeypatch.setattr(package_manager.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", "nope"))

    with pytest.raises(package_manager.PackageError, match="pip install failed"):
        package_manager.install_package(tmp_path, "bad", _Context(tmp_path, _ToolRegistry()))

    assert not (installed / "tools" / "tool_bad.py").exists()
    assert not package_manager.installed_packages()


def test_install_resolves_full_graph_before_writing_files(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle": {"id": "bundle", "requires": ["missing-file", "echo-tool"], "files": []},
            "missing-file": {"id": "missing-file", "requires": [], "files": ["helpers/missing.txt"]},
            "echo-tool": {"id": "echo-tool", "requires": [], "files": ["tools/tool_echo.py"]},
        },
        {("echo-tool", "tools/tool_echo.py"): _tool_source()},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    with pytest.raises(package_manager.PackageError, match="missing file"):
        package_manager.install_package(tmp_path, "bundle", _Context(tmp_path, _ToolRegistry()))

    assert not (installed / "tools" / "tool_echo.py").exists()
    assert not package_manager.installed_packages()


def test_install_auto_installs_dependency(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "base": {"id": "base", "requires": [], "files": ["helpers/base.txt"]},
            "echo-tool": {"id": "echo-tool", "requires": ["base"], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]},
        },
        {
            ("base", "helpers/base.txt"): b"base",
            ("echo-tool", "tools/tool_echo.py"): _tool_source(),
            ("echo-tool", "tools/helpers/echo_format.py"): b"def fmt(value):\n    return value\n",
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    package_manager.install_package(tmp_path, "echo-tool", _Context(tmp_path, _ToolRegistry()))
    receipts = {r["id"]: r for r in package_manager.installed_packages()}

    assert receipts["base"]["requested"] is False
    assert receipts["echo-tool"]["requires"] == ["base"]


def test_bundle_install_reloads_parser_once_for_multiple_helpers(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle": {"id": "bundle", "requires": ["parser-one", "parser-two"], "files": []},
            "parser-one": {"id": "parser-one", "requires": [], "files": ["services/helpers/parse_one.py"], "entrypoints": []},
            "parser-two": {"id": "parser-two", "requires": [], "files": ["services/helpers/parse_two.py"], "entrypoints": []},
        },
        {
            ("parser-one", "services/helpers/parse_one.py"): b"",
            ("parser-two", "services/helpers/parse_two.py"): b"",
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    parser = type("Parser", (), {"loaded": True, "loads": 0, "unloads": 0})()
    parser.load = lambda: setattr(parser, "loads", parser.loads + 1)
    parser.unload = lambda: setattr(parser, "unloads", parser.unloads + 1)

    context = _Context(tmp_path, _ToolRegistry())
    context.services = {"parser": parser}
    result = package_manager.install_package(tmp_path, "bundle", context)

    assert parser.loads == 1
    assert result.lines.count("Reloaded parser service; file parsers are now active.") == 1


def test_install_refuses_unowned_file_collision(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    target = installed / "tools" / "tool_echo.py"
    target.parent.mkdir(parents=True)
    target.write_text("mine", encoding="utf-8")
    backend = _Backend(
        {"echo-tool": {"id": "echo-tool", "requires": [], "files": ["tools/tool_echo.py", "helpers/new.txt"]}},
        {("echo-tool", "tools/tool_echo.py"): _tool_source(), ("echo-tool", "helpers/new.txt"): b"new"},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    with pytest.raises(package_manager.PackageError):
        package_manager.install_package(tmp_path, "echo-tool", _Context(tmp_path, _ToolRegistry()))
    assert not (installed / "helpers" / "new.txt").exists()


def test_uninstall_removes_files_receipt_and_prunes_auto_dependency(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "base": {"id": "base", "requires": [], "files": ["helpers/base.txt"]},
            "echo-tool": {"id": "echo-tool", "requires": ["base"], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]},
        },
        {
            ("base", "helpers/base.txt"): b"base",
            ("echo-tool", "tools/tool_echo.py"): _tool_source(),
            ("echo-tool", "tools/helpers/echo_format.py"): b"def fmt(value):\n    return value\n",
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    registry = _ToolRegistry()
    context = _Context(tmp_path, registry)
    package_manager.install_package(tmp_path, "echo-tool", context)
    monkeypatch.setattr(plugin_discovery, "unload_plugin", lambda *a, **k: pytest.fail("package uninstall should not unload plugins"))

    result = package_manager.uninstall_package("echo-tool", context)

    assert result.ok
    assert not (installed / "tools" / "tool_echo.py").exists()
    assert not package_manager.installed_packages()
    assert registry.unloaded == []


def test_uninstall_refuses_when_another_package_depends_on_target(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "base": {"id": "base", "requires": [], "files": ["helpers/base.txt"]},
            "echo-tool": {"id": "echo-tool", "requires": ["base"], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]},
        },
        {
            ("base", "helpers/base.txt"): b"base",
            ("echo-tool", "tools/tool_echo.py"): _tool_source(),
            ("echo-tool", "tools/helpers/echo_format.py"): b"def fmt(value):\n    return value\n",
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    context = _Context(tmp_path, _ToolRegistry())
    package_manager.install_package(tmp_path, "echo-tool", context)

    with pytest.raises(package_manager.PackageError):
        package_manager.uninstall_package("base", context)


def test_uninstall_can_delete_owned_config_and_tables_after_approval(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {"pkg": {"id": "pkg", "requires": [], "files": ["tasks/task_owned.py"], "entrypoints": []}},
        {("pkg", "tasks/task_owned.py"): (
            "from plugins.BaseTask import BaseTask\n"
            "class OwnedTask(BaseTask):\n"
            "    name = 'owned'\n"
            "    writes = ['owned_table']\n"
            "    config_settings = [('Owned', 'owned_key', '', 'x', {})]\n"
        ).encode()},
    )
    saved = {}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    monkeypatch.setattr("config.config_manager.load_plugin_config", lambda: {"owned_key": "secret", "keep": "v"})
    monkeypatch.setattr("config.config_manager.save_plugin_config", lambda data: saved.update(data))
    context = _Context(tmp_path, _ToolRegistry())
    context.config = {"owned_key": "secret"}
    context.db = _Db()
    context.db.conn.execute("CREATE TABLE owned_table (id INTEGER)")
    package_manager.install_package(tmp_path, "pkg", context)

    result = package_manager.uninstall_package("pkg", context, cleanup_choices={"config": {"pkg": True}, "tables": {"pkg": True}, "pip": {}})

    assert "Deleted config setting(s): owned_key" in result.lines
    assert "Deleted table(s): owned_table" in result.lines
    assert "owned_key" not in saved
    assert "owned_key" not in context.config
    with pytest.raises(sqlite3.OperationalError):
        context.db.conn.execute("SELECT * FROM owned_table")
    assert not (installed / "tasks" / "task_owned.py").exists()


def test_uninstall_keeps_state_used_by_other_plugins(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    other = installed / "tasks" / "task_other.py"
    other.parent.mkdir(parents=True)
    other.write_text(
        "from plugins.BaseTask import BaseTask\n"
        "class OtherTask(BaseTask):\n"
        "    name = 'other'\n"
        "    reads = ['shared_table']\n"
        "    config_settings = [('Shared', 'shared_key', '', 'x', {})]\n",
        encoding="utf-8",
    )
    backend = _Backend(
        {"pkg": {"id": "pkg", "requires": [], "files": ["tasks/task_owned.py"], "entrypoints": []}},
        {("pkg", "tasks/task_owned.py"): (
            "from plugins.BaseTask import BaseTask\n"
            "class OwnedTask(BaseTask):\n"
            "    name = 'owned'\n"
            "    writes = ['shared_table']\n"
            "    config_settings = [('Shared', 'shared_key', '', 'x', {})]\n"
        ).encode()},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    context = _Context(tmp_path, _ToolRegistry())
    context.request_user_input = lambda *a, **k: pytest.fail("should not ask")
    package_manager.install_package(tmp_path, "pkg", context)

    result = package_manager.uninstall_package("pkg", context)

    assert "Kept config setting(s) still declared by other plugins: shared_key" in result.lines
    assert "Kept table(s) still used by remaining tasks; their data may now be stale: shared_table" in result.lines


def test_uninstall_without_approval_keeps_cleanup_data(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {"pkg": {"id": "pkg", "requires": [], "files": ["tools/tool_cfg.py"], "entrypoints": []}},
        {("pkg", "tools/tool_cfg.py"): b"config_settings = [('Owned', 'owned_key', '', 'x', {})]\n"},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    context = _Context(tmp_path, _ToolRegistry())
    package_manager.install_package(tmp_path, "pkg", context)

    result = package_manager.uninstall_package("pkg", context)

    assert "Kept package config setting(s)." in result.lines


def test_packages_uninstall_form_collects_pruned_dependency_cleanup(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "starter": {"id": "starter", "requires": ["task-owned"], "files": []},
            "task-owned": {"id": "task-owned", "requires": [], "files": ["tasks/task_owned.py"], "entrypoints": []},
        },
        {("task-owned", "tasks/task_owned.py"): b"config_settings = [('Owned', 'owned_key', '', 'x', {})]\n"},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    context = _Context(tmp_path, _ToolRegistry())
    package_manager.install_package(tmp_path, "starter", context)

    command = PackagesCommand()
    steps = command.form({"action": "uninstall", "package_id": "starter"}, context)

    assert [step.name for step in steps] == ["action", "package_id", "cleanup_config"]
    assert steps[-1].enum == ["all", "none", "specific"]
    assert steps[-1].enum_labels == ["All", "None", "Specific"]
    assert steps[-1].default == "all"
    plan = package_manager.build_uninstall_plan("starter")
    assert _cleanup_choices(plan, {})["config"] == {"starter": False, "task-owned": True}
    assert _cleanup_choices(plan, {"cleanup_config": "none"})["config"] == {"starter": False, "task-owned": False}
    assert _cleanup_choices(plan, {"cleanup_config": "specific", "cleanup_config__task_owned": True})["config"] == {"starter": False, "task-owned": True}
    steps = command.form({"action": "uninstall", "package_id": "starter", "cleanup_config": "specific"}, context)
    assert [step.name for step in steps] == ["action", "package_id", "cleanup_config", "cleanup_config__task_owned"]
    assert "Config settings: owned_key" in steps[-1].prompt
    context.request_user_input = lambda *a, **k: pytest.fail("cleanup should be collected by the command form")
    result = command.run({"action": "uninstall", "package_id": "starter", "cleanup_config": "none"}, context)

    assert "Kept package config setting(s)." in result
    assert not package_manager.installed_packages()


def test_uninstall_pip_cleanup_only_removes_safe_candidates(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    package_manager._write_receipt({"id": "pkg-a", "requested": True, "requires": [], "files": [], "entrypoints": [], "pip_packages": ["orphan-lib", "litellm", "shared-lib"]})
    package_manager._write_receipt({"id": "pkg-b", "requested": True, "requires": [], "files": [], "entrypoints": [], "pip_packages": ["shared-lib"]})
    calls = []
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    plan = package_manager.build_uninstall_plan("pkg-a")
    result = package_manager.execute_uninstall_plan(plan, _Context(tmp_path, _ToolRegistry()), {"pip": {"pkg-a": True}})

    assert plan.pip_removals == {"pkg-a": ["orphan-lib"]}
    assert calls == [[sys.executable, "-m", "pip", "uninstall", "-y", "orphan-lib"]]
    assert "Kept Python package(s): litellm (kernel requirement), shared-lib (needed by another installed package)" in result.lines


def test_execute_uses_package_operation_lock(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)

    class Lock:
        entered = 0

        def __enter__(self):
            self.entered += 1

        def __exit__(self, *_exc):
            return False

    lock = Lock()
    monkeypatch.setattr(package_manager, "_PACKAGE_LOCK", lock)
    plan = package_manager.InstallPlan("empty", ["empty"], [], [], [], [], False, [])

    result = package_manager.execute_install_plan(plan, _Context(tmp_path, _ToolRegistry()))

    assert result.ok
    assert lock.entered == 1


def test_legacy_receipt_entrypoint_names_do_not_block_uninstall(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    target = installed / "tools" / "tool_legacy.py"
    target.parent.mkdir(parents=True)
    target.write_text("class Legacy: pass\n", encoding="utf-8")
    package_manager._write_receipt({
        "id": "legacy",
        "requested": True,
        "requires": [],
        "files": [{"path": "tools/tool_legacy.py", "sha256": ""}],
        "entrypoints": [{"path": "tools/tool_legacy.py", "type": "tool", "name": "legacy_name"}],
        "pip_packages": [],
    })
    monkeypatch.setattr(plugin_discovery, "unload_plugin", lambda *a, **k: pytest.fail("legacy uninstall should not call plugin unload"))

    result = package_manager.uninstall_package("legacy", _Context(tmp_path, _ToolRegistry()))

    assert result.ok
    assert not target.exists()
    assert not package_manager.installed_packages()


def test_install_plan_uses_one_store_cache_per_operation(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)

    class CountingBackend(_Backend):
        def __init__(self):
            super().__init__(
                {"pkg": {"id": "pkg", "requires": [], "files": ["tools/tool_echo.py"], "entrypoints": []}},
                {("pkg", "tools/tool_echo.py"): _tool_source()},
            )
            self.manifest_bytes_calls = 0
            self.file_calls = 0

        def get_manifest_bytes(self, package_id):
            self.manifest_bytes_calls += 1
            return super().get_manifest_bytes(package_id)

        def get_file_bytes(self, package_id, rel_path):
            self.file_calls += 1
            return super().get_file_bytes(package_id, rel_path)

    backend = CountingBackend()
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    package_manager.build_install_plan(tmp_path, "pkg")

    assert backend.manifest_bytes_calls == 1
    assert backend.file_calls == 1


def test_packages_install_missing_package_hides_git_manifest_error(tmp_path, monkeypatch):
    def missing(*_args, **_kwargs):
        raise StoreBackendError("Could not read origin/store:packages/hfs/manifest.json: fatal: path 'packages/hfs/manifest.json' does not exist in 'origin/store'")

    monkeypatch.setattr(package_manager, "install_package", missing)

    result = PackagesCommand().run({"action": "install", "package_id": "hfs"}, _Context(tmp_path, _ToolRegistry()))

    assert result == "Package install failed: 'hfs' not found."
