"""Tests for package store install/uninstall V1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from plugins.commands.helpers import package_manager
from plugins.commands.helpers.store_backend import GitStoreBackend
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
    monkeypatch.setattr(paths, "PLUGIN_ROOTS", roots)
    monkeypatch.setattr(paths, "PLUGIN_CONFIG", config)
    monkeypatch.setattr(package_manager, "INSTALLED_PLUGINS", installed)
    monkeypatch.setattr(package_manager, "RECEIPTS_DIR", receipts)
    monkeypatch.setattr(plugin_discovery, "PLUGIN_ROOTS", roots)
    return installed, receipts


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

    result = package_manager.install_package(tmp_path, "echo-tool", context)

    assert result.ok
    assert registry.tools["echo"].run(None).llm_summary == "echo ok"
    receipt = package_manager.installed_packages()[0]
    assert receipt["id"] == "echo-tool"
    assert receipt["entrypoints"][0]["path"] == "tools/tool_echo.py"


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


def test_install_refuses_unowned_file_collision(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    target = installed / "tools" / "tool_echo.py"
    target.parent.mkdir(parents=True)
    target.write_text("mine", encoding="utf-8")
    backend = _Backend(
        {"echo-tool": {"id": "echo-tool", "requires": [], "files": ["tools/tool_echo.py"]}},
        {("echo-tool", "tools/tool_echo.py"): _tool_source()},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    with pytest.raises(package_manager.PackageError):
        package_manager.install_package(tmp_path, "echo-tool", _Context(tmp_path, _ToolRegistry()))


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

    result = package_manager.uninstall_package("echo-tool", context)

    assert result.ok
    assert not (installed / "tools" / "tool_echo.py").exists()
    assert not package_manager.installed_packages()
    assert "echo" in registry.unloaded


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
