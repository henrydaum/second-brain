"""Tests for tree-based package-store install/uninstall."""

from __future__ import annotations

import subprocess
import json
from pathlib import Path

import pytest

from plugins.commands.helpers import package_manager
from plugins.helpers import plugin_paths


class _Backend:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def list_python_files(self):
        return sorted(path for path in self.files if path.endswith(".py"))

    def list_tree_files(self):
        return sorted(self.files)

    def get_tree_file_bytes(self, rel):
        try:
            return self.files[rel]
        except KeyError:
            raise package_manager.PackageError(f"missing file: {rel}")


class _Context:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.config = {}
        self.runtime = None
        self.services = {}


class _Parser:
    loaded = True

    def __init__(self):
        self.loads = 0
        self.unloads = 0

    def load(self):
        self.loads += 1

    def unload(self):
        self.unloads += 1


def _patch_roots(monkeypatch, tmp_path):
    installed = tmp_path / "installed_plugins"
    sandbox = tmp_path / "sandbox_plugins"
    built_in = tmp_path / "plugins"
    roots = (
        plugin_paths.PluginRoot("built_in", built_in, "plugins", True),
        plugin_paths.PluginRoot("sandbox", sandbox, "sandbox_plugins"),
        plugin_paths.PluginRoot("installed", installed, "installed_plugins"),
    )
    monkeypatch.setattr(package_manager, "INSTALLED_PLUGINS", installed)
    monkeypatch.setattr(package_manager, "PLUGIN_ROOTS", roots)
    monkeypatch.setattr(plugin_paths, "PLUGIN_ROOTS", roots)
    return built_in, sandbox, installed


def _write(root: Path, rel: str, text: str):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _tool(deps=(), pip=()):
    return (
        "from plugins.BaseTool import BaseTool\n"
        "class T(BaseTool):\n"
        "    name = 't'\n"
        f"    dependencies_files = {list(deps)!r}\n"
        f"    dependencies_pip = {list(pip)!r}\n"
    ).encode()


def _helper(deps=(), pip=()):
    return (
        f"dependencies_files = {list(deps)!r}\n"
        f"dependencies_pip = {list(pip)!r}\n"
        "VALUE = 1\n"
    ).encode()


def test_metadata_parser_reads_class_and_module_fields():
    plugin = package_manager.read_dependency_meta(
        "tools/tool_x.py",
        "class X:\n    dependencies_files = ['tools/helpers/x.py']\n    dependencies_pip = ['lib-x']\n",
    )
    helper = package_manager.read_dependency_meta(
        "tools/helpers/x.py",
        "dependencies_files = []\ndependencies_pip = ['helper-lib']\n",
    )

    assert plugin.dependencies_files == ("tools/helpers/x.py",)
    assert plugin.dependencies_pip == ("lib-x",)
    assert helper.dependencies_pip == ("helper-lib",)


def test_install_telegram_shape_copies_frontend_helper_and_pip(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    calls = []
    files = {
        "frontends/frontend_telegram.py": (
            "from plugins.BaseFrontend import BaseFrontend\n"
            "class Telegram(BaseFrontend):\n"
            "    dependencies_files = ['frontends/helpers/telegram_renderers.py']\n"
            "    dependencies_pip = ['python-telegram-bot']\n"
        ).encode(),
        "frontends/helpers/telegram_renderers.py": _helper(),
    }
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.install_package(tmp_path, "frontend_telegram", _Context(tmp_path))

    assert result.ok
    assert (package_manager.INSTALLED_PLUGINS / "frontends" / "frontend_telegram.py").exists()
    assert (package_manager.INSTALLED_PLUGINS / "frontends" / "helpers" / "telegram_renderers.py").exists()
    assert calls == [[__import__("sys").executable, "-m", "pip", "install", "python-telegram-bot"]]


def test_install_frontend_preserves_existing_saved_frontends(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {"frontends/frontend_telegram.py": b"from plugins.BaseFrontend import BaseFrontend\nclass Telegram(BaseFrontend): pass\n"}
    saved = {"enabled_frontends": ["repl"], "autoload_services": ["llm"]}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    result = package_manager.install_package(tmp_path, "frontend_telegram", _Context(tmp_path))

    assert result.ok
    assert saved["enabled_frontends"] == ["repl", "telegram"]


def test_install_service_preserves_existing_saved_autoload_services(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {"services/service_mcp.py": b"from plugins.BaseService import BaseService\nclass MCP(BaseService): pass\n"}
    saved = {"enabled_frontends": ["repl"], "autoload_services": ["llm", "parser"]}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    result = package_manager.install_package(tmp_path, "service_mcp", _Context(tmp_path))

    assert result.ok
    assert saved["autoload_services"] == ["llm", "parser", "mcp"]


def test_helper_can_be_installed_and_uninstalled_by_stem(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    calls = []
    files = {"tools/helpers/shared.py": _helper(pip=["helper-lib"])}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    package_manager.install_package(tmp_path, "shared", _Context(tmp_path))
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "helpers" / "shared.py").exists()

    result = package_manager.uninstall_package("shared", _Context(tmp_path))

    assert result.ok
    assert not (package_manager.INSTALLED_PLUGINS / "tools" / "helpers" / "shared.py").exists()
    assert calls[-1] == [__import__("sys").executable, "-m", "pip", "uninstall", "-y", "helper-lib"]


def test_recursive_helper_dependencies_are_collected(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {
        "tools/tool_a.py": _tool(deps=["tools/helpers/a.py"], pip=["tool-lib"]),
        "tools/helpers/a.py": _helper(deps=["tools/helpers/b.py"], pip=["a-lib"]),
        "tools/helpers/b.py": _helper(pip=["b-lib"]),
    }
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))

    plan = package_manager.build_install_plan(tmp_path, "tool_a")

    assert [file.path for file in plan.files] == ["tools/tool_a.py", "tools/helpers/a.py", "tools/helpers/b.py"]
    assert plan.pip_packages == ["tool-lib", "a-lib", "b-lib"]


def test_bundle_install_collects_each_root_once(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {
        "bundles/bundle_search.json": json.dumps({"name": "Search", "files": ["tools/tool_a.py", "tools/tool_b.py"]}).encode(),
        "tools/tool_a.py": _tool(deps=["tools/helpers/shared.py"], pip=["a-lib"]),
        "tools/tool_b.py": _tool(deps=["tools/helpers/shared.py"], pip=["b-lib"]),
        "tools/helpers/shared.py": _helper(pip=["shared-lib"]),
    }
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))

    plan = package_manager.build_install_plan(tmp_path, "bundle_search")

    assert [file.path for file in plan.files] == ["tools/tool_a.py", "tools/helpers/shared.py", "tools/tool_b.py"]
    assert plan.pip_packages == ["a-lib", "shared-lib", "b-lib"]


def test_bundle_uninstall_skips_missing_roots_and_keeps_shared_refs(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_a.py", _tool(deps=["tools/helpers/shared.py"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_c.py", _tool(deps=["tools/helpers/shared.py"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/helpers/shared.py", _helper().decode())
    files = {"bundles/bundle_search.json": json.dumps({"files": ["tools/tool_a.py", "tools/tool_b.py"]}).encode()}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.uninstall_package("bundle_search", _Context(tmp_path), root_dir=tmp_path)

    assert result.ok
    assert not (package_manager.INSTALLED_PLUGINS / "tools" / "tool_a.py").exists()
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "helpers" / "shared.py").exists()


def test_uninstall_keeps_file_and_pip_referenced_by_other_installed_plugin(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_a.py", _tool(deps=["tools/helpers/shared.py"], pip=["shared-lib"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_b.py", _tool(deps=["tools/helpers/shared.py"], pip=["shared-lib"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/helpers/shared.py", _helper(pip=["shared-lib"]).decode())
    calls = []
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.uninstall_package("tool_a", _Context(tmp_path))

    assert result.ok
    assert not (package_manager.INSTALLED_PLUGINS / "tools" / "tool_a.py").exists()
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "helpers" / "shared.py").exists()
    assert calls == []
    assert any("Kept file: tools/helpers/shared.py" in line for line in result.lines)


def test_uninstall_keeps_dependency_referenced_by_builtin_or_sandbox(tmp_path, monkeypatch):
    built_in, sandbox, installed = _patch_roots(monkeypatch, tmp_path)
    _write(installed, "tools/tool_a.py", _tool(deps=["tools/helpers/shared.py"]).decode())
    _write(installed, "tools/helpers/shared.py", _helper(pip=["shared-lib"]).decode())
    _write(built_in, "tools/tool_builtin.py", _tool(deps=["tools/helpers/shared.py"], pip=["shared-lib"]).decode())
    _write(sandbox, "tools/tool_sandbox.py", _tool(pip=["shared-lib"]).decode())
    calls = []
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.uninstall_package("tool_a", _Context(tmp_path))

    assert (installed / "tools" / "helpers" / "shared.py").exists()
    assert calls == []
    assert "Kept Python package(s): shared-lib" in "\n".join(result.lines)


def test_parser_helper_install_and_uninstall_reload_parser(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    parser = _Parser()
    context = _Context(tmp_path)
    context.services = {"parser": parser}
    files = {"services/helpers/parse_pdf.py": _helper()}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))

    package_manager.install_package(tmp_path, "parse_pdf", context)
    package_manager.uninstall_package("parse_pdf", context)

    assert parser.loads == 2
    assert parser.unloads == 2


def test_existing_receipts_are_ignored(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    receipts = tmp_path / "packages" / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "ghost.json").write_text('{"id": "ghost"}', encoding="utf-8")

    assert package_manager.installed_packages() == []
