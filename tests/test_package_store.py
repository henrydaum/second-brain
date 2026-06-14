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


class _ColdService:
    loaded = False

    def __init__(self):
        self.loads = 0

    def load(self):
        self.loads += 1
        self.loaded = True
        return True


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


def test_install_loads_service_registered_before_autoload_update(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {"services/service_mcp.py": b"from plugins.BaseService import BaseService\nclass MCP(BaseService): pass\n"}
    saved = {"enabled_frontends": ["repl"], "autoload_services": ["llm"]}
    context = _Context(tmp_path)
    context.services["mcp"] = _ColdService()
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    result = package_manager.install_package(tmp_path, "service_mcp", context)

    assert result.ok
    assert saved["autoload_services"] == ["llm", "mcp"]
    assert context.services["mcp"].loads == 1
    assert "Loaded service: mcp" in result.text()


def test_install_llm_backend_autoloads_llm_router_not_backend_stem(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {"services/service_litellm.py": b"from plugins.services.service_llm import BaseLLM\nclass LiteLLMService(BaseLLM):\n    is_llm_backend = True\n"}
    saved = {"enabled_frontends": ["repl"], "autoload_services": ["llm", "parser"]}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    result = package_manager.install_package(tmp_path, "service_litellm", _Context(tmp_path))

    assert result.ok
    # The backend maps to the kernel ``llm`` router (already present, idempotent),
    # never adds a bogus ``litellm`` autoload entry.
    assert saved["autoload_services"] == ["llm", "parser"]


def test_uninstall_llm_backend_keeps_llm_in_autoload(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    _write(package_manager.INSTALLED_PLUGINS, "services/service_litellm.py",
           "from plugins.services.service_llm import BaseLLM\nclass LiteLLMService(BaseLLM):\n    is_llm_backend = True\n")
    saved = {"enabled_frontends": ["repl"], "autoload_services": ["llm", "parser"]}
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    result = package_manager.uninstall_package("service_litellm", _Context(tmp_path), root_dir=tmp_path)

    assert result.ok
    assert not (package_manager.INSTALLED_PLUGINS / "services" / "service_litellm.py").exists()
    # ``llm`` is the kernel-owned router and must survive uninstall of a backend.
    assert saved["autoload_services"] == ["llm", "parser"]


def test_install_replaces_existing_file_with_store_copy(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {"tools/tool_a.py": b"STORE = True\n"}
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_a.py", "STORE = False\n")
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.install_package(tmp_path, "tool_a", _Context(tmp_path))

    assert result.ok
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "tool_a.py").read_text(encoding="utf-8") == "STORE = True\n"
    assert "Updated file: tools/tool_a.py" in result.lines


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


def test_bundle_install_replaces_existing_files_and_continues(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    files = {
        "bundles/bundle_search.json": json.dumps({"files": ["tools/tool_a.py", "tools/tool_b.py"]}).encode(),
        "tools/tool_a.py": b"STORE_A = True\n",
        "tools/tool_b.py": b"STORE_B = True\n",
    }
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_a.py", "STORE_A = False\n")
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.install_package(tmp_path, "bundle_search", _Context(tmp_path))

    assert result.ok
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "tool_a.py").read_text(encoding="utf-8") == "STORE_A = True\n"
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "tool_b.py").exists()
    assert "Updated file: tools/tool_a.py" in result.lines


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


# ──────────────────────────────────────────────────────────────────────
# Skills as store packages (skills/<name>/ folders)
# ──────────────────────────────────────────────────────────────────────

def _skill_md(name, deps="tools/tool_use_skill.py"):
    return (
        f"---\nname: {name}\ndescription: A {name} skill.\n"
        f"dependencies_files: {deps}\n---\n# {name}\nBody.\n"
    ).encode()


_SKILL_FILES = {
    "skills/demo/SKILL.md": _skill_md("demo"),
    "skills/demo/reference/extra.md": b"support\n",
    "skills/other/SKILL.md": _skill_md("other"),
    "tools/tool_use_skill.py": _tool(deps=["services/service_skills.py"]),
    "services/service_skills.py": b"from plugins.BaseService import BaseService\nclass S(BaseService): pass\n",
}


def test_skill_paths_validate_and_plain_paths_still_reject():
    assert package_manager._validate_rel_path("skills/demo/SKILL.md") == "skills/demo/SKILL.md"
    assert package_manager._validate_rel_path("skills/demo/scripts/x.py")
    for bad in ("skills/SKILL.md", "helpers/x.md", "helpers/skills/demo/SKILL.md", "tools/tool_a.md", "skills/../evil/SKILL.md"):
        with pytest.raises(package_manager.PackageError):
            package_manager._validate_rel_path(bad)


def test_skill_frontmatter_declares_dependencies():
    meta = package_manager.read_dependency_meta(
        "skills/demo/SKILL.md",
        "---\nname: demo\ndescription: d\ndependencies_files: [tools/tool_use_skill.py]\ndependencies_pip: requests\n---\nBody",
    )
    assert meta.dependencies_files == ("tools/tool_use_skill.py",)
    assert meta.dependencies_pip == ("requests",)
    # Support files inside a skill folder carry no metadata and are never AST-parsed.
    meta = package_manager.read_dependency_meta("skills/demo/scripts/x.py", "this is ! not python")
    assert meta.dependencies_files == () and meta.dependencies_pip == ()


def test_skill_installs_whole_folder_plus_frontmatter_deps(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(dict(_SKILL_FILES)))

    result = package_manager.install_package(tmp_path, "demo", _Context(tmp_path))

    assert result.ok
    base = package_manager.INSTALLED_PLUGINS
    assert (base / "skills" / "demo" / "SKILL.md").exists()
    assert (base / "skills" / "demo" / "reference" / "extra.md").exists()
    assert (base / "tools" / "tool_use_skill.py").exists()
    assert (base / "services" / "service_skills.py").exists()
    # Listing shows the skill once, as one item.
    skills = [i for i in package_manager.installed_packages() if i["family"] == "skills"]
    assert [i["id"] for i in skills] == ["demo"]


def test_skill_uninstall_removes_folder_keeps_shared_machinery(tmp_path, monkeypatch):
    _patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(dict(_SKILL_FILES)))
    package_manager.install_package(tmp_path, "demo", _Context(tmp_path))
    package_manager.install_package(tmp_path, "other", _Context(tmp_path))

    result = package_manager.uninstall_package("demo", _Context(tmp_path))

    assert result.ok
    base = package_manager.INSTALLED_PLUGINS
    assert not (base / "skills" / "demo").exists()
    # The other installed skill still references the machinery via frontmatter.
    assert (base / "tools" / "tool_use_skill.py").exists()
    assert (base / "skills" / "other" / "SKILL.md").exists()

    package_manager.uninstall_package("other", _Context(tmp_path))
    assert not (base / "tools" / "tool_use_skill.py").exists()
