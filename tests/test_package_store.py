"""Tests for package store install/uninstall V1.

Package ids match the plugin/helper module they ship: ``bundle_*`` is a soft
collection (manifest only), a plugin-family prefix (``tool_``/``task_``/…) is a
plugin whose id equals its entrypoint stem, and an unprefixed id is a shared
helper package. Install writes one manifest record per package under
``packages/receipts``; uninstall is greedy and reference-counts members across
those records.
"""

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


def test_manifest_validation_enforces_id_prefix_contract():
    # A fileless meta-package must carry the bundle_ prefix.
    with pytest.raises(package_manager.PackageError, match="bundle_"):
        package_manager._validate_manifest({"id": "collection", "requires": ["x"], "files": []})
    # A bundle must not ship files.
    with pytest.raises(package_manager.PackageError, match="must not ship files"):
        package_manager._validate_manifest({"id": "bundle_x", "requires": [], "files": ["tools/tool_x.py"]})
    # A plugin-prefixed id must ship its matching entrypoint file.
    with pytest.raises(package_manager.PackageError, match="entrypoint file"):
        package_manager._validate_manifest({"id": "service_x", "requires": [], "files": ["services/service_y.py"]})
    # Hyphens are no longer valid ids.
    with pytest.raises(package_manager.PackageError):
        package_manager._validate_manifest({"id": "service-x", "requires": [], "files": ["services/service-x.py"]})


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
    # Family-grouped layout at the store root: <family>/<id>/, family in the index.
    (repo / "tools" / "tool_echo").mkdir(parents=True)
    (repo / "index.json").write_text(
        '{"packages":[{"id":"tool_echo","family":"tools"},{"id":"legacy_tool"}]}', encoding="utf-8")
    (repo / "tools" / "tool_echo" / "manifest.json").write_text('{"id":"tool_echo","files":[],"requires":[]}', encoding="utf-8")
    # A family-less index entry must still resolve via the flat (root) path.
    (repo / "legacy_tool").mkdir(parents=True)
    (repo / "legacy_tool" / "manifest.json").write_text('{"id":"legacy_tool","files":[],"requires":[]}', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "store"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "checkout", main_branch], cwd=repo, check=True, stdout=subprocess.PIPE)

    backend = GitStoreBackend(repo, ref="store")

    assert backend.get_index()[0]["id"] == "tool_echo"
    # Family entry resolves under tools/; family-less entry falls back to flat root.
    assert backend.get_manifest("tool_echo")["id"] == "tool_echo"
    assert backend.get_manifest("legacy_tool")["id"] == "legacy_tool"
    assert not (repo / "tools").exists()


def _echo_backend(extra_manifests=None, extra_files=None):
    """A ``tool_echo`` plugin package (entrypoint + private helper)."""
    manifests = {"tool_echo": {"id": "tool_echo", "name": "Echo", "description": "", "requires": [], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]}}
    files = {
        ("tool_echo", "tools/tool_echo.py"): _tool_source(),
        ("tool_echo", "tools/helpers/echo_format.py"): b"def fmt(value):\n    return value\n",
    }
    manifests.update(extra_manifests or {})
    files.update(extra_files or {})
    return _Backend(manifests, files)


def test_install_copies_loads_and_writes_record(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _echo_backend()
    registry = _ToolRegistry()
    context = _Context(tmp_path, registry)
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    monkeypatch.setattr(plugin_discovery, "load_single_plugin", lambda *a, **k: pytest.fail("package install should not register plugins"))

    result = package_manager.install_package(tmp_path, "tool_echo", context)

    assert result.ok
    assert registry.tools == {}
    record = package_manager.installed_packages()[0]
    assert record["id"] == "tool_echo"
    assert record["files"] == ["tools/tool_echo.py", "tools/helpers/echo_format.py"]
    assert record["entrypoints"][0]["path"] == "tools/tool_echo.py"
    assert record["entrypoints"][0]["type"] == "tool"
    assert "requested" not in record  # the explicit-install flag is gone


def test_install_pip_installs_missing_imports_in_current_python(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {"service_litellm": {"id": "service_litellm", "requires": [], "files": ["services/service_litellm.py"], "entrypoints": []}},
        {("service_litellm", "services/service_litellm.py"): b"import pathlib\nimport litellm\nfrom plugins.services.service_llm import BaseLLM\n"},
    )
    calls = []
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    monkeypatch.setattr(package_manager.importlib.util, "find_spec", lambda name: None if name == "litellm" else object())
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.install_package(tmp_path, "service_litellm", _Context(tmp_path, _ToolRegistry()))

    assert calls[0][0] == [sys.executable, "-m", "pip", "install", "litellm"]
    assert "Installed Python package(s): litellm" in result.lines
    assert package_manager.installed_packages()[0]["pip"] == ["litellm"]


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
    backend = _echo_backend(
        {
            "bundle_x": {"id": "bundle_x", "requires": ["missing_file", "tool_echo"], "files": []},
            "missing_file": {"id": "missing_file", "requires": [], "files": ["helpers/missing.txt"]},
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    with pytest.raises(package_manager.PackageError, match="missing file"):
        package_manager.install_package(tmp_path, "bundle_x", _Context(tmp_path, _ToolRegistry()))

    assert not (installed / "tools" / "tool_echo.py").exists()
    assert not package_manager.installed_packages()


def test_install_rejects_dependency_cycle(tmp_path, monkeypatch):
    """A requires-cycle is detected during planning — nothing is written."""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "cyc_a": {"id": "cyc_a", "requires": ["cyc_b"], "files": ["tools/tool_a.py"]},
            "cyc_b": {"id": "cyc_b", "requires": ["cyc_a"], "files": ["tools/tool_b.py"]},
        },
        {("cyc_a", "tools/tool_a.py"): _tool_source(), ("cyc_b", "tools/tool_b.py"): _tool_source()},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    with pytest.raises(package_manager.PackageError, match="cycle"):
        package_manager.install_package(tmp_path, "cyc_a", _Context(tmp_path, _ToolRegistry()))

    assert not package_manager.installed_packages()
    assert not (installed / "tools" / "tool_a.py").exists()
    assert not (installed / "tools" / "tool_b.py").exists()


def test_install_rejects_conflicting_file_across_manifests(tmp_path, monkeypatch):
    """Two packages in one install graph claiming the same file path with
    *different* content is refused during planning, before any file is written.
    (Identical content is allowed — that is a co-owned shared helper.)"""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle_dup": {"id": "bundle_dup", "requires": ["dup_x", "dup_y"], "files": []},
            "dup_x": {"id": "dup_x", "requires": [], "files": ["tools/tool_dup.py"]},
            "dup_y": {"id": "dup_y", "requires": [], "files": ["tools/tool_dup.py"]},
        },
        {("dup_x", "tools/tool_dup.py"): _tool_source('"x"'), ("dup_y", "tools/tool_dup.py"): _tool_source('"y"')},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    with pytest.raises(package_manager.PackageError, match="Conflicting content"):
        package_manager.install_package(tmp_path, "bundle_dup", _Context(tmp_path, _ToolRegistry()))

    assert not package_manager.installed_packages()
    assert not (installed / "tools" / "tool_dup.py").exists()


def test_uninstall_form_offers_all_installed_packages(tmp_path, monkeypatch):
    """Bundle membership is soft, so the uninstall picker offers every installed
    package — including a member pulled in by a bundle."""
    _patch_install_root(monkeypatch, tmp_path)
    backend = _echo_backend({"bundle_x": {"id": "bundle_x", "requires": ["tool_echo"], "files": []}})
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    package_manager.install_package(tmp_path, "bundle_x", _Context(tmp_path, _ToolRegistry()))

    assert sorted(p["id"] for p in package_manager.removable_packages()) == ["bundle_x", "tool_echo"]

    steps = PackagesCommand().form({"action": "uninstall"}, _Context(tmp_path, _ToolRegistry()))
    pkg_step = next(s for s in steps if s.name == "package_id")
    assert sorted(pkg_step.enum) == ["bundle_x", "tool_echo"]


def test_bundle_uninstall_greedily_removes_members(tmp_path, monkeypatch):
    """Greedy: uninstalling a bundle removes every member that loses its last
    referrer — there is no explicit-install flag to keep one behind, even one
    whose files were installed before the bundle."""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle_x": {"id": "bundle_x", "requires": ["tool_a"], "files": []},
            "tool_a": {"id": "tool_a", "requires": [], "files": ["tools/tool_a.py"]},
        },
        {("tool_a", "tools/tool_a.py"): _tool_source()},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    package_manager.install_package(tmp_path, "tool_a", _Context(tmp_path, _ToolRegistry()))
    package_manager.install_package(tmp_path, "bundle_x", _Context(tmp_path, _ToolRegistry()))

    package_manager.uninstall_package("bundle_x", _Context(tmp_path, _ToolRegistry()))

    assert not package_manager.installed_packages()
    assert not (installed / "tools" / "tool_a.py").exists()


def test_bundle_uninstall_keeps_member_held_by_another_bundle(tmp_path, monkeypatch):
    """A shared member survives a bundle uninstall when a second installed bundle
    still lists it — so the surviving bundle is never left missing a member."""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle_a": {"id": "bundle_a", "requires": ["tool_a"], "files": []},
            "bundle_b": {"id": "bundle_b", "requires": ["tool_a"], "files": []},
            "tool_a": {"id": "tool_a", "requires": [], "files": ["tools/tool_a.py"]},
        },
        {("tool_a", "tools/tool_a.py"): _tool_source()},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    package_manager.install_package(tmp_path, "bundle_a", _Context(tmp_path, _ToolRegistry()))
    package_manager.install_package(tmp_path, "bundle_b", _Context(tmp_path, _ToolRegistry()))

    package_manager.uninstall_package("bundle_a", _Context(tmp_path, _ToolRegistry()))

    assert {r["id"] for r in package_manager.installed_packages()} == {"bundle_b", "tool_a"}
    assert (installed / "tools" / "tool_a.py").exists()


def test_uninstall_tolerates_missing_member_and_clears_rest(tmp_path, monkeypatch):
    """If a member has gone missing out of band, uninstalling the bundle still
    clears the remaining members instead of crashing."""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle_x": {"id": "bundle_x", "requires": ["dep1", "dep2"], "files": []},
            "dep1": {"id": "dep1", "requires": [], "files": ["helpers/d1.txt"]},
            "dep2": {"id": "dep2", "requires": [], "files": ["tools/tool_d2.py"]},
        },
        {("dep1", "helpers/d1.txt"): b"d1", ("dep2", "tools/tool_d2.py"): _tool_source()},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    package_manager.install_package(tmp_path, "bundle_x", _Context(tmp_path, _ToolRegistry()))

    # dep1 vanishes out of band (record + file).
    (installed / "helpers" / "d1.txt").unlink()
    package_manager._receipt_path("dep1").unlink()

    result = package_manager.uninstall_package("bundle_x", _Context(tmp_path, _ToolRegistry()))

    assert result.ok
    assert not package_manager.installed_packages()
    assert not (installed / "tools" / "tool_d2.py").exists()


def test_reinstall_after_uninstall_restores_package_and_deps(tmp_path, monkeypatch):
    """Uninstalling a package then reinstalling it restores the package and its
    dependencies (files back on disk, records back)."""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "tool_a": {"id": "tool_a", "requires": ["base"], "files": ["tools/tool_a.py"]},
            "base": {"id": "base", "requires": [], "files": ["helpers/base.txt"]},
        },
        {("tool_a", "tools/tool_a.py"): _tool_source(), ("base", "helpers/base.txt"): b"base"},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    package_manager.install_package(tmp_path, "tool_a", _Context(tmp_path, _ToolRegistry()))
    package_manager.uninstall_package("tool_a", _Context(tmp_path, _ToolRegistry()))
    assert not package_manager.installed_packages()

    result = package_manager.install_package(tmp_path, "tool_a", _Context(tmp_path, _ToolRegistry()))

    assert result.ok
    assert {p["id"] for p in package_manager.installed_packages()} == {"tool_a", "base"}
    assert (installed / "tools" / "tool_a.py").exists()
    assert (installed / "helpers" / "base.txt").exists()


def test_uninstall_greedy_plan_collects_each_member_once(tmp_path, monkeypatch):
    """A shared dependency reachable through several requirers is collected once,
    not once per path to it (diamond: bundle→cmd→tool→svc with extra direct
    edges). All become orphaned ``pruned`` members of the bundle removal."""
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle_d": {"id": "bundle_d", "requires": ["cmd", "svc", "tool"], "files": []},
            "cmd": {"id": "cmd", "requires": ["svc", "tool"], "files": ["helpers/cmd.txt"]},
            "tool": {"id": "tool", "requires": ["svc"], "files": ["helpers/tool.txt"]},
            "svc": {"id": "svc", "requires": [], "files": ["helpers/svc.txt"]},
        },
        {
            ("cmd", "helpers/cmd.txt"): b"c",
            ("tool", "helpers/tool.txt"): b"t",
            ("svc", "helpers/svc.txt"): b"s",
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    package_manager.install_package(tmp_path, "bundle_d", _Context(tmp_path, _ToolRegistry()))

    plan = package_manager.build_uninstall_plan("bundle_d")
    ids = [pkg.id for pkg in plan.packages]

    assert len(ids) == len(set(ids)), f"duplicates in uninstall plan: {ids}"
    assert set(ids) == {"bundle_d", "cmd", "tool", "svc"}
    assert sorted(plan.pruned) == ["cmd", "svc", "tool"]
    assert plan.needs_confirm is False


def test_install_auto_installs_dependency(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _echo_backend({"base": {"id": "base", "requires": [], "files": ["helpers/base.txt"]}, "tool_echo": {"id": "tool_echo", "name": "Echo", "description": "", "requires": ["base"], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]}}, {("base", "helpers/base.txt"): b"base"})
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    package_manager.install_package(tmp_path, "tool_echo", _Context(tmp_path, _ToolRegistry()))
    records = {r["id"]: r for r in package_manager.installed_packages()}

    assert set(records) == {"tool_echo", "base"}
    assert records["tool_echo"]["requires"] == ["base"]


def test_bundle_install_reloads_parser_once_for_multiple_helpers(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "bundle_p": {"id": "bundle_p", "requires": ["parse_one", "parse_two"], "files": []},
            "parse_one": {"id": "parse_one", "requires": [], "files": ["services/helpers/parse_one.py"], "entrypoints": []},
            "parse_two": {"id": "parse_two", "requires": [], "files": ["services/helpers/parse_two.py"], "entrypoints": []},
        },
        {
            ("parse_one", "services/helpers/parse_one.py"): b"",
            ("parse_two", "services/helpers/parse_two.py"): b"",
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    parser = type("Parser", (), {"loaded": True, "loads": 0, "unloads": 0})()
    parser.load = lambda: setattr(parser, "loads", parser.loads + 1)
    parser.unload = lambda: setattr(parser, "unloads", parser.unloads + 1)

    context = _Context(tmp_path, _ToolRegistry())
    context.services = {"parser": parser}
    result = package_manager.install_package(tmp_path, "bundle_p", context)

    assert parser.loads == 1
    assert result.lines.count("Reloaded parser service; file parsers are now active.") == 1


def test_install_refuses_unowned_file_collision(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    target = installed / "tools" / "tool_echo.py"
    target.parent.mkdir(parents=True)
    target.write_text("mine", encoding="utf-8")
    backend = _Backend(
        {"tool_echo": {"id": "tool_echo", "requires": [], "files": ["tools/tool_echo.py", "helpers/new.txt"]}},
        {("tool_echo", "tools/tool_echo.py"): _tool_source(), ("tool_echo", "helpers/new.txt"): b"new"},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    with pytest.raises(package_manager.PackageError):
        package_manager.install_package(tmp_path, "tool_echo", _Context(tmp_path, _ToolRegistry()))
    assert not (installed / "helpers" / "new.txt").exists()


def test_uninstall_removes_files_record_and_prunes_orphaned_dependency(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _echo_backend({"base": {"id": "base", "requires": [], "files": ["helpers/base.txt"]}, "tool_echo": {"id": "tool_echo", "name": "Echo", "description": "", "requires": ["base"], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]}}, {("base", "helpers/base.txt"): b"base"})
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    registry = _ToolRegistry()
    context = _Context(tmp_path, registry)
    package_manager.install_package(tmp_path, "tool_echo", context)
    monkeypatch.setattr(plugin_discovery, "unload_plugin", lambda *a, **k: pytest.fail("package uninstall should not unload plugins"))

    result = package_manager.uninstall_package("tool_echo", context)

    assert result.ok
    assert not (installed / "tools" / "tool_echo.py").exists()
    assert not package_manager.installed_packages()
    assert registry.unloaded == []


def test_uninstall_hard_dependent_flags_confirm(tmp_path, monkeypatch):
    """Removing a helper a remaining plugin still requires is allowed, but the
    plan flags ``needs_confirm`` and the command cancels without a yes."""
    _patch_install_root(monkeypatch, tmp_path)
    backend = _echo_backend({"base": {"id": "base", "requires": [], "files": ["helpers/base.txt"]}, "tool_echo": {"id": "tool_echo", "name": "Echo", "description": "", "requires": ["base"], "files": ["tools/tool_echo.py", "tools/helpers/echo_format.py"]}}, {("base", "helpers/base.txt"): b"base"})
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    context = _Context(tmp_path, _ToolRegistry())
    package_manager.install_package(tmp_path, "tool_echo", context)

    plan = package_manager.build_uninstall_plan("base")
    assert plan.needs_confirm is True
    assert plan.broken_dependents == ["tool_echo"]

    result = PackagesCommand().run({"action": "uninstall", "package_id": "base"}, context)
    assert result == "Uninstall cancelled."
    assert {r["id"] for r in package_manager.installed_packages()} == {"tool_echo", "base"}


def test_uninstall_raw_file_deletes_private_helper(tmp_path, monkeypatch):
    """A name that matches no package but does match an installed file deletes
    that file directly; the owning record is left listing the now-missing file
    and gates a confirm."""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _echo_backend()
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    context = _Context(tmp_path, _ToolRegistry())
    package_manager.install_package(tmp_path, "tool_echo", context)

    plan = package_manager.build_uninstall_plan("echo_format")  # the private helper's stem
    assert plan.raw_files == ["tools/helpers/echo_format.py"]
    assert plan.needs_confirm is True
    assert plan.broken_dependents == ["tool_echo"]

    package_manager.execute_uninstall_plan(plan, context)
    assert not (installed / "tools" / "helpers" / "echo_format.py").exists()
    assert {r["id"] for r in package_manager.installed_packages()} == {"tool_echo"}


def test_uninstall_unknown_target_errors(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    with pytest.raises(package_manager.PackageError, match="not installed"):
        package_manager.build_uninstall_plan("nope")


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
            "bundle_starter": {"id": "bundle_starter", "requires": ["task_owned"], "files": []},
            "task_owned": {"id": "task_owned", "requires": [], "files": ["tasks/task_owned.py"], "entrypoints": []},
        },
        {("task_owned", "tasks/task_owned.py"): b"config_settings = [('Owned', 'owned_key', '', 'x', {})]\n"},
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    context = _Context(tmp_path, _ToolRegistry())
    package_manager.install_package(tmp_path, "bundle_starter", context)

    command = PackagesCommand()
    steps = command.form({"action": "uninstall", "package_id": "bundle_starter"}, context)

    assert [step.name for step in steps] == ["action", "package_id", "cleanup_config"]
    assert steps[-1].enum == ["all", "none", "specific"]
    assert steps[-1].enum_labels == ["All", "None", "Specific"]
    assert steps[-1].default == "all"
    plan = package_manager.build_uninstall_plan("bundle_starter")
    assert _cleanup_choices(plan, {})["config"] == {"bundle_starter": False, "task_owned": True}
    assert _cleanup_choices(plan, {"cleanup_config": "none"})["config"] == {"bundle_starter": False, "task_owned": False}
    assert _cleanup_choices(plan, {"cleanup_config": "specific", "cleanup_config__task_owned": True})["config"] == {"bundle_starter": False, "task_owned": True}
    steps = command.form({"action": "uninstall", "package_id": "bundle_starter", "cleanup_config": "specific"}, context)
    assert [step.name for step in steps] == ["action", "package_id", "cleanup_config", "cleanup_config__task_owned"]
    assert "Config settings: owned_key" in steps[-1].prompt
    context.request_user_input = lambda *a, **k: pytest.fail("cleanup should be collected by the command form")
    result = command.run({"action": "uninstall", "package_id": "bundle_starter", "cleanup_config": "none"}, context)

    assert "Kept package config setting(s)." in result
    assert not package_manager.installed_packages()


def test_uninstall_pip_cleanup_only_removes_safe_candidates(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    package_manager._write_receipt({"id": "pkg_a", "requires": [], "files": [], "entrypoints": [], "pip": ["orphan-lib", "litellm", "shared-lib"]})
    package_manager._write_receipt({"id": "pkg_b", "requires": [], "files": [], "entrypoints": [], "pip": ["shared-lib"]})
    calls = []
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    plan = package_manager.build_uninstall_plan("pkg_a")
    result = package_manager.execute_uninstall_plan(plan, _Context(tmp_path, _ToolRegistry()), {"pip": {"pkg_a": True}})

    assert plan.pip_removals == {"pkg_a": ["litellm", "orphan-lib"]}
    assert calls == [[sys.executable, "-m", "pip", "uninstall", "-y", "litellm", "orphan-lib"]]
    assert "Kept Python package(s): shared-lib (needed by another installed package)" in result.lines


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


def _coowner_backend(shared=b"SHARED = 1\n"):
    """Two plugin packages that each ship the same shared plugin-level helper."""
    return _Backend(
        {
            "tool_x": {"id": "tool_x", "requires": [], "files": ["tools/tool_x.py", "tools/helpers/shared.py"]},
            "tool_y": {"id": "tool_y", "requires": [], "files": ["tools/tool_y.py", "tools/helpers/shared.py"]},
        },
        {
            ("tool_x", "tools/tool_x.py"): b"# x\n",
            ("tool_x", "tools/helpers/shared.py"): shared,
            ("tool_y", "tools/tool_y.py"): b"# y\n",
            ("tool_y", "tools/helpers/shared.py"): shared,
        },
    )


def test_co_owned_identical_helper_installs_once_and_refcounts(tmp_path, monkeypatch):
    """A shared plugin-level helper shipped by two packages installs once; the
    file survives until its last owning package is uninstalled."""
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _coowner_backend())
    package_manager.install_package(tmp_path, "tool_x", _Context(tmp_path, _ToolRegistry()))
    package_manager.install_package(tmp_path, "tool_y", _Context(tmp_path, _ToolRegistry()))
    shared = installed / "tools" / "helpers" / "shared.py"
    assert shared.exists()

    package_manager.uninstall_package("tool_x", _Context(tmp_path, _ToolRegistry()))
    assert shared.exists()  # tool_y still co-owns it
    assert {r["id"] for r in package_manager.installed_packages()} == {"tool_y"}

    package_manager.uninstall_package("tool_y", _Context(tmp_path, _ToolRegistry()))
    assert not shared.exists()
    assert not package_manager.installed_packages()


def test_bundle_co_owns_shared_helper_in_one_graph(tmp_path, monkeypatch):
    installed, _receipts = _patch_install_root(monkeypatch, tmp_path)
    backend = _coowner_backend()
    backend.manifests["bundle_xy"] = {"id": "bundle_xy", "requires": ["tool_x", "tool_y"], "files": []}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)

    package_manager.install_package(tmp_path, "bundle_xy", _Context(tmp_path, _ToolRegistry()))

    assert (installed / "tools" / "helpers" / "shared.py").exists()
    assert {r["id"] for r in package_manager.installed_packages()} == {"bundle_xy", "tool_x", "tool_y"}


def test_conflicting_shared_file_content_is_refused(tmp_path, monkeypatch):
    _patch_install_root(monkeypatch, tmp_path)
    backend = _Backend(
        {
            "tool_x": {"id": "tool_x", "requires": [], "files": ["tools/tool_x.py", "tools/helpers/shared.py"]},
            "tool_y": {"id": "tool_y", "requires": [], "files": ["tools/tool_y.py", "tools/helpers/shared.py"]},
        },
        {
            ("tool_x", "tools/tool_x.py"): b"# x\n", ("tool_x", "tools/helpers/shared.py"): b"A = 1\n",
            ("tool_y", "tools/tool_y.py"): b"# y\n", ("tool_y", "tools/helpers/shared.py"): b"B = 2\n",
        },
    )
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: backend)
    package_manager.install_package(tmp_path, "tool_x", _Context(tmp_path, _ToolRegistry()))

    with pytest.raises(package_manager.PackageError, match="different content"):
        package_manager.install_package(tmp_path, "tool_y", _Context(tmp_path, _ToolRegistry()))


def test_packages_install_missing_package_hides_git_manifest_error(tmp_path, monkeypatch):
    def missing(*_args, **_kwargs):
        raise StoreBackendError("Could not read origin/store:packages/hfs/manifest.json: fatal: path 'packages/hfs/manifest.json' does not exist in 'origin/store'")

    monkeypatch.setattr(package_manager, "install_package", missing)

    result = PackagesCommand().run({"action": "install", "package_id": "hfs"}, _Context(tmp_path, _ToolRegistry()))

    assert result == "Package install failed: 'hfs' not found."
