"""Tests for tree-based package-store install/uninstall."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import pytest

from bundled.commands.helpers import package_manager
from plugins import plugin_paths


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

    def refresh(self, force=False):
        """``outdated_packages`` pulls before it compares; nothing to pull here."""


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
        "from guest.bases import BaseTool\n"
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


def _hooked(method: str, name: str = "t", body: str = "return 1"):
    """A tool declaring one lifecycle hook, so the AST pass has something to find."""
    return (
        "from guest.bases import BaseTool\n"
        "class T(BaseTool):\n"
        f"    name = {name!r}\n"
        f"    def {method}(self, sdk):\n"
        f"        {body}\n"
    ).encode()


class _RecordingSandbox:
    """Stands in for the real box runner and remembers what it was asked to do.

    The package manager's job here is *orchestration* — which files, which
    method, in what order, and what happens when one fails. Running a real box
    would test the facade instead, and would do it against a temp directory in
    no known tree, where isolation deliberately fails closed. The facade's own
    tests cover the other half.
    """

    def __init__(self, ok=True, error=""):
        self.calls = []
        self.ok = ok
        self.error = error

    def run(self, source, entry="", **kwargs):
        from sandbox.guest.requests import Result

        self.calls.append({"source": Path(source), "entry": entry,
                           "exists": Path(source).exists(), **kwargs})
        return Result(ok=self.ok, error=self.error)


def _capture_sandbox(monkeypatch, **kwargs) -> _RecordingSandbox:
    recorder = _RecordingSandbox(**kwargs)
    opened = []
    monkeypatch.setattr("sandbox.bridge.get_sandbox",
                        lambda: opened.append(1) or recorder)
    recorder.opened = opened
    return recorder


def test_install_runs_on_install_under_the_installers_identity(tmp_path, monkeypatch):
    """The hook fires, and it fires named as the registry knows the plugin.

    ``name`` becomes the run's chain link, and the chain link is what
    ``policy._owns_setting`` matches against the setting registry. The box
    would otherwise be called after the *file* (``tool_setup``), so a plugin
    touching a setting it declared itself would read as a stranger to its own
    bookkeeping — the same mismatch ``PersistentBox._identity`` fixes for
    resident boxes.
    """
    _patch_roots(monkeypatch, tmp_path)
    files = {"tools/tool_setup.py": _hooked("on_install", name="setup")}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    recorder = _capture_sandbox(monkeypatch)

    result = package_manager.install_package(tmp_path, "tool_setup", _Context(tmp_path))

    assert result.ok
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["entry"] == "T"
    assert call["method"] == "on_install"
    assert call["name"] == "setup"
    assert call["once"] is True
    assert "Set up: tool_setup" in result.lines


def test_a_file_declaring_no_hook_never_opens_a_box(tmp_path, monkeypatch):
    """Every plugin *inherits* both no-ops, so asking the class would say yes.

    Detection is therefore "does this class define one", by AST, and the cost
    of a package that wants nothing is one parse per file rather than one
    subprocess per file.
    """
    _patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(package_manager, "GitStoreBackend",
                        lambda _root: _Backend({"tools/tool_plain.py": _tool()}))
    recorder = _capture_sandbox(monkeypatch)

    package_manager.install_package(tmp_path, "tool_plain", _Context(tmp_path))

    assert recorder.calls == []
    assert recorder.opened == [], "no sandbox was even asked for"


def test_on_install_reruns_only_when_the_bytes_changed(tmp_path, monkeypatch):
    """Install, update, and nothing else — decided by one existing condition.

    The copy loop already tells "Already installed" (byte-identical) from
    "Updated file", so the firing policy costs no separate bookkeeping. That
    matters because the alternative — a marker file, or a record of what has
    been set up — is state that can disagree with the tree.
    """
    _patch_roots(monkeypatch, tmp_path)
    files = {"tools/tool_setup.py": _hooked("on_install")}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    recorder = _capture_sandbox(monkeypatch)

    package_manager.install_package(tmp_path, "tool_setup", _Context(tmp_path))
    assert len(recorder.calls) == 1

    package_manager.install_package(tmp_path, "tool_setup", _Context(tmp_path))
    assert len(recorder.calls) == 1, "byte-identical: nothing to do"

    files["tools/tool_setup.py"] = _hooked("on_install", body="return 2")
    package_manager.install_package(tmp_path, "tool_setup", _Context(tmp_path))
    assert len(recorder.calls) == 2, "a new version may need new setup"


def test_update_runs_on_install_for_what_it_rewrote(tmp_path, monkeypatch):
    """``/packages update`` is an install of the changed files, hooks included.

    It routes through ``execute_install_plan``, so this holds for free — but
    "for free" is exactly the kind of claim that stops being true silently, and
    the symptom is a plugin whose new version never gets set up while the
    command reports success.
    """
    _patch_roots(monkeypatch, tmp_path)
    files = {"tools/tool_setup.py": _hooked("on_install")}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    recorder = _capture_sandbox(monkeypatch)
    package_manager.install_package(tmp_path, "tool_setup", _Context(tmp_path))

    assert package_manager.update_packages(tmp_path, _Context(tmp_path)).ok
    assert len(recorder.calls) == 1, "nothing was outdated"

    files["tools/tool_setup.py"] = _hooked("on_install", body="return 2")
    result = package_manager.update_packages(tmp_path, _Context(tmp_path))

    assert len(recorder.calls) == 2
    assert recorder.calls[1]["method"] == "on_install"
    assert "Set up: tool_setup" in result.lines


def test_a_failing_on_install_still_leaves_the_package_installed(tmp_path, monkeypatch):
    """A declined dialog is a failed Request, and the files are already down.

    Rolling back a whole package because its setup step was refused would
    punish the safe answer. The line names it instead, so the user can finish
    the job by hand.
    """
    _patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        package_manager, "GitStoreBackend",
        lambda _root: _Backend({"tools/tool_setup.py": _hooked("on_install")}))
    _capture_sandbox(monkeypatch, ok=False, error="denied: config.write")

    result = package_manager.install_package(tmp_path, "tool_setup", _Context(tmp_path))

    assert result.ok
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "tool_setup.py").exists()
    assert any("on_install failed for tool_setup" in line for line in result.lines)


def test_on_uninstall_runs_while_the_file_is_still_there(tmp_path, monkeypatch):
    """First step of the uninstall, and the ordering is the whole contract.

    A hook cannot be loaded from a file that has been unlinked, cannot import a
    pip package that has been removed, and cannot look itself up in a registry
    it has been dropped from. So it runs before any of that happens.
    """
    _patch_roots(monkeypatch, tmp_path)
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_setup.py",
           _hooked("on_uninstall", name="setup").decode())
    recorder = _capture_sandbox(monkeypatch)

    result = package_manager.uninstall_package("tool_setup", _Context(tmp_path))

    assert result.ok
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["method"] == "on_uninstall"
    assert recorder.calls[0]["exists"], "the file was still on disk"
    assert not (package_manager.INSTALLED_PLUGINS / "tools" / "tool_setup.py").exists()
    assert "Cleaned up: tool_setup" in result.lines


def _dependency_triangle(monkeypatch):
    """``tool_a`` and ``tool_b`` both need ``tool_shared``, which needs nobody."""
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_a.py",
           _tool(deps=["tools/tool_shared.py"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_b.py",
           _tool(deps=["tools/tool_shared.py"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_shared.py",
           _hooked("on_uninstall", name="shared").decode())
    return _capture_sandbox(monkeypatch)


def test_uninstalling_a_dependent_leaves_what_it_depended_on(tmp_path, monkeypatch):
    """A dependency is a claim about what I need, never a claim of ownership.

    ``tool_shared`` works exactly as well without ``tool_a``, and nothing in
    the tree records whether it was installed for its own sake or came along
    for the ride — so the forward edge cannot answer the question and is not
    followed. Its teardown must not fire either: that would have one package
    drop another's tables on its way out, surfacing much later in the
    surviving plugin.
    """
    _patch_roots(monkeypatch, tmp_path)
    recorder = _dependency_triangle(monkeypatch)

    result = package_manager.uninstall_package("tool_a", _Context(tmp_path))

    assert recorder.calls == []
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "tool_shared.py").exists()
    assert (package_manager.INSTALLED_PLUGINS / "tools" / "tool_b.py").exists()
    assert result.lines == ["Removed file: tools/tool_a.py"]


def test_uninstalling_a_dependency_takes_everything_that_needed_it(tmp_path, monkeypatch):
    """The other direction, and the one that is actually decidable.

    ``tool_a`` and ``tool_b`` cannot run without ``tool_shared``, so leaving
    them installed leaves two registered plugins failing every call — the
    quietest possible breakage. Their own teardowns run, since they really are
    being uninstalled.
    """
    _patch_roots(monkeypatch, tmp_path)
    recorder = _dependency_triangle(monkeypatch)

    result = package_manager.uninstall_package("tool_shared", _Context(tmp_path))

    for stem in ("tool_a", "tool_b", "tool_shared"):
        assert not (package_manager.INSTALLED_PLUGINS / "tools" / f"{stem}.py").exists()
    assert [call["method"] for call in recorder.calls] == ["on_uninstall"]
    assert "Cleaned up: tool_shared" in result.lines


def test_dependents_are_followed_transitively(tmp_path, monkeypatch):
    """A dependent's dependents are broken just as thoroughly.

    This is the user's own chain: uninstall ``lexical_search`` and
    ``hybrid_search`` cannot run, which means ``memory_retrieve`` cannot
    either.
    """
    _patch_roots(monkeypatch, tmp_path)
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_lexical.py", _tool().decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_hybrid.py",
           _tool(deps=["tools/tool_lexical.py"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_memory.py",
           _tool(deps=["tools/tool_hybrid.py"]).decode())

    plan = package_manager.build_uninstall_plan("tool_lexical")

    assert set(plan.remove_files) == {"tools/tool_lexical.py", "tools/tool_hybrid.py",
                                      "tools/tool_memory.py"}


def test_a_failing_on_uninstall_still_removes_the_package(tmp_path, monkeypatch):
    """The user asked for it gone. A cleanup that cannot run must not veto that."""
    _patch_roots(monkeypatch, tmp_path)
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_setup.py",
           _hooked("on_uninstall").decode())
    _capture_sandbox(monkeypatch, ok=False, error="boom")

    result = package_manager.uninstall_package("tool_setup", _Context(tmp_path))

    assert result.ok
    assert not (package_manager.INSTALLED_PLUGINS / "tools" / "tool_setup.py").exists()
    assert any("on_uninstall failed for tool_setup" in line for line in result.lines)


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
            "from guest.bases import BaseFrontend\n"
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
    files = {"frontends/frontend_telegram.py": b"from guest.bases import BaseFrontend\nclass Telegram(BaseFrontend): pass\n"}
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
    files = {"services/service_mcp.py": b"from guest.bases import BaseService\nclass MCP(BaseService): pass\n"}
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
    files = {"services/service_mcp.py": b"from guest.bases import BaseService\nclass MCP(BaseService): pass\n"}
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


def test_autoload_records_the_name_the_service_registers_under(tmp_path, monkeypatch):
    """The filename is a guess at the registry key, and often a wrong one.

    ``services/service_drive.py`` registers as ``google_drive``, so installing
    it enabled a service that does not exist and left the real one off. Every
    boot then warned "unknown service 'drive', skipping" over a config line
    only a reinstall would ever rewrite.
    """
    _patch_roots(monkeypatch, tmp_path)
    files = {"services/service_drive.py": (
        b"from guest.bases import BaseService\n"
        b"class GoogleDriveService(BaseService):\n"
        b"    name = 'google_drive'\n")}
    saved = {"enabled_frontends": ["repl"], "autoload_services": []}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    assert package_manager.install_package(tmp_path, "service_drive", _Context(tmp_path)).ok
    assert saved["autoload_services"] == ["google_drive"]


def test_a_file_holding_two_services_enables_both(tmp_path, monkeypatch):
    """One file, two services — which a filename cannot express at all, so
    ``service_embed.py`` enabled neither embedder."""
    _patch_roots(monkeypatch, tmp_path)
    files = {"services/service_embed.py": (
        b"from guest.bases import BaseService\n"
        b"class TextEmbedder(BaseService):\n"
        b"    name = 'text_embedder'\n"
        b"class ImageEmbedder(BaseService):\n"
        b"    name = 'image_embedder'\n")}
    saved = {"enabled_frontends": ["repl"], "autoload_services": []}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    assert package_manager.install_package(tmp_path, "service_embed", _Context(tmp_path)).ok
    assert saved["autoload_services"] == ["text_embedder", "image_embedder"]


def test_uninstall_disables_the_same_names_install_enabled(tmp_path, monkeypatch):
    """Read off the installed file on the way out, so the two halves agree —
    otherwise the entry install wrote could never be taken back."""
    _patch_roots(monkeypatch, tmp_path)
    files = {"services/service_drive.py": (
        b"from guest.bases import BaseService\n"
        b"class GoogleDriveService(BaseService):\n"
        b"    name = 'google_drive'\n")}
    saved = {"enabled_frontends": ["repl"], "autoload_services": []}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))
    package_manager.install_package(tmp_path, "service_drive", _Context(tmp_path))

    assert package_manager.uninstall_package("service_drive", _Context(tmp_path)).ok
    assert saved["autoload_services"] == []


def test_an_unmigrated_service_still_falls_back_to_its_filename(tmp_path, monkeypatch):
    """A plugin that declares no name has nothing better to offer."""
    _patch_roots(monkeypatch, tmp_path)
    files = {"services/service_legacy.py": (
        b"from guest.bases import BaseService\n"
        b"class Legacy(BaseService): pass\n")}
    saved = {"enabled_frontends": ["repl"], "autoload_services": []}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr("config.config_manager.load", lambda: dict(saved))
    monkeypatch.setattr("config.config_manager.save", lambda config: saved.update(config))

    assert package_manager.install_package(tmp_path, "service_legacy", _Context(tmp_path)).ok
    assert saved["autoload_services"] == ["legacy"]


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


def test_uninstall_keeps_pip_another_installed_plugin_still_declares(tmp_path, monkeypatch):
    """Files are decided by the dependency graph; pip is decided by declaration.

    A library is shared by whoever names it, with no edge between them, so this
    is the one keep-list the reverse closure does not answer on its own.
    """
    _patch_roots(monkeypatch, tmp_path)
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_a.py", _tool(pip=["shared-lib", "only-a-lib"]).decode())
    _write(package_manager.INSTALLED_PLUGINS, "tools/tool_b.py", _tool(pip=["shared-lib"]).decode())
    calls = []
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.uninstall_package("tool_a", _Context(tmp_path))

    assert result.ok
    assert not (package_manager.INSTALLED_PLUGINS / "tools" / "tool_a.py").exists()
    assert calls == [[sys.executable, "-m", "pip", "uninstall", "-y", "only-a-lib"]]
    assert "Kept Python package(s): shared-lib" in "\n".join(result.lines)


def test_uninstall_keeps_pip_declared_by_a_builtin_or_workspace_file(tmp_path, monkeypatch):
    """The other two trees count too — they are not the store's to prune."""
    built_in, sandbox, installed = _patch_roots(monkeypatch, tmp_path)
    _write(installed, "tools/tool_a.py", _tool(pip=["builtin-lib", "sandbox-lib"]).decode())
    _write(built_in, "tools/tool_builtin.py", _tool(pip=["builtin-lib"]).decode())
    _write(sandbox, "tools/tool_sandbox.py", _tool(pip=["sandbox-lib"]).decode())
    calls = []
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    result = package_manager.uninstall_package("tool_a", _Context(tmp_path))

    assert calls == []
    kept = "\n".join(result.lines)
    assert "builtin-lib" in kept and "sandbox-lib" in kept


def test_parser_helper_install_and_uninstall_rescan_parsers(tmp_path, monkeypatch):
    """Installing a parser helper makes it live without a restart.

    Parsing is kernel routing rather than a service, so this is a rescan and
    not a load/unload cycle — there is nothing holding state to tear down.
    """
    import parsing

    _patch_roots(monkeypatch, tmp_path)
    scans = []
    monkeypatch.setattr(parsing, "discover", lambda: scans.append(1) or len(scans))

    context = _Context(tmp_path)
    files = {"parsers/parse_pdf.py": _helper()}
    monkeypatch.setattr(package_manager, "GitStoreBackend", lambda _root: _Backend(files))
    monkeypatch.setattr(package_manager.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "", ""))

    package_manager.install_package(tmp_path, "parse_pdf", context)
    package_manager.uninstall_package("parse_pdf", context)

    assert len(scans) == 2, "both install and uninstall must rescan"


# ──────────────────────────────────────────────────────────────────────
# Tree roots, and what is no longer one
# ──────────────────────────────────────────────────────────────────────

def test_skills_are_not_a_store_family_any_more():
    """Skills are gone, and the package layer must not quietly still accept one.

    Pinned as a negative rather than simply deleted: the folder-as-package
    handling was the only thing in the store that was not a single file, and a
    path that validates but installs nothing would fail silently.
    """
    assert "skills" not in package_manager.EXTRA_FAMILIES
    for bad in ("skills/demo/SKILL.md", "skills/demo/scripts/x.py"):
        with pytest.raises(package_manager.PackageError):
            package_manager._validate_rel_path(bad)


def test_scripts_are_a_tree_root_and_only_at_the_top():
    """The store ships a script the way it ships a helper: the file is the package.

    Top level only, and it has to stay that way — ``isolation.is_script``
    decides whether an installed script may run at all, and it recognises
    ``scripts/<name>.py`` and nothing deeper. A package manager that accepted a
    shape the isolation rule refuses would install files that can never run.
    """
    assert package_manager._validate_rel_path("scripts/backfill.py")
    assert "scripts" in package_manager.TREE_ROOTS
    for bad in ("scripts/helpers/x.py", "scripts/sub/x.py", "scripts/x.md"):
        with pytest.raises(package_manager.PackageError):
            package_manager._validate_rel_path(bad)


# ────────────────────────────────────────────────────────────────────
# The publisher dev tool (was test_package_store_publisher.py)
# ────────────────────────────────────────────────────────────────────

from dev import package_publisher


def test_write_package_copies_file_and_dir_and_validates(tmp_path):
    tool = tmp_path / "tool_echo.py"
    helper_dir = tmp_path / "helpers"
    helper = helper_dir / "echo_format.py"
    tool.write_text("print('tool')\n", encoding="utf-8")
    helper_dir.mkdir()
    helper.write_text("def fmt(value): return value\n", encoding="utf-8")

    written = package_publisher.write_package(
        tmp_path / "store",
        package_id="tool_echo",
        file_specs=[f"{tool}=tools/tool_echo.py", f"{helper_dir}=tools/helpers"],
        requires=["tools/helpers/echo_format.py"],
        pip=["echo-lib"],
        update=False,
    )

    assert written == ["tools/tool_echo.py", "tools/helpers/echo_format.py"]
    assert (tmp_path / "store" / "tools" / "helpers" / "echo_format.py").exists()
    meta = package_manager.read_dependency_meta("tools/tool_echo.py", (tmp_path / "store" / "tools" / "tool_echo.py").read_text())
    assert meta.dependencies_files == ("tools/helpers/echo_format.py",)
    assert meta.dependencies_pip == ("echo-lib",)
    package_publisher.validate_store(tmp_path / "store")


def test_write_package_refuses_existing_package_without_update(tmp_path):
    source = tmp_path / "tool_echo.py"
    source.write_text("print('tool')\n", encoding="utf-8")
    kwargs = dict(
        store_root=tmp_path / "store",
        package_id="tool_echo",
        file_specs=[f"{source}=tools/tool_echo.py"],
        requires=[],
        pip=None,
    )

    package_publisher.write_package(**kwargs, update=False)
    source.write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(package_publisher.StorePublishError):
        package_publisher.write_package(**kwargs, update=False)


def test_validate_store_rejects_missing_dependency(tmp_path):
    path = tmp_path / "store" / "tools" / "tool_echo.py"
    path.parent.mkdir(parents=True)
    path.write_text("dependencies_files = ['tools/helpers/missing.py']\n", encoding="utf-8")

    with pytest.raises(package_publisher.StorePublishError):
        package_publisher.validate_store(tmp_path / "store")


def test_validate_store_checks_bundle_files(tmp_path):
    tool = tmp_path / "store" / "tools" / "tool_echo.py"
    bundle = tmp_path / "store" / "bundles" / "bundle_starter.json"
    tool.parent.mkdir(parents=True)
    bundle.parent.mkdir(parents=True)
    tool.write_text("VALUE = 1\n", encoding="utf-8")
    bundle.write_text('{"name": "Starter", "files": ["tools/tool_echo.py"]}\n', encoding="utf-8")

    package_publisher.validate_store(tmp_path / "store")

    bundle.write_text('{"files": ["tools/tool_missing.py"]}\n', encoding="utf-8")
    with pytest.raises(package_publisher.StorePublishError):
        package_publisher.validate_store(tmp_path / "store")


def test_dependency_metadata_is_written_after_future_import(tmp_path):
    source = tmp_path / "tool_future.py"
    source.write_text('"""Doc."""\n\nfrom __future__ import annotations\n\nVALUE = 1\n', encoding="utf-8")

    package_publisher.write_package(
        tmp_path / "store",
        package_id="tool_future",
        file_specs=[f"{source}=tools/tool_future.py"],
        requires=[],
        pip=["future-lib"],
        update=False,
    )

    text = (tmp_path / "store" / "tools" / "tool_future.py").read_text(encoding="utf-8")
    assert text.index("from __future__ import annotations") < text.index("dependencies_pip")
    compile(text, "tool_future.py", "exec")
