"""Tree-based package-store install/uninstall operations."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from paths import INSTALLED_PLUGINS, ROOT_DIR
from plugins.commands.helpers.store_backend import GitStoreBackend
from plugins.helpers.plugin_paths import PLUGIN_FAMILIES, PLUGIN_ROOTS


TREE_ROOTS = {family for family, _prefix in PLUGIN_FAMILIES.values()}
DEPENDENCY_FIELDS = ("dependencies_files", "dependencies_pip")
_PACKAGE_LOCK = threading.RLock()
Progress = Callable[[str], None]


class PackageError(RuntimeError):
    """Raised for package-store validation or execution failures."""


@dataclass
class PackageActionResult:
    ok: bool
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines) if self.lines else ("OK" if self.ok else "Failed")


@dataclass(frozen=True)
class DependencyMeta:
    path: str
    dependencies_files: tuple[str, ...] = ()
    dependencies_pip: tuple[str, ...] = ()


@dataclass
class PlannedFile:
    path: str
    content: bytes | None = None


@dataclass
class InstallPlan:
    target: str
    files: list[PlannedFile]
    pip_packages: list[str]
    existing_files: list[str]
    parser_reload_needed: bool
    progress_steps: list[str]


@dataclass
class UninstallPlan:
    target: str
    remove_files: list[str]
    keep_files: dict[str, str]
    pip_packages: list[str]
    kept_pip_packages: dict[str, str]
    parser_reload_needed: bool
    progress_steps: list[str]


def search_packages(root_dir: str | Path, query: str = "") -> list[dict]:
    """Return available store files matching a stem/name query."""
    q = (query or "").strip().lower()
    items = [_item(rel, installed=False) for rel in GitStoreBackend(root_dir).list_python_files()] + search_bundles(root_dir)
    if q:
        items = [item for item in items if q in item["id"].lower() or q in item["path"].lower()]
    return sorted(items, key=lambda item: (item["family"], item["id"], item["path"]))


def search_bundles(root_dir: str | Path) -> list[dict]:
    """Return cloud-only bundle manifests."""
    out = []
    for rel in _bundle_manifest_files(GitStoreBackend(root_dir)):
        manifest = _read_bundle_manifest(GitStoreBackend(root_dir), rel)
        out.append({"id": PurePosixPath(rel).stem, "name": manifest.get("name") or PurePosixPath(rel).stem, "path": rel, "family": "bundles", "helper": False, "installed": False})
    return out


def installed_packages() -> list[dict]:
    """Return installed plugin/helper files; the tree is the source of truth."""
    return sorted((_item(rel, installed=True) for rel in _installed_rel_files()), key=lambda item: (item["family"], item["id"], item["path"]))


def removable_packages() -> list[dict]:
    return installed_packages()


def package_info(root_dir: str | Path, target: str) -> dict:
    rel = _resolve_store_target(root_dir, target)
    meta = _meta_from_bytes(rel, GitStoreBackend(root_dir).get_tree_file_bytes(rel))
    return {**_item(rel, installed=False), "dependencies_files": list(meta.dependencies_files), "dependencies_pip": list(meta.dependencies_pip)}


def install_package(root_dir: str | Path, target: str, context=None, *, requested: bool = True, progress: Progress | None = None) -> PackageActionResult:
    return execute_install_plan(build_install_plan(root_dir, target), context, progress=progress)


def uninstall_package(target: str, context=None, cleanup_choices=None, progress: Progress | None = None, cleanup_approvals=None, root_dir: str | Path | None = None) -> PackageActionResult:
    return execute_uninstall_plan(build_uninstall_plan(target, root_dir=root_dir), context, progress=progress)


def build_install_plan(root_dir: str | Path, target: str, *, requested: bool = True) -> InstallPlan:
    """Resolve target + recursive file deps from origin/store."""
    store = GitStoreBackend(root_dir)
    bundle = _resolve_bundle_target(store, target)
    if bundle:
        manifest = _read_bundle_manifest(store, bundle)
        return _install_plan_from_roots(store, manifest["files"], bundle)
    return _install_plan_from_roots(store, [_resolve_store_target(root_dir, target)], _target_stem(target))


def _install_plan_from_roots(store: GitStoreBackend, roots: list[str], target: str) -> InstallPlan:
    active: list[str] = []
    collected: dict[str, PlannedFile] = {}
    pip: list[str] = []

    def visit(rel: str):
        rel = _validate_rel_path(rel)
        if rel in active:
            raise PackageError(f"Dependency cycle includes {rel}.")
        if rel in collected:
            return
        active.append(rel)
        try:
            content = store.get_tree_file_bytes(rel)
            meta = _meta_from_bytes(rel, content)
            collected[rel] = PlannedFile(rel, content)
            pip.extend(meta.dependencies_pip)
            for dep in meta.dependencies_files:
                visit(dep)
        finally:
            active.pop()

    for root in roots:
        visit(root)
    existing = [rel for rel in collected if _target(rel).exists()]
    pip_packages = _unique(pip)
    steps = ["Resolving dependency plan"]
    if pip_packages:
        steps.append(f"Installing Python package(s): {', '.join(pip_packages)}")
    steps.append("Copying package files")
    if any(_is_parser_helper(rel) for rel in collected):
        steps.append("Reloading parser service")
    return InstallPlan(target, list(collected.values()), pip_packages, existing, any(_is_parser_helper(rel) for rel in collected), steps)


def execute_install_plan(plan: InstallPlan, context=None, progress: Progress | None = None) -> PackageActionResult:
    lines: list[str] = []
    written: list[Path] = []
    with _PACKAGE_LOCK:
        _progress(progress, "Resolving dependency plan")
        _install_python_packages(plan.pip_packages, progress)
        try:
            _progress(progress, "Copying package files")
            for file in plan.files:
                target = _target(file.path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    content = file.content or b""
                    if target.read_bytes() == content:
                        lines.append(f"Already installed: {file.path}")
                    else:
                        target.write_bytes(content)
                        lines.append(f"Updated file: {file.path}")
                    continue
                target.write_bytes(file.content or b"")
                written.append(target)
                lines.append(f"Installed file: {file.path}")
        except Exception:
            for path in reversed(written):
                path.unlink(missing_ok=True)
            _remove_empty_dirs()
            raise
        if plan.parser_reload_needed:
            _progress(progress, "Reloading parser service")
            _reload_parser_service(context, lines)
        if context is not None:
            _set_enabled_frontends(context, add=_frontends(plan.files), remove=[], lines=lines)
            _set_autoload_services(context, add=_services(plan.files), remove=[], lines=lines)
        if plan.pip_packages:
            lines.append(f"Installed Python package(s): {', '.join(plan.pip_packages)}")
    return PackageActionResult(True, lines)


def build_uninstall_plan(target: str, *, root_dir: str | Path | None = None) -> UninstallPlan:
    """Resolve installed target + recursive deps, then keep externally referenced deps."""
    candidates: set[str]
    bundle = _resolve_bundle_target(GitStoreBackend(root_dir), target) if root_dir is not None else None
    if bundle:
        candidates = set()
        for rel in _read_bundle_manifest(GitStoreBackend(root_dir), bundle)["files"]:
            candidates.update(_dependency_closure_from_installed(_validate_rel_path(rel)))
        if not candidates:
            return UninstallPlan(bundle, [], {}, [], {}, False, ["Resolving dependency plan"])
    else:
        candidates = _dependency_closure_from_installed(_resolve_installed_target(target))
    return _uninstall_plan_from_candidates(target, candidates)


def _uninstall_plan_from_candidates(target: str, candidates: set[str]) -> UninstallPlan:
    keep_files: dict[str, str] = {}
    kept_pip: dict[str, str] = {}

    refs = _external_references(candidates)
    changed = True
    while changed:
        changed = False
        for rel in list(candidates):
            if rel in keep_files:
                continue
            reason = refs["files"].get(rel)
            if reason:
                keep_files[rel] = reason
                changed = True
                for dep in _meta_from_installed(rel).dependencies_files:
                    if dep in candidates and dep not in keep_files:
                        keep_files[dep] = f"needed by kept dependency {rel}"
                        changed = True
                for dep in _meta_from_installed(rel).dependencies_pip:
                    kept_pip.setdefault(dep, f"needed by kept dependency {rel}")

    kernel = _kernel_requirements()
    pip_candidates = _unique(pip for rel in candidates for pip in _meta_from_installed(rel).dependencies_pip)
    pip_remove = []
    for name in pip_candidates:
        norm = _normalize_pip(name)
        if norm in kernel:
            kept_pip[name] = "kernel requirement"
        elif refs["pip"].get(norm):
            kept_pip[name] = refs["pip"][norm]
        elif name not in kept_pip:
            pip_remove.append(name)

    remove_files = sorted((rel for rel in candidates if rel not in keep_files), key=lambda rel: len(PurePosixPath(rel).parts), reverse=True)
    steps = ["Resolving dependency plan", "Deleting package files"]
    if pip_remove:
        steps.append("Uninstalling Python package(s): " + ", ".join(pip_remove))
    if any(_is_parser_helper(rel) for rel in candidates):
        steps.append("Reloading parser service")
    return UninstallPlan(target, remove_files, keep_files, pip_remove, kept_pip, any(_is_parser_helper(rel) for rel in candidates), steps)


def execute_uninstall_plan(plan: UninstallPlan, context=None, cleanup_choices=None, progress: Progress | None = None) -> PackageActionResult:
    lines: list[str] = []
    with _PACKAGE_LOCK:
        _progress(progress, "Resolving dependency plan")
        if context is not None:
            _set_enabled_frontends(context, add=[], remove=_frontends([PlannedFile(rel) for rel in plan.remove_files]), lines=lines)
            _set_autoload_services(context, add=[], remove=_services([PlannedFile(rel) for rel in plan.remove_files]), lines=lines)
        _progress(progress, "Deleting package files")
        for rel in plan.remove_files:
            _target(rel).unlink(missing_ok=True)
            lines.append(f"Removed file: {rel}")
        for rel, reason in sorted(plan.keep_files.items()):
            lines.append(f"Kept file: {rel} ({reason})")
        _remove_empty_dirs()
        _uninstall_python_packages(plan.pip_packages, progress, lines)
        if plan.kept_pip_packages:
            kept = ", ".join(f"{name} ({reason})" for name, reason in sorted(plan.kept_pip_packages.items(), key=lambda item: item[0].lower()))
            lines.append(f"Kept Python package(s): {kept}")
        if plan.parser_reload_needed:
            _progress(progress, "Reloading parser service")
            _reload_parser_service(context, lines)
    return PackageActionResult(True, lines)


def read_dependency_meta(path: str | Path, content: bytes | str) -> DependencyMeta:
    """Parse dependency metadata without importing plugin code."""
    rel = _validate_rel_path(str(path))
    text = content.decode("utf-8") if isinstance(content, bytes) else content
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise PackageError(f"Cannot parse dependency metadata from {rel}: {e}") from e
    found = {name: [] for name in DEPENDENCY_FIELDS}

    def collect(assign):
        targets = []
        value = None
        if isinstance(assign, ast.Assign):
            targets = [t.id for t in assign.targets if isinstance(t, ast.Name)]
            value = assign.value
        elif isinstance(assign, ast.AnnAssign) and isinstance(assign.target, ast.Name):
            targets = [assign.target.id]
            value = assign.value
        for name in targets:
            if name in found:
                found[name].extend(_literal_str_list(value, name, rel))

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            collect(node)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.Assign, ast.AnnAssign)):
                    collect(item)
    return DependencyMeta(rel, tuple(_validate_rel_path(p) for p in found["dependencies_files"]), tuple(_unique(found["dependencies_pip"])))


def _literal_str_list(node, field: str, rel: str) -> list[str]:
    if node is None:
        return []
    try:
        value = ast.literal_eval(node)
    except Exception as e:
        raise PackageError(f"{field} in {rel} must be a literal list of strings.") from e
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise PackageError(f"{field} in {rel} must be a literal list of strings.")
    return list(value)


def _meta_from_bytes(rel: str, content: bytes) -> DependencyMeta:
    return read_dependency_meta(rel, content)


def _meta_from_installed(rel: str) -> DependencyMeta:
    path = _target(rel)
    if not path.exists():
        return DependencyMeta(_validate_rel_path(rel))
    return read_dependency_meta(rel, path.read_text(encoding="utf-8"))


def _dependency_closure_from_installed(target_rel: str) -> set[str]:
    active: list[str] = []
    out: set[str] = set()

    def visit(rel: str):
        rel = _validate_rel_path(rel)
        if rel in active:
            raise PackageError(f"Dependency cycle includes {rel}.")
        if rel in out:
            return
        path = _target(rel)
        if not path.exists():
            return
        active.append(rel)
        try:
            out.add(rel)
            for dep in _meta_from_installed(rel).dependencies_files:
                visit(dep)
        finally:
            active.pop()

    visit(target_rel)
    return out


def _external_references(candidates: set[str]) -> dict[str, dict[str, str]]:
    file_refs: dict[str, str] = {}
    pip_refs: dict[str, str] = {}
    for root in PLUGIN_ROOTS:
        if not root.path.exists():
            continue
        for path in _tree_files(root.path):
            rel = path.resolve().relative_to(root.path.resolve()).as_posix()
            if root.name == "installed" and rel in candidates:
                continue
            try:
                meta = read_dependency_meta(rel, path.read_text(encoding="utf-8"))
            except PackageError:
                continue
            for dep in meta.dependencies_files:
                file_refs.setdefault(dep, f"needed by {root.name}:{rel}")
            for dep in meta.dependencies_pip:
                pip_refs.setdefault(_normalize_pip(dep), f"needed by {root.name}:{rel}")
    return {"files": file_refs, "pip": pip_refs}


def _installed_rel_files() -> list[str]:
    return [path.relative_to(INSTALLED_PLUGINS).as_posix() for path in _tree_files(INSTALLED_PLUGINS)]


def _tree_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.py") if path.is_file() and _is_valid_tree_rel(path.relative_to(root).as_posix()))


def _resolve_store_target(root_dir: str | Path, target: str) -> str:
    return _resolve_stem(target, GitStoreBackend(root_dir).list_python_files(), "store")


def _resolve_installed_target(target: str) -> str:
    return _resolve_stem(target, _installed_rel_files(), "installed plugins")


def _resolve_bundle_target(store: GitStoreBackend, target: str) -> str | None:
    stem = _target_stem(target)
    matches = [rel for rel in _bundle_manifest_files(store) if PurePosixPath(rel).stem == stem]
    if len(matches) > 1:
        raise PackageError(f"Ambiguous bundle target {stem}: {', '.join(sorted(matches))}")
    return matches[0] if matches else None


def _bundle_manifest_files(store: GitStoreBackend) -> list[str]:
    return sorted(rel for rel in store.list_tree_files() if _is_bundle_manifest(rel))


def _is_bundle_manifest(rel: str) -> bool:
    p = PurePosixPath(rel.replace("\\", "/"))
    return len(p.parts) == 2 and p.parts[0] == "bundles" and p.suffix == ".json"


def _read_bundle_manifest(store: GitStoreBackend, rel: str) -> dict:
    try:
        manifest = json.loads(store.get_tree_file_bytes(rel).decode("utf-8"))
    except json.JSONDecodeError as e:
        raise PackageError(f"Invalid bundle manifest {rel}: {e}") from e
    if not isinstance(manifest, dict):
        raise PackageError(f"Bundle manifest must be an object: {rel}")
    files = manifest.get("files", [])
    if not isinstance(files, list) or not files:
        raise PackageError(f"Bundle manifest needs a non-empty files list: {rel}")
    manifest["files"] = [_validate_rel_path(path) for path in files]
    return manifest


def _resolve_stem(target: str, paths: list[str], label: str) -> str:
    stem = _target_stem(target)
    matches = [rel for rel in paths if PurePosixPath(rel).stem == stem]
    if not matches:
        raise PackageError(f"No {label} file named {stem}.")
    if len(matches) > 1:
        raise PackageError(f"Ambiguous {label} target {stem}: {', '.join(sorted(matches))}")
    return _validate_rel_path(matches[0])


def _target_stem(target: str) -> str:
    text = (target or "").strip().replace("\\", "/")
    if not text:
        raise PackageError("Package target is required.")
    return PurePosixPath(text).stem


def _item(rel: str, *, installed: bool) -> dict:
    rel = _validate_rel_path(rel)
    parts = PurePosixPath(rel).parts
    return {"id": PurePosixPath(rel).stem, "name": PurePosixPath(rel).stem, "path": rel, "family": parts[0], "helper": len(parts) > 1 and parts[1] == "helpers", "installed": installed}


def _validate_rel_path(path: str) -> str:
    p = PurePosixPath(str(path).replace("\\", "/"))
    if p.is_absolute() or not p.parts or p.suffix != ".py" or any(part in {"", ".", ".."} for part in p.parts):
        raise PackageError(f"Invalid package file path: {path}")
    if p.parts[0] not in TREE_ROOTS:
        raise PackageError(f"Package file path must start with one of {sorted(TREE_ROOTS)}: {path}")
    if len(p.parts) not in (2, 3) or (len(p.parts) == 3 and p.parts[1] != "helpers"):
        raise PackageError(f"Package file path must be a plugin or helper file: {path}")
    if not _is_valid_tree_rel(p.as_posix()):
        raise PackageError(f"Invalid plugin/helper file path: {path}")
    return p.as_posix()


def _is_valid_tree_rel(rel: str) -> bool:
    p = PurePosixPath(rel)
    if len(p.parts) == 2:
        return any(p.parts[0] == family and p.name.startswith(prefix) for family, prefix in PLUGIN_FAMILIES.values())
    return len(p.parts) == 3 and p.parts[0] in TREE_ROOTS and p.parts[1] == "helpers" and p.suffix == ".py"


def _target(rel_path: str) -> Path:
    rel = _validate_rel_path(rel_path)
    target = (INSTALLED_PLUGINS / rel).resolve()
    root = INSTALLED_PLUGINS.resolve()
    if target != root and root not in target.parents:
        raise PackageError(f"Target escapes installed plugin root: {rel_path}")
    return target


def _install_python_packages(packages: list[str], progress: Progress | None) -> None:
    if not packages:
        return
    _progress(progress, f"Installing Python package(s): {', '.join(packages)}. This may take a while.")
    result = subprocess.run([sys.executable, "-m", "pip", "install", *packages], capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise PackageError(f"pip install failed for {', '.join(packages)}:\n{result.stderr or result.stdout}")


def _uninstall_python_packages(packages: list[str], progress: Progress | None, lines: list[str]) -> None:
    if not packages:
        return
    _progress(progress, f"Uninstalling Python package(s): {', '.join(packages)}. This may take a while.")
    result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *packages], capture_output=True, text=True, timeout=600)
    if result.returncode:
        lines.append(f"Python package uninstall failed for {', '.join(packages)}: {result.stderr or result.stdout}")
    else:
        lines.append(f"Uninstalled Python package(s): {', '.join(packages)}")


def _kernel_requirements() -> set[str]:
    req = ROOT_DIR / "requirements.txt"
    if not req.exists():
        return set()
    return {_normalize_pip(name) for name in (_requirement_name(line) for line in req.read_text(encoding="utf-8").splitlines()) if name}


def _requirement_name(line: str) -> str | None:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("-"):
        return None
    return re.split(r"[<>=!~;]", line, maxsplit=1)[0].split("[", 1)[0].strip() or None


def _normalize_pip(name: str | None) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").strip().lower())


def _set_enabled_frontends(context, add: list[str], remove: list[str], lines: list[str]) -> None:
    _set_config_list(context, "enabled_frontends", add, remove, "frontend", lines, restart=True)


def _set_autoload_services(context, add: list[str], remove: list[str], lines: list[str]) -> None:
    _set_config_list(context, "autoload_services", add, remove, "service", lines, restart=False)


def _set_config_list(context, key: str, add: list[str], remove: list[str], label: str, lines: list[str], *, restart: bool) -> None:
    config = getattr(context, "config", None)
    if config is None:
        return
    from config import config_manager
    current = _unique([*_config_list(config_manager.load().get(key)), *_config_list(config.get(key))])
    added = [name for name in add if name not in current]
    kept = [name for name in current if name not in remove]
    if not added and kept == current:
        return
    config[key] = kept + added
    config_manager.save(config)
    runtime = getattr(context, "runtime", None)
    if runtime is not None and getattr(runtime, "config", None) is not None:
        runtime.config[key] = config[key]
    if added:
        suffix = " — restart to activate." if restart else " — loading now."
        lines.append(f"Enabled {label}(s): {', '.join(added)}{suffix}")
    dropped = [name for name in current if name in remove]
    if dropped:
        lines.append(f"Disabled {label}(s): {', '.join(dropped)}.")


def _config_list(value) -> list:
    return value if isinstance(value, list) else ([value] if value not in (None, "") else [])


def _frontends(files: list[PlannedFile]) -> list[str]:
    return _unique(_plugin_name(file.path, "frontend") for file in files if _entry_type(file.path) == "frontend")


def _services(files: list[PlannedFile]) -> list[str]:
    return _unique(_plugin_name(file.path, "service") for file in files if _entry_type(file.path) == "service")


def _entry_type(rel: str) -> str | None:
    p = PurePosixPath(rel)
    if len(p.parts) != 2:
        return None
    for plugin_type, (family, prefix) in PLUGIN_FAMILIES.items():
        if p.parts[0] == family and p.name.startswith(prefix):
            return plugin_type
    return None


def _plugin_name(rel: str, plugin_type: str) -> str:
    prefix = PLUGIN_FAMILIES[plugin_type][1]
    stem = PurePosixPath(rel).stem
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def _is_parser_helper(rel: str) -> bool:
    p = PurePosixPath(rel)
    return len(p.parts) == 3 and p.parts[0] == "services" and p.parts[1] == "helpers" and p.name.startswith("parse_")


def _reload_parser_service(context, lines: list[str]) -> None:
    parser = (getattr(context, "services", None) or {}).get("parser") if context is not None else None
    if parser is None:
        return
    try:
        if getattr(parser, "loaded", False):
            parser.unload()
        parser.load()
        lines.append("Reloaded parser service; file parsers are now active.")
    except Exception as e:
        lines.append(f"Parser service reload failed (restart to apply): {e}")


def _remove_empty_dirs():
    root = INSTALLED_PLUGINS
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _progress(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


def _unique(items) -> list:
    return list(dict.fromkeys(item for item in items if item))


PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _validate_package_id(package_id: str):
    if not isinstance(package_id, str) or not PACKAGE_ID_RE.match(package_id):
        raise PackageError(f"Invalid package id: {package_id!r}")
