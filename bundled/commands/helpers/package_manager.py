"""Tree-based package-store install/uninstall operations."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

import trees
from paths import ROOT_DIR
from bundled.commands.helpers.store_backend import GitStoreBackend
from plugins.plugin_paths import PLUGIN_FAMILIES, PLUGIN_ROOTS

INSTALLED_PLUGINS = trees.tree("installed").path

# Every root a tree may hold, read from the one table that declares them. A
# package ships ``parsers/parse_pdf.py`` and whatever needs it imports it;
# ``scripts/`` holds SDK code that is run rather than registered, which the
# store can ship for the same reason — the file is the whole package.
#
# ``bundles`` is deliberately absent: it is not a root the kernel routes, so it
# is not in ``trees``, and it keeps the bespoke handling below.
TREE_ROOTS = {root.name for root in trees.ROOTS}

#: Families the store carries that are *not* tree roots. Named here rather
#: than in ``trees`` for exactly the reason above — the kernel routes it not —
#: but named somewhere, so a menu of "what can I install" can be derived
#: instead of retyped.
EXTRA_FAMILIES = ("bundles",)

DEPENDENCY_FIELDS = ("dependencies_files", "dependencies_pip")
_PACKAGE_LOCK = threading.RLock()
Progress = Callable[[str], None]

logger = logging.getLogger("PackageManager")


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
    helper_rescan_needed: bool
    progress_steps: list[str]
    # Store commit the plan was resolved against — recorded in the action
    # ledger on install as provenance (and the seed of future versioning).
    store_commit: str | None = None


@dataclass
class UninstallPlan:
    target: str
    remove_files: list[str]
    keep_files: dict[str, str]
    pip_packages: list[str]
    kept_pip_packages: dict[str, str]
    helper_rescan_needed: bool
    progress_steps: list[str]


def search_packages(root_dir: str | Path, query: str = "") -> list[dict]:
    """Return available store files matching a stem/name query."""
    q = (query or "").strip().lower()
    store = GitStoreBackend(root_dir)
    items = [_item(rel, installed=False) for rel in store.list_python_files()]
    items += search_bundles(root_dir)
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
    """Return installed plugin/helper items; the tree is the source of truth."""
    return sorted((_item(rel, installed=True) for rel in _installed_rel_files()), key=lambda item: (item["family"], item["id"], item["path"]))


def removable_packages() -> list[dict]:
    return installed_packages()


def package_info(root_dir: str | Path, target: str) -> dict:
    rel = _resolve_store_target(root_dir, target)
    meta = _meta_from_bytes(rel, GitStoreBackend(root_dir).get_tree_file_bytes(rel))
    return {**_item(rel, installed=False), "dependencies_files": list(meta.dependencies_files), "dependencies_pip": list(meta.dependencies_pip)}


def install_package(root_dir: str | Path, target: str, context=None, *, requested: bool = True, progress: Progress | None = None) -> PackageActionResult:
    return execute_install_plan(build_install_plan(root_dir, target), context, progress=progress)


def uninstall_package(target: str, context=None, progress: Progress | None = None, root_dir: str | Path | None = None) -> PackageActionResult:
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
    if any(_is_rescannable_helper(rel) for rel in collected):
        steps.append("Rescanning parsers and LLM backends")
    # Provenance is best-effort: stub/test backends may not resolve a commit.
    resolve_commit = getattr(store, "resolve_commit", lambda: None)
    return InstallPlan(target, list(collected.values()), pip_packages, existing,
                       any(_is_rescannable_helper(rel) for rel in collected), steps,
                       store_commit=resolve_commit())


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _record_package_action(context, action_type: str, target: str, *, ok: bool,
                           commit: str | None = None, files: dict[str, str] | None = None,
                           pip: list[str] | None = None, error: str | None = None) -> None:
    """Best-effort provenance row in the action ledger: which package, from
    which store commit, with which per-file content hashes. No-op when the
    context carries no ledger-capable db (e.g. plan-only unit tests)."""
    db = getattr(context, "db", None) if context is not None else None
    record = getattr(db, "record_action", None)
    if record is None:
        return
    try:
        record(origin="system", action_type=action_type, ok=ok, name=target,
               user_id=getattr(context, "user_id", None),
               data={"commit": commit, "files": files or {}, "pip": pip or []},
               error_message=error)
    except Exception:
        pass


def execute_install_plan(plan: InstallPlan, context=None, progress: Progress | None = None) -> PackageActionResult:
    try:
        result = _execute_install_plan(plan, context, progress)
    except Exception as e:
        _record_package_action(context, "package_install", plan.target, ok=False,
                               commit=plan.store_commit, error=str(e))
        raise
    _record_package_action(
        context, "package_install", plan.target, ok=True, commit=plan.store_commit,
        files={f.path: _sha256(f.content or b"") for f in plan.files},
        pip=plan.pip_packages)
    return result


def _execute_install_plan(plan: InstallPlan, context=None, progress: Progress | None = None) -> PackageActionResult:
    lines: list[str] = []
    written: list[Path] = []
    # Every file whose bytes actually landed — new or changed. A byte-identical
    # file is already set up, so this is also what keeps ``on_install`` firing
    # on a fresh install and on an update that changed something, and on
    # nothing else.
    changed: list[str] = []
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
                        changed.append(file.path)
                        lines.append(f"Updated file: {file.path}")
                    continue
                target.write_bytes(file.content or b"")
                written.append(target)
                changed.append(file.path)
                lines.append(f"Installed file: {file.path}")
        except Exception:
            for path in reversed(written):
                path.unlink(missing_ok=True)
            _remove_empty_dirs()
            raise
        if plan.helper_rescan_needed:
            _progress(progress, "Rescanning parsers and LLM backends")
            _rescan_helpers(context, lines)
        # Before the services load, so a plugin that arranges its own config
        # has done it by the time it starts reading any.
        _progress(progress, "Running package setup")
        _run_lifecycle(changed, "on_install", lines, context)
        if context is not None:
            services = _services(plan.files)
            _set_enabled_frontends(context, add=_frontends(plan.files), remove=[], lines=lines)
            _set_autoload_services(context, add=services, remove=[], lines=lines)
            _load_registered_services(context, services, lines)
        if plan.pip_packages:
            lines.append(f"Installed Python package(s): {', '.join(plan.pip_packages)}")
    return PackageActionResult(True, lines)


def outdated_packages(root_dir: str | Path) -> list[str]:
    """Installed files whose store copy now differs (the store always wins)."""
    store = GitStoreBackend(root_dir)
    store.refresh(force=True)
    store_files = set(store.list_python_files())
    out = []
    for rel in _installed_rel_files():
        if rel in store_files and store.get_tree_file_bytes(rel) != _target(rel).read_bytes():
            out.append(rel)
    return sorted(out)


def update_packages(root_dir: str | Path, context=None, progress: Progress | None = None) -> PackageActionResult:
    """Re-copy every installed file whose store copy changed, plus any new
    dependencies those updated files now declare."""
    rels = outdated_packages(root_dir)
    if not rels:
        return PackageActionResult(True, ["All installed packages are up to date."])
    plan = _install_plan_from_roots(GitStoreBackend(root_dir), rels, "update")
    result = execute_install_plan(plan, context, progress=progress)
    result.lines.insert(0, f"Updating {len(rels)} file(s): " + ", ".join(PurePosixPath(rel).stem for rel in rels))
    return result


def build_uninstall_plan(target: str, *, root_dir: str | Path | None = None) -> UninstallPlan:
    """Resolve installed target + recursive deps, then keep externally referenced deps."""
    candidates: set[str]
    bundle = _resolve_bundle_target(GitStoreBackend(root_dir), target) if root_dir is not None else None
    if bundle:
        candidates = set()
        for rel in _read_bundle_manifest(GitStoreBackend(root_dir), bundle)["files"]:
            candidates.update(_dependents_closure_from_installed(_validate_rel_path(rel)))
        if not candidates:
            return UninstallPlan(bundle, [], {}, [], {}, False, ["Resolving dependency plan"])
    else:
        candidates = _dependents_closure_from_installed(_resolve_installed_target(target))
    return _uninstall_plan_from_candidates(target, candidates)


def _uninstall_plan_from_candidates(target: str, candidates: set[str]) -> UninstallPlan:
    keep_files: dict[str, str] = {}
    kept_pip: dict[str, str] = {}

    refs = _external_references(candidates)

    def keep(rel: str, reason: str) -> None:
        if rel in keep_files:
            return
        keep_files[rel] = reason
        meta = _meta_from_installed(rel)
        for dep in meta.dependencies_pip:
            kept_pip.setdefault(dep, f"needed by kept dependency {rel}")
        for dep in meta.dependencies_files:
            if dep in candidates:
                keep(dep, f"needed by kept dependency {rel}")

    for rel in sorted(candidates):
        reason = refs["files"].get(rel)
        if reason:
            keep(rel, reason)

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
    if any(_is_rescannable_helper(rel) for rel in candidates):
        steps.append("Rescanning parsers and LLM backends")
    return UninstallPlan(target, remove_files, keep_files, pip_remove, kept_pip, any(_is_rescannable_helper(rel) for rel in candidates), steps)


def execute_uninstall_plan(plan: UninstallPlan, context=None, progress: Progress | None = None) -> PackageActionResult:
    # Hash installed bytes up front — after execution the files are gone.
    removed_hashes = {}
    for rel in plan.remove_files:
        try:
            removed_hashes[rel] = _sha256(_target(rel).read_bytes())
        except OSError:
            removed_hashes[rel] = None
    try:
        result = _execute_uninstall_plan(plan, context, progress)
    except Exception as e:
        _record_package_action(context, "package_uninstall", plan.target, ok=False, error=str(e))
        raise
    _record_package_action(context, "package_uninstall", plan.target, ok=True,
                           files=removed_hashes, pip=plan.pip_packages)
    return result


def _execute_uninstall_plan(plan: UninstallPlan, context=None, progress: Progress | None = None) -> PackageActionResult:
    lines: list[str] = []
    with _PACKAGE_LOCK:
        _progress(progress, "Resolving dependency plan")
        # First, while the world is still as the plugin left it: its file is
        # on disk to load from, it is still registered, and its pip
        # dependencies are still installed. Only what is actually going —
        # ``plan.keep_files`` is a dependency somebody else still needs, and
        # tearing down for that would be a package uninstalling a neighbour.
        _progress(progress, "Running package cleanup")
        _run_lifecycle(plan.remove_files, "on_uninstall", lines, context)
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
        if plan.helper_rescan_needed:
            _progress(progress, "Rescanning parsers and LLM backends")
            _rescan_helpers(context, lines)
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


def _dependents_closure_from_installed(target_rel: str) -> set[str]:
    """The target plus everything installed that would stop working without it.

    **The edge is followed backwards, and that is the whole of it.** Uninstall
    used to walk ``dependencies_files`` *forwards* — remove a package, remove
    what it needed — which is the relation stated in the file but the opposite
    of the one the question asks. Removing ``tool_hybrid_search`` took
    ``tool_semantic_search`` and ``tool_lexical_search`` with it, two tools that
    work perfectly well alone and that the user may well have installed for
    their own sake, while leaving ``service_memory_retrieve`` — which cannot
    run without the thing that just left — installed, registered, autoloaded,
    and failing every turn.

    A dependency is a *claim about what I need*, never a claim of ownership,
    and nothing in the tree records which files were installed for their own
    sake and which came along for the ride. So the forward direction cannot be
    answered and is not attempted: a file the target needed is left on disk,
    where it is visible in ``/packages list`` and costs a few kilobytes. That
    is the cheap failure. Removing something that still works is not.

    Transitive, because a dependent's dependents are broken just as thoroughly
    — uninstalling ``lexical_search`` takes ``hybrid_search`` and then
    ``memory_retrieve``. Cycles terminate on the visited set rather than
    raising: a cycle is a real problem when *resolving* an install, and merely
    a shape of the graph when sweeping it.
    """
    target_rel = _validate_rel_path(target_rel)
    dependents: dict[str, list[str]] = {}
    for rel in _installed_rel_files():
        try:
            meta = _meta_from_installed(rel)
        except PackageError:
            continue
        for dep in meta.dependencies_files:
            dependents.setdefault(dep, []).append(rel)

    out: set[str] = set()
    queue = [target_rel]
    while queue:
        rel = queue.pop()
        if rel in out or not _target(rel).exists():
            continue
        out.add(rel)
        queue.extend(dependents.get(rel, ()))
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
    matches = [rel for rel in paths if _rel_id(rel) == stem]
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
    if p.is_absolute() or not p.parts or any(part in {"", ".", ".."} for part in p.parts):
        raise PackageError(f"Invalid package file path: {path}")
    if p.suffix != ".py":
        raise PackageError(f"Invalid package file path: {path}")
    if p.parts[0] not in TREE_ROOTS:
        raise PackageError(f"Package file path must start with one of {sorted(TREE_ROOTS)}: {path}")
    if len(p.parts) not in (2, 3) or (len(p.parts) == 3 and p.parts[1] != trees.HELPERS_DIRNAME):
        raise PackageError(f"Package file path must be a plugin or helper file: {path}")
    if not _is_valid_tree_rel(p.as_posix()):
        raise PackageError(f"Invalid plugin/helper file path: {path}")
    return p.as_posix()


def _rel_id(rel: str) -> str:
    """The id a user targets: the file's stem."""
    return PurePosixPath(rel).stem


def _is_valid_tree_rel(rel: str) -> bool:
    """Whether a store path names a file one of the trees can hold.

    Three shapes are legal:

        tools/tool_x.py            a registered file, carrying its root's prefix
        parsers/parse_pdf.py       likewise — the prefix is what a scanner globs
        scripts/backfill.py        an unprefixed root: run, never registered
        tools/helpers/x.py         a helper belonging to one family

    The prefix rule is read off the root rather than restated here: a root
    declaring one is scanned and its files must carry it, and a root declaring
    none is reached by being named, where a prefix would only make the
    validator expect a plugin class. Unprefixed roots are top level only —
    ``scripts/helpers/x.py`` falls through to the three-part branch, which
    admits family folders and nothing else, matching ``isolation.is_script``,
    which is what decides whether an installed script may run at all.
    """
    p = PurePosixPath(rel)
    if p.suffix != ".py":
        return False
    root = trees.roots_by_name.get(p.parts[0])
    if root is None:
        return False
    if len(p.parts) == 2:
        return p.name.startswith(root.prefix)
    return (len(p.parts) == 3 and p.parts[1] == trees.HELPERS_DIRNAME
            and root.family is not None)


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


#: What a successful lifecycle hook is called in the install report. The verb
#: has to differ per moment — "Set up" and "Cleaned up" say what happened,
#: where a shared "Ran on_install" would make the user read a method name.
_LIFECYCLE_LABEL = {"on_install": "Set up", "on_uninstall": "Cleaned up"}


def _run_lifecycle(rels: list[str], method: str, lines: list[str],
                   context=None) -> None:
    """Run one administrative hook over the files a package operation touched.

    A declaration cannot describe what an arbitrary plugin did to the system,
    which is why the config- and table-cleanup this replaces was deferred for
    so long. The plugin's own code can, so this runs it — in an ordinary
    ephemeral box, where every effect is a Request classified like anybody
    else's. Reads and SQL are free; a config write raises one approval dialog.

    **The chain is the authorization.** Taken from the caller rather than
    invented, so the run appears below the ``/packages`` command the person
    typed: attended, so an unsafe Request can actually be *asked* about, but
    two links deep, so it inherits neither ``Chain.typed_command`` nor the
    install's ``approved`` grant. The context comes from the same place, and
    that half is not optional — the reverted attempt at this passed ``None``
    and every config write came back "config is not available in this kernel".

    **Never raises, and never blocks the operation.** A package whose setup
    was declined is still installed; a package whose cleanup failed is still
    removed. Stranding either would be worse than the mess.
    """
    if not rels:
        return
    try:
        from sandbox import bridge, provenance
    except Exception:
        logger.exception("could not reach the sandbox to run %s hooks", method)
        return

    caller = provenance.current()
    chain = getattr(caller, "chain", None)
    ctx = getattr(caller, "context", None) or context
    sandbox = None
    for rel in rels:
        if not rel.endswith(".py"):
            continue
        path = _target(rel)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # The cheap question first: an AST parse per file, and a box only for
        # the rare file that actually declares a hook. Every plugin *inherits*
        # both no-ops, so asking the class would open a box for all of them.
        entries = bridge.lifecycle_entries(source, method)
        if not entries:
            continue
        if sandbox is None:
            sandbox = bridge.get_sandbox()
        stem = PurePosixPath(rel).stem
        for entry, plugin_name in entries:
            try:
                # ``name`` is the plugin's registered identity rather than the
                # file stem, because that is what the chain link becomes and
                # what ``policy._owns_setting`` matches — see
                # ``bridge.lifecycle_entries``.
                result = sandbox.run(path, entry, method=method, once=True,
                                     name=plugin_name or None,
                                     chain=chain, context=ctx)
            except Exception as exc:
                logger.exception("%s failed for %s", method, rel)
                lines.append(f"{method} failed for {stem}: {exc}")
                continue
            if result.ok:
                lines.append(f"{_LIFECYCLE_LABEL[method]}: {stem}")
            else:
                lines.append(f"{method} failed for {stem}: {result.error}")


def _load_registered_services(context, names: list[str], lines: list[str]) -> None:
    services = getattr(context, "services", None) or {}
    for name in names:
        svc = services.get(name)
        if svc is None or getattr(svc, "loaded", False):
            continue
        try:
            if svc.load():
                lines.append(f"Loaded service: {name}")
        except Exception as e:
            lines.append(f"Service '{name}' is enabled but failed to load: {e}")


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
    return _unique(name for file in files
                   if _entry_type(file.path) == "frontend"
                   for name in _registered_names(file, "frontend"))


def _services(files: list[PlannedFile]) -> list[str]:
    """Service autoload names these files carry.

    Install and uninstall used to call two functions here — ``_services`` and
    ``_services_removed`` — whose bodies were character-for-character
    identical. The second existed back when LLM backends were services and
    mapped to a shared ``llm`` router that had to stay autoloaded whatever was
    installed; its docstring still pointed at the helper that did the mapping,
    which has been gone since backends stopped being services.
    """
    return _unique(name for file in files
                   if _entry_type(file.path) == "service"
                   for name in _registered_names(file, "service"))


def _registered_names(file: PlannedFile, plugin_type: str) -> list[str]:
    """What the registry will call the plugins in this file.

    The filename is a *guess* at that and was being written into config as if
    it were the answer: ``services/service_drive.py`` became ``drive`` while
    the class registers as ``google_drive``, so installing it enabled a
    service that does not exist and left the real one off. Every boot then
    warned "unknown service 'drive', skipping" — the class of warning a person
    learns to scroll past, over a config line only a reinstall would rewrite.

    ``service_embed.py`` shows the other half: one file, two services. The
    filename cannot express that at all, so *neither* embedder was enabled.

    Read rather than imported, like every other declaration this module needs
    — and the fallback is the old guess, because an unmigrated plugin declares
    no ``name`` and its filename is the best answer available.
    """
    declared = _declared_names(_source_of(file))
    return declared or [_plugin_name(file.path, plugin_type)]


def _source_of(file: PlannedFile) -> str:
    """A planned file's text: the store copy on install, the installed one on
    uninstall — where the plan carries paths and the files are still there."""
    if file.content is not None:
        return file.content.decode("utf-8", "replace")
    path = _target(file.path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _declared_names(source: str) -> list[str]:
    """Every ``name = "..."`` a class in this source declares, in file order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "name"
                       for target in item.targets):
                continue
            if (isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                    and item.value.value):
                found.append(item.value.value)
    return _unique(found)


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


#: Roots whose contents a kernel registry scans, so installing one means
#: rescanning rather than loading. Both belong to no plugin family — a parser
#: and a backend are routed to by extension and by profile, not registered as
#: capabilities.
RESCANNED_ROOTS = ("parsers", "llm")


def _is_rescannable_helper(rel: str) -> bool:
    """Whether an installed file is a parser or an LLM backend."""
    p = PurePosixPath(rel)
    return len(p.parts) == 2 and p.parts[0] in RESCANNED_ROOTS


def _rescan_helpers(context, lines: list[str]) -> None:
    """Rescan the helper trees so a newly installed helper is live at once.

    Both families here are kernel *routing* rather than services, so this is a
    rescan and not a load/unload cycle — there is nothing holding state to
    tear down. An LLM backend additionally needs the brains rebuilt, since a
    profile naming a backend that was missing a moment ago should start
    working without a restart.
    """
    try:
        import parsing

        count = parsing.discover()
        lines.append(f"Rescanned parsers: {count} module(s) now active.")
    except Exception as e:
        lines.append(f"Parser rescan failed (restart to apply): {e}")

    try:
        import llm

        count = llm.discover()
        config = getattr(context, "config", None)
        if config is not None:
            # ``force``, and it is not optional. ``refresh`` short-circuits a
            # profile whose *dict* is unchanged, which is exactly the case
            # here: installing or updating a backend rewrites its **source**
            # and leaves every profile untouched. Without this an update left
            # every open brain running the code it was already running, so
            # `/packages update` reported success and changed nothing.
            llm.refresh(config, force=True)
        lines.append(f"Rescanned LLM backends: {count} now available.")
    except Exception as e:
        lines.append(f"LLM backend rescan failed (restart to apply): {e}")


def _remove_empty_dirs():
    """Tidy up after an uninstall, without un-declaring the layout.

    A declared root stays whether or not anything is installed in it — it is
    the layout's claim about where things go, not a by-product of something
    being there. This used to delete them, so an uninstall silently reverted
    ``trees.materialize`` and ``installed/`` grew and shrank a different set
    of folders depending on what you happened to have.
    """
    root = INSTALLED_PLUGINS
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        if trees.is_root_dir(path):
            continue
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
