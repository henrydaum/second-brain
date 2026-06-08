"""Plan and execute package-store install/uninstall operations."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from paths import INSTALLED_PLUGINS, PACKAGES_DIR, ROOT_DIR
from plugins.commands.helpers.store_backend import GitStoreBackend
from plugins.helpers.plugin_paths import PLUGIN_FAMILIES, plugin_dirs


PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
ALLOWED_ROOTS = {family for family, _prefix in PLUGIN_FAMILIES.values()} | {"helpers"}
# A package's kind is read from its id: ``bundle_*`` is a soft collection
# (manifest only), a plugin family prefix (``tool_``/``task_``/``service_``/
# ``command_``/``frontend_``) is a plugin whose id equals its entrypoint stem,
# and anything else is a shared helper package. See ``_package_kind``.
BUNDLE_PREFIX = "bundle_"
PLUGIN_PREFIXES = tuple(prefix for _family, prefix in PLUGIN_FAMILIES.values())
RECEIPTS_DIR = PACKAGES_DIR / "receipts"
INTERNAL_IMPORTS = {"agent", "attachments", "config", "events", "helpers", "installed_plugins", "paths", "pipeline", "plugins", "runtime", "sandbox_plugins", "state_machine", "templates", *ALLOWED_ROOTS}
PIP_NAMES = {"PIL": "Pillow", "bs4": "beautifulsoup4", "cv2": "opencv-python", "docx": "python-docx", "fitz": "PyMuPDF", "google": "google-api-python-client", "googleapiclient": "google-api-python-client", "pptx": "python-pptx", "sklearn": "scikit-learn", "telegram": "python-telegram-bot", "yaml": "PyYAML"}
_PACKAGE_LOCK = threading.RLock()


class PackageError(RuntimeError):
    """Raised for package validation or execution failures."""


@dataclass
class PackageActionResult:
    """Structured package operation result."""
    ok: bool
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines) if self.lines else ("OK" if self.ok else "Failed")


@dataclass
class PlannedFile:
    package_id: str
    path: str
    content: bytes
    sha256: str


@dataclass
class PlannedPackage:
    id: str
    manifest: dict
    requested: bool
    files: list[PlannedFile]
    entrypoints: list[dict]
    pip_packages: list[str]
    manifest_hash: str


@dataclass
class InstallPlan:
    package_id: str
    requested_packages: list[str]
    auto_packages: list[str]
    packages: list[PlannedPackage]
    existing_dependencies: list[str]
    pip_packages: list[str]
    parser_reload_needed: bool
    progress_steps: list[str]


@dataclass
class UninstallPackagePlan:
    id: str
    receipt: dict
    files: list[str]
    entrypoints: list[dict]
    cleanup: dict[str, list[str]]
    pip_candidates: list[str]


@dataclass
class UninstallPlan:
    target: str
    packages: list[UninstallPackagePlan]
    raw_files: list[str]
    pruned: list[str]
    pip_removals: dict[str, list[str]]
    kept_pip_packages: dict[str, str]
    parser_reload_needed: bool
    needs_confirm: bool
    broken_dependents: list[str]
    progress_steps: list[str]


Progress = Callable[[str], None]


def search_packages(root_dir: str | Path, query: str = "") -> list[dict]:
    """Return packages from the store index matching query."""
    items = GitStoreBackend(root_dir).get_index()
    q = (query or "").strip().lower()
    if not q:
        return sorted(items, key=lambda item: item.get("id", ""))

    def hay(item):
        return " ".join(str(item.get(k, "")) for k in ("id", "name", "description")).lower()

    return sorted([item for item in items if q in hay(item)], key=lambda item: item.get("id", ""))


def installed_packages() -> list[dict]:
    """Return installed package receipts."""
    return sorted((_load_receipt(path) for path in RECEIPTS_DIR.glob("*.json")), key=lambda receipt: receipt.get("id", "")) if RECEIPTS_DIR.exists() else []


def removable_packages() -> list[dict]:
    """Every installed package — all are individually removable.

    Bundle membership is now *soft*, so a package required by an installed bundle
    is still removable on its own (the bundle's manifest is simply left listing a
    member that is gone, tolerated on the bundle's own uninstall). A package
    hard-required by another *plugin* is still offered here; its plan flags
    ``needs_confirm`` so the command can warn before breaking the dependent."""
    return installed_packages()


def package_info(root_dir: str | Path, package_id: str) -> dict:
    """Return one package manifest."""
    _validate_package_id(package_id)
    return _validate_manifest(_CachedStore(GitStoreBackend(root_dir)).get_manifest(package_id))


def install_package(root_dir: str | Path, package_id: str, context=None, *, requested: bool = True, progress: Progress | None = None) -> PackageActionResult:
    """Build and execute an install plan."""
    return execute_install_plan(build_install_plan(root_dir, package_id, requested=requested), context, progress=progress)


def uninstall_package(package_id: str, context=None, cleanup_choices: dict[str, dict[str, bool]] | None = None, progress: Progress | None = None, cleanup_approvals: dict[str, bool] | None = None) -> PackageActionResult:
    """Build and execute an uninstall plan."""
    plan = build_uninstall_plan(package_id)
    if cleanup_choices is None and cleanup_approvals is not None:
        cleanup_choices = {"config": cleanup_approvals, "tables": cleanup_approvals, "pip": {}}
    return execute_uninstall_plan(plan, context, cleanup_choices or {}, progress=progress)


def build_install_plan(root_dir: str | Path, package_id: str, *, requested: bool = True) -> InstallPlan:
    """Resolve a complete install graph before mutating files, receipts, or pip."""
    _validate_package_id(package_id)
    store = _CachedStore(GitStoreBackend(root_dir))
    packages: list[PlannedPackage] = []
    existing: list[str] = []
    active: list[str] = []
    planned_paths: dict[str, bytes] = {}  # path -> content, for cross-package shared-file checks

    def collect(pid: str, is_requested: bool):
        _validate_package_id(pid)
        if pid in active:
            raise PackageError(f"Dependency cycle includes {pid}.")
        if _receipt_path(pid).exists():
            if is_requested:
                raise PackageError(f"Package already installed: {pid}")
            existing.append(pid)
            return
        if any(pkg.id == pid for pkg in packages):
            return
        active.append(pid)
        try:
            manifest = _validate_manifest(store.get_manifest(pid))
            if manifest["id"] != pid:
                raise PackageError(f"Manifest id mismatch: requested {pid}, got {manifest['id']}.")
            for dep in manifest["requires"]:
                collect(dep, False)
            files = _validated_files(manifest)
            entrypoints = _entrypoint_metadata(_entrypoints(manifest, files))
            planned_files = [PlannedFile(pid, rel, store.get_file_bytes(pid, rel), "") for rel in files]
            planned_files = [PlannedFile(f.package_id, f.path, f.content, _sha256(f.content)) for f in planned_files]
            for file in planned_files:
                # Two packages may ship the *same* file (a shared plugin-level
                # helper, e.g. tools/helpers/email_context.py co-owned by every
                # email tool) as long as the bytes are identical — it installs
                # once and is reference-counted on uninstall. Differing bytes for
                # one path is a genuine conflict.
                prior = planned_paths.get(file.path)
                if prior is not None and prior != file.content:
                    raise PackageError(f"Conflicting content for shared file across packages: {file.path}")
                planned_paths[file.path] = file.content
            pip_packages = _packages_to_install({f.path: f.content for f in planned_files}, manifest.get("pip"))
            packages.append(PlannedPackage(pid, manifest, bool(is_requested), planned_files, entrypoints, pip_packages, _sha256(store.get_manifest_bytes(pid))))
        finally:
            active.pop()

    collect(package_id, requested)
    _preflight_collisions({file.path: file.content for pkg in packages for file in pkg.files})
    pip_packages = _unique(name for pkg in packages for name in pkg.pip_packages)
    parser_reload_needed = any(_is_parser_helper(file.path) for pkg in packages for file in pkg.files)
    steps = ["Resolving package plan"]
    if pip_packages:
        steps.append(f"Installing Python package(s): {', '.join(pip_packages)}")
    if packages:
        steps += ["Writing package files", "Writing receipts"]
    if parser_reload_needed:
        steps.append("Reloading parser service")
    return InstallPlan(package_id, [package_id], [pkg.id for pkg in packages if not pkg.requested], packages, _unique(existing), pip_packages, parser_reload_needed, steps)


def execute_install_plan(plan: InstallPlan, context=None, progress: Progress | None = None) -> PackageActionResult:
    """Execute a prebuilt install plan without registering plugins."""
    lines: list[str] = []
    written: list[Path] = []
    receipts: list[Path] = []
    with _PACKAGE_LOCK:
        _progress(progress, "Resolving package plan")
        for dep in plan.existing_dependencies:
            lines.append(f"Dependency already installed: {dep}")
        _preflight_collisions({file.path: file.content for pkg in plan.packages for file in pkg.files})
        _install_python_packages(plan.pip_packages, progress)
        try:
            if plan.packages:
                _progress(progress, "Writing package files")
            for pkg in plan.packages:
                for file in pkg.files:
                    target = _target(file.path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        # A co-owned shared helper already on disk (preflight
                        # guaranteed identical bytes). Leave it; don't track it
                        # for rollback — another package owns it too.
                        continue
                    target.write_bytes(file.content)
                    written.append(target)
            if plan.packages:
                _progress(progress, "Writing receipts")
            for pkg in plan.packages:
                # The install record is the manifest itself plus the two fields
                # only resolvable at install time: the concrete entrypoint
                # metadata and the actual pip list (the store manifest's ``pip``
                # is often null/auto-scanned). Uninstall reads these offline.
                record = {
                    "id": pkg.id,
                    "name": pkg.manifest.get("name", pkg.id),
                    "description": pkg.manifest.get("description", ""),
                    "requires": pkg.manifest["requires"],
                    "files": [f.path for f in pkg.files],
                    "entrypoints": pkg.entrypoints,
                    "pip": pkg.pip_packages,
                }
                _write_receipt(record)
                receipts.append(_receipt_path(pkg.id))
        except Exception:
            for path in reversed(written):
                path.unlink(missing_ok=True)
            for path in reversed(receipts):
                path.unlink(missing_ok=True)
            _remove_empty_dirs()
            raise
        if plan.parser_reload_needed:
            _progress(progress, "Reloading parser service")
            _reload_parser_service_if_needed([file.path for pkg in plan.packages for file in pkg.files], context, lines)
        if context is not None:
            frontends = _unique(name for pkg in plan.packages for name in _frontend_names(pkg.entrypoints))
            if frontends:
                _set_enabled_frontends(context, add=frontends, remove=[], lines=lines)
            services = _unique(name for pkg in plan.packages for name in _service_names(pkg.entrypoints))
            if services:
                _set_autoload_services(context, add=services, remove=[], lines=lines)
        for pkg in plan.packages:
            if pkg.pip_packages:
                lines.append(f"Installed Python package(s): {', '.join(pkg.pip_packages)}")
            lines.append(f"Installed {pkg.id}: {len(pkg.files)} file(s), {len(pkg.entrypoints)} plugin entrypoint(s).")
        if not plan.packages:
            lines.append("Nothing installed.")
    return PackageActionResult(True, lines)


def build_uninstall_plan(name: str) -> UninstallPlan:
    """Resolve an uninstall target into a concrete removal plan.

    ``name`` is matched, in order, against: an installed package id (a bundle, a
    plugin, or a helper package), then an installed file (a private helper named
    directly). A bundle is removed *greedily* — every member that loses its last
    referrer goes with it, but a member kept alive by a package outside the
    removal set survives (``_greedy_removal_set``). A direct package uninstall
    removes that package unconditionally plus any deps it orphans; if a remaining
    non-bundle package still hard-requires it, the plan flags ``needs_confirm``.
    A raw-file target deletes just that file (``needs_confirm`` if a manifest
    still lists it)."""
    installed = {receipt["id"]: receipt for receipt in installed_packages()}
    if name in installed:
        return _plan_package_removal(name, installed)
    rel = _find_installed_file(name)
    if rel is not None:
        return _plan_raw_file_removal(rel, installed)
    raise PackageError(f"Package is not installed: {name}")


def _plan_package_removal(target: str, installed: dict[str, dict]) -> UninstallPlan:
    removal_ids = _greedy_removal_set(target, installed)
    order = [pid for pid in installed if pid in removal_ids]
    broken = sorted(
        pid for pid, receipt in installed.items()
        if pid not in removal_ids and _package_kind(pid) != "bundle"
        and (set(receipt.get("requires") or []) & removal_ids)
    )
    removed_paths = {_target(rel).resolve() for pid in removal_ids for rel in installed[pid].get("files", [])}
    packages = [
        UninstallPackagePlan(
            pid,
            installed[pid],
            list(installed[pid].get("files", [])),
            list(installed[pid].get("entrypoints", [])),
            _cleanup_plan(installed[pid], removed_paths),
            sorted(set(installed[pid].get("pip", [])), key=str.lower),
        )
        for pid in order
    ]
    pip_removals, kept_pip = _safe_pip_removals(packages)
    parser_reload_needed = any(_is_parser_helper(rel) for pkg in packages for rel in pkg.files)
    pruned = sorted(pid for pid in removal_ids if pid != target)
    steps = ["Resolving package plan", "Deleting package files", "Writing receipts"]
    if parser_reload_needed:
        steps.append("Reloading parser service")
    if any(pip_removals.values()):
        steps.append("Uninstalling Python package(s): " + ", ".join(_unique(pkg for pkgs in pip_removals.values() for pkg in pkgs)))
    return UninstallPlan(target, packages, [], pruned, pip_removals, kept_pip, parser_reload_needed, bool(broken), broken, steps)


def _plan_raw_file_removal(rel: str, installed: dict[str, dict]) -> UninstallPlan:
    """Delete a single installed file directly — the escape hatch for a private
    helper named by file. Any manifest still listing it is reported (and gates a
    confirm) but its receipt is left intact: a later package uninstall tolerates
    the already-missing file."""
    owners = sorted(pid for pid, receipt in installed.items() if rel in (receipt.get("files") or []))
    parser_reload_needed = _is_parser_helper(rel)
    steps = ["Resolving package plan", "Deleting file"]
    if parser_reload_needed:
        steps.append("Reloading parser service")
    return UninstallPlan(rel, [], [rel], [], {}, {}, parser_reload_needed, bool(owners), owners, steps)


def _greedy_removal_set(root: str, installed: dict[str, dict]) -> set[str]:
    """``root`` plus every dependency it transitively orphans.

    A package reachable from ``root`` through ``requires`` is removed once all of
    its referrers are themselves in the removal set; a member kept alive by a
    package outside the set (another installed bundle, or an unrelated plugin)
    survives. There is no explicit-install flag — membership alone never
    preserves anything (the greedy rule the user chose)."""
    removal = {root}
    closure = _requires_closure(root, installed)
    changed = True
    while changed:
        changed = False
        for pid in closure:
            if pid not in removal and all(ref in removal for ref in _referrers(pid, installed)):
                removal.add(pid)
                changed = True
    return removal


def _requires_closure(root: str, installed: dict[str, dict]) -> set[str]:
    """Installed packages transitively reachable from ``root`` via ``requires``."""
    closure: set[str] = set()
    stack = [root]
    while stack:
        for dep in installed.get(stack.pop(), {}).get("requires", []) or []:
            if dep in installed and dep not in closure:
                closure.add(dep)
                stack.append(dep)
    return closure


def _referrers(pid: str, installed: dict[str, dict]) -> list[str]:
    return [other for other, receipt in installed.items() if pid in (receipt.get("requires") or [])]


def _find_installed_file(name: str) -> str | None:
    """The rel path of an installed file whose name or stem equals ``name``."""
    root = INSTALLED_PLUGINS
    if not root.exists():
        return None
    for path in sorted(root.rglob("*.py")):
        if path.is_file() and name in (path.name, path.stem):
            return path.resolve().relative_to(root.resolve()).as_posix()
    return None


def execute_uninstall_plan(plan: UninstallPlan, context=None, cleanup_choices: dict[str, dict[str, bool]] | None = None, progress: Progress | None = None) -> PackageActionResult:
    """Execute a prebuilt uninstall plan without unloading plugins."""
    choices = cleanup_choices or {}
    lines: list[str] = []
    with _PACKAGE_LOCK:
        _progress(progress, "Resolving package plan")
        for pkg in plan.packages:
            if pkg.cleanup["kept_settings"]:
                lines.append(f"Kept config setting(s) still declared by other plugins: {', '.join(pkg.cleanup['kept_settings'])}")
            if pkg.cleanup["kept_tables"]:
                lines.append(f"Kept table(s) still used by remaining tasks; their data may now be stale: {', '.join(pkg.cleanup['kept_tables'])}")
        _progress(progress, "Deleting package files")
        removed_rels = [rel for pkg in plan.packages for rel in pkg.files] + list(plan.raw_files)
        # A shared plugin-level helper may be co-owned by a package that survives
        # this removal — keep such files on disk (reference-counted).
        removed_ids = {pkg.id for pkg in plan.packages}
        kept_by_others = {rel for receipt in installed_packages() if receipt.get("id") not in removed_ids for rel in receipt.get("files", [])}
        for pkg in plan.packages:
            _apply_selected_cleanup(context, pkg.cleanup, choices, pkg.id, lines)
            for rel in sorted(pkg.files, reverse=True):
                if rel not in kept_by_others:
                    _target(rel).unlink(missing_ok=True)
            _receipt_path(pkg.id).unlink(missing_ok=True)
            lines.append(f"Uninstalled {pkg.id}.")
        for rel in plan.raw_files:
            _target(rel).unlink(missing_ok=True)
            lines.append(f"Deleted file {rel}.")
            if plan.broken_dependents:
                lines.append(f"Warning: {rel} was still listed by package(s): {', '.join(plan.broken_dependents)}.")
        _remove_empty_dirs()
        if plan.parser_reload_needed:
            _progress(progress, "Reloading parser service")
            _reload_parser_service_if_needed(removed_rels, context, lines)
        if context is not None:
            frontends = _unique(name for pkg in plan.packages for name in _frontend_names(pkg.entrypoints))
            if frontends:
                _set_enabled_frontends(context, add=[], remove=frontends, lines=lines)
            services = _unique(name for pkg in plan.packages for name in _service_names(pkg.entrypoints))
            if services:
                _set_autoload_services(context, add=[], remove=services, lines=lines)
        selected_pip = _unique(name for pkg in plan.packages for name in plan.pip_removals.get(pkg.id, []) if choices.get("pip", {}).get(pkg.id))
        _uninstall_python_packages(selected_pip, progress, lines)
        if plan.kept_pip_packages:
            kept = ", ".join(f"{name} ({reason})" for name, reason in sorted(plan.kept_pip_packages.items(), key=lambda item: item[0].lower()))
            lines.append(f"Kept Python package(s): {kept}")
        if plan.pruned:
            lines.append(f"Also removed (no longer needed): {', '.join(plan.pruned)}")
    return PackageActionResult(True, lines)


def uninstall_cleanup_plans(package_id: str) -> list[tuple[str, dict]]:
    """Compatibility helper for callers/tests that need cleanup plan summaries."""
    plan = build_uninstall_plan(package_id)
    return [(pkg.id, pkg.cleanup) for pkg in plan.packages if pkg.cleanup["settings"] or pkg.cleanup["tables"]]


def cleanup_prompt(cleanup: dict) -> str:
    prompt = "Delete package-owned data?\n\n"
    if cleanup["settings"]:
        prompt += "Config settings: " + ", ".join(cleanup["settings"]) + "\n"
    if cleanup["tables"]:
        prompt += "Tables: " + ", ".join(cleanup["tables"]) + "\n"
    return prompt.strip()


def cleanup_pip_prompt(packages: list[str]) -> str:
    return "Uninstall safe package-owned Python deps?\n\nPython packages: " + ", ".join(packages)


def _progress(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


class _CachedStore:
    def __init__(self, backend):
        self.backend = backend
        self.manifest_bytes: dict[str, bytes] = {}
        self.files: dict[tuple[str, str], bytes] = {}

    def get_manifest(self, package_id: str) -> dict:
        return json.loads(self.get_manifest_bytes(package_id).decode("utf-8"))

    def get_manifest_bytes(self, package_id: str) -> bytes:
        if package_id not in self.manifest_bytes:
            self.manifest_bytes[package_id] = self.backend.get_manifest_bytes(package_id)
        return self.manifest_bytes[package_id]

    def get_file_bytes(self, package_id: str, rel_path: str) -> bytes:
        key = (package_id, rel_path)
        if key not in self.files:
            self.files[key] = self.backend.get_file_bytes(package_id, rel_path)
        return self.files[key]


def _entrypoint_metadata(entrypoints: list[str]) -> list[dict]:
    return [{"path": rel, "type": _entrypoint_type(rel), "name": ""} for rel in entrypoints]


def _frontend_names(entrypoints: list[dict]) -> list[str]:
    """Frontend transport names contributed by these entrypoints.

    A frontend ships as ``frontends/frontend_<name>.py`` and registers under
    ``<name>``, so the transport name is the filename stem minus the ``frontend_``
    prefix — the same convention discovery keys on.
    """
    prefix = PLUGIN_FAMILIES["frontend"][1]
    names = []
    for ep in entrypoints:
        if ep.get("type") != "frontend":
            continue
        stem = PurePosixPath(ep["path"]).stem
        names.append(stem[len(prefix):] if stem.startswith(prefix) else stem)
    return names


def _set_enabled_frontends(context, add: list[str], remove: list[str], lines: list[str]) -> None:
    """Add/remove frontend names in ``enabled_frontends`` and persist.

    Installing a frontend should activate it without the user hunting through
    ``/frontends`` — but frontends only start at boot, so we update config and
    tell the user to restart. The ``config_manager.save`` emits ``CONFIG_CHANGED``,
    which refreshes the running config in place (see ConversationRuntime).
    """
    config = getattr(context, "config", None)
    if config is None:
        return
    enabled = list(config.get("enabled_frontends", []) or [])
    added = [n for n in add if n not in enabled]
    kept = [n for n in enabled if n not in remove]
    if not added and kept == enabled:
        return
    config["enabled_frontends"] = kept + added
    from config import config_manager
    config_manager.save(config)
    # config is a per-call copy; keep the canonical runtime config in step so
    # /frontends reflects the change without a restart (the frontend still starts
    # on restart).
    runtime = getattr(context, "runtime", None)
    if runtime is not None and getattr(runtime, "config", None) is not None:
        runtime.config["enabled_frontends"] = config["enabled_frontends"]
    if added:
        lines.append(f"Enabled frontend(s): {', '.join(added)} — restart to activate.")
    dropped = [n for n in enabled if n in remove]
    if dropped:
        lines.append(f"Disabled frontend(s): {', '.join(dropped)}.")


def _service_names(entrypoints: list[dict]) -> list[str]:
    """Service registry names contributed by these entrypoints.

    A service ships as ``services/service_<name>.py`` and registers under
    ``<name>`` — the same filename convention discovery and ``autoload_services``
    key on, mirroring ``_frontend_names``.
    """
    prefix = PLUGIN_FAMILIES["service"][1]
    names = []
    for ep in entrypoints:
        if ep.get("type") != "service":
            continue
        stem = PurePosixPath(ep["path"]).stem
        names.append(stem[len(prefix):] if stem.startswith(prefix) else stem)
    return names


def _set_autoload_services(context, add: list[str], remove: list[str], lines: list[str]) -> None:
    """Add/remove service names in ``autoload_services`` and persist.

    Installing a service should start it without the user editing config — the
    plugin watcher hot-loads the freshly written file, and a service listed in
    ``autoload_services`` loads on registration (see ``plugin_discovery``) and on
    every future boot. Mirrors ``_set_enabled_frontends``; unlike frontends,
    services hot-load, so no restart is needed.
    """
    config = getattr(context, "config", None)
    if config is None:
        return
    enabled = list(config.get("autoload_services", []) or [])
    added = [n for n in add if n not in enabled]
    kept = [n for n in enabled if n not in remove]
    if not added and kept == enabled:
        return
    config["autoload_services"] = kept + added
    from config import config_manager
    config_manager.save(config)
    # config is a per-call copy; keep the canonical runtime config in step so the
    # watcher (which shares the runtime config object) sees the new entry when it
    # fires on the just-written file and loads the service live.
    runtime = getattr(context, "runtime", None)
    if runtime is not None and getattr(runtime, "config", None) is not None:
        runtime.config["autoload_services"] = config["autoload_services"]
    if added:
        lines.append(f"Enabled service(s): {', '.join(added)} — loading now.")
    dropped = [n for n in enabled if n in remove]
    if dropped:
        lines.append(f"Disabled service(s): {', '.join(dropped)}.")


def _entrypoint_type(rel: str) -> str:
    p = PurePosixPath(rel)
    for plugin_type, (family, prefix) in PLUGIN_FAMILIES.items():
        if p.parts and p.parts[0] == family and p.name.startswith(prefix):
            return plugin_type
    raise PackageError(f"Invalid plugin entrypoint path: {rel}")


def _is_parser_helper(rel: str) -> bool:
    p = PurePosixPath(rel)
    return len(p.parts) == 3 and p.parts[0] == "services" and p.parts[1] == "helpers" and p.suffix == ".py" and p.name.startswith("parse_")


def _reload_parser_service_if_needed(file_rels, context, lines: list[str]) -> None:
    if not any(_is_parser_helper(rel) for rel in file_rels):
        return
    parser = (getattr(context, "services", None) or {}).get("parser")
    if parser is None:
        return
    try:
        if getattr(parser, "loaded", False):
            parser.unload()
        parser.load()
        lines.append("Reloaded parser service; file parsers are now active.")
    except Exception as e:
        lines.append(f"Parser service reload failed (restart to apply): {e}")


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


def _packages_to_install(file_bytes: dict[str, bytes], declared_pip: list[str] | None = None) -> list[str]:
    if declared_pip is not None:
        return sorted(set(declared_pip), key=str.lower)
    return _missing_pip_packages(file_bytes)


def _missing_pip_packages(file_bytes: dict[str, bytes]) -> list[str]:
    roots = set()
    for rel, content in file_bytes.items():
        if rel.endswith(".py"):
            roots.update(_import_roots(content))
    stdlib = set(getattr(sys, "stdlib_module_names", set())) | set(sys.builtin_module_names)
    return sorted({PIP_NAMES.get(root, root) for root in roots if root not in stdlib and root not in INTERNAL_IMPORTS and not _module_available(root)}, key=str.lower)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return True


def _import_roots(content: bytes) -> set[str]:
    try:
        tree = ast.parse(content.decode("utf-8"))
    except Exception:
        return set()
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _safe_pip_removals(packages: list[UninstallPackagePlan]) -> tuple[dict[str, list[str]], dict[str, str]]:
    removed_ids = {pkg.id for pkg in packages}
    kernel = _kernel_requirements()
    remaining = {_normalize_pip(name) for receipt in installed_packages() if receipt.get("id") not in removed_ids for name in receipt.get("pip", [])}
    removals: dict[str, list[str]] = {}
    kept: dict[str, str] = {}
    for pkg in packages:
        safe: list[str] = []
        for name in pkg.pip_candidates:
            norm = _normalize_pip(name)
            if not norm:
                kept[name] = "unknown ownership"
            elif norm in kernel:
                kept[name] = "kernel requirement"
            elif norm in remaining:
                kept[name] = "needed by another installed package"
            else:
                safe.append(name)
        removals[pkg.id] = sorted(set(safe), key=str.lower)
    return removals, kept


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


def _cleanup_plan(receipt: dict, removed_paths: set[Path] | None = None) -> dict[str, list[str]]:
    owned_paths = {_target(rel).resolve() for rel in receipt.get("files", [])}
    owned = _declared_state(owned_paths)
    used_elsewhere = _declared_state(_other_plugin_files(removed_paths or owned_paths))
    settings = sorted(owned["settings"] - used_elsewhere["settings"])
    tables = sorted(owned["writes"] - (used_elsewhere["reads"] | used_elsewhere["writes"]))
    return {"settings": settings, "tables": tables, "kept_settings": sorted(owned["settings"] - set(settings)), "kept_tables": sorted(owned["writes"] - set(tables))}


def _apply_selected_cleanup(context, cleanup: dict, choices: dict[str, dict[str, bool]], package_id: str, lines: list[str]) -> None:
    settings = cleanup["settings"] if choices.get("config", {}).get(package_id) else []
    tables = cleanup["tables"] if choices.get("tables", {}).get(package_id) else []
    if cleanup["settings"] and not settings:
        lines.append("Kept package config setting(s).")
    if cleanup["tables"] and not tables:
        lines.append("Kept package SQL table(s).")
    if settings:
        from config import config_manager
        saved = config_manager.load_plugin_config()
        for key in settings:
            saved.pop(key, None)
            if getattr(context, "config", None) is not None:
                context.config.pop(key, None)
        config_manager.save_plugin_config(saved)
        lines.append(f"Deleted config setting(s): {', '.join(settings)}")
    if tables and getattr(context, "db", None) is not None:
        with context.db.lock:
            for table in tables:
                context.db._validate_identifier(table)
                context.db.conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            context.db.conn.commit()
        lines.append(f"Deleted table(s): {', '.join(tables)}")


def _other_plugin_files(excluded: set[Path]) -> set[Path]:
    files = set()
    for plugin_type in PLUGIN_FAMILIES:
        for plugin_dir in plugin_dirs(plugin_type):
            if plugin_dir.path.exists():
                files.update(path.resolve() for path in plugin_dir.path.glob(f"{plugin_dir.prefix}*.py") if path.resolve() not in excluded)
    return files


def _declared_state(paths: set[Path]) -> dict[str, set[str]]:
    state = {"settings": set(), "reads": set(), "writes": set()}
    for path in paths:
        if path.suffix != ".py" or not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.Module)):
                for assign in [item for item in getattr(node, "body", []) if isinstance(item, ast.Assign)]:
                    names = [target.id for target in assign.targets if isinstance(target, ast.Name)]
                    if "config_settings" in names:
                        state["settings"].update(_setting_keys(assign.value))
                    for attr in ("reads", "writes"):
                        if attr in names:
                            state[attr].update(_literal_strings(assign.value))
    return state


def _setting_keys(node) -> set[str]:
    keys = set()
    try:
        settings = ast.literal_eval(node)
    except Exception:
        return keys
    for entry in settings if isinstance(settings, list) else []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], str):
            keys.add(entry[1])
    return keys


def _literal_strings(node) -> set[str]:
    try:
        value = ast.literal_eval(node)
    except Exception:
        return set()
    return {item for item in value if isinstance(item, str)} if isinstance(value, list) else set()


def _validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        raise PackageError("Manifest must be a JSON object.")
    package_id = manifest.get("id", "")
    _validate_package_id(package_id)
    requires = manifest.get("requires", [])
    files = manifest.get("files", [])
    entrypoints = manifest.get("entrypoints", None)
    if not isinstance(requires, list) or any(not isinstance(dep, str) for dep in requires):
        raise PackageError("Manifest requires must be a list of package ids.")
    for dep in requires:
        _validate_package_id(dep)
    if not isinstance(files, list) or any(not isinstance(path, str) for path in files):
        raise PackageError("Manifest files must be a list of relative paths.")
    if entrypoints is not None and (not isinstance(entrypoints, list) or any(not isinstance(path, str) for path in entrypoints)):
        raise PackageError("Manifest entrypoints must be a list of relative paths.")
    pip = manifest.get("pip", None)
    if pip is not None and (not isinstance(pip, list) or any(not isinstance(name, str) for name in pip)):
        raise PackageError("Manifest pip must be a list of PyPI package names.")
    normalized = dict(manifest)
    normalized["requires"] = requires
    normalized["files"] = [_validate_rel_path(path) for path in files]
    if entrypoints is not None:
        normalized["entrypoints"] = [_validate_rel_path(path) for path in entrypoints]
    if pip is not None:
        normalized["pip"] = pip
    _validate_kind(normalized)
    return normalized


def _package_kind(package_id: str) -> str:
    """Classify a package id by its prefix: ``bundle`` | ``plugin`` | ``helper``."""
    if package_id.startswith(BUNDLE_PREFIX):
        return "bundle"
    if package_id.startswith(PLUGIN_PREFIXES):
        return "plugin"
    return "helper"


def _validate_kind(manifest: dict) -> None:
    """Enforce the id-prefix contract: a ``bundle_`` package ships no files (it is
    a soft collection); any other fileless meta-package is rejected (it must be
    named ``bundle_*``); a plugin-prefixed package ships an entrypoint named
    exactly after its id (``service_x`` ⇒ ``services/service_x.py``). Helper
    packages (no prefix, with files) are otherwise unconstrained."""
    package_id = manifest["id"]
    kind = _package_kind(package_id)
    files = manifest["files"]
    if kind == "bundle":
        if files:
            raise PackageError(f"Bundle {package_id} must not ship files; it only lists requires.")
        return
    if not files:
        raise PackageError(f"Fileless meta-package {package_id} must be named with the 'bundle_' prefix.")
    if kind == "plugin":
        family, _prefix = next((fam, pre) for fam, pre in PLUGIN_FAMILIES.values() if package_id.startswith(pre))
        expected = f"{family}/{package_id}.py"
        if expected not in files:
            raise PackageError(f"Plugin package {package_id} must ship its entrypoint file {expected}.")


def _validated_files(manifest: dict) -> list[str]:
    files = manifest["files"]
    if not files and not manifest.get("requires"):
        raise PackageError("Manifest must include at least one file or dependency.")
    return files


def _entrypoints(manifest: dict, files: list[str]) -> list[str]:
    entrypoints = manifest.get("entrypoints") if manifest.get("entrypoints") is not None else [path for path in files if _is_entrypoint(path)]
    file_set = set(files)
    for path in entrypoints:
        if path not in file_set:
            raise PackageError(f"Entrypoint is not listed in files: {path}")
        if not _is_entrypoint(path):
            raise PackageError(f"Invalid plugin entrypoint path: {path}")
    return entrypoints


def _is_entrypoint(path: str) -> bool:
    p = PurePosixPath(path)
    if len(p.parts) != 2 or p.suffix != ".py":
        return False
    return any(p.parts[0] == family and p.name.startswith(prefix) for family, prefix in PLUGIN_FAMILIES.values())


def _validate_rel_path(path: str) -> str:
    p = PurePosixPath(path.replace("\\", "/"))
    if p.is_absolute() or not p.parts or any(part in {"", ".", ".."} for part in p.parts):
        raise PackageError(f"Invalid package file path: {path}")
    if p.parts[0] not in ALLOWED_ROOTS:
        raise PackageError(f"Package file path must start with one of {sorted(ALLOWED_ROOTS)}: {path}")
    return p.as_posix()


def _preflight_collisions(file_bytes: dict[str, bytes]):
    for rel, content in file_bytes.items():
        target = _target(rel)
        # An existing file is fine only if it is byte-identical — a shared
        # plugin-level helper already on disk from another co-owning package.
        # Different content for the same path is a real overwrite, refused.
        if target.exists() and target.read_bytes() != content:
            raise PackageError(f"Refusing to overwrite existing file with different content: {target}")


def _target(rel_path: str) -> Path:
    target = (INSTALLED_PLUGINS / rel_path).resolve()
    root = INSTALLED_PLUGINS.resolve()
    if target != root and root not in target.parents:
        raise PackageError(f"Target escapes installed plugin root: {rel_path}")
    return target


def _receipt_path(package_id: str) -> Path:
    return RECEIPTS_DIR / f"{package_id}.json"


def _write_receipt(receipt: dict):
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    _receipt_path(receipt["id"]).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def _load_receipt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_empty_dirs():
    root = INSTALLED_PLUGINS
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _unique(items) -> list:
    return list(dict.fromkeys(items))


def _validate_package_id(package_id: str):
    if not isinstance(package_id, str) or not PACKAGE_ID_RE.match(package_id):
        raise PackageError(f"Invalid package id: {package_id!r}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
