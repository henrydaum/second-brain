"""Install and uninstall packages into DATA_DIR/installed_plugins."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from paths import INSTALLED_PLUGINS, PACKAGES_DIR
from plugins.helpers.plugin_paths import PLUGIN_FAMILIES, plugin_dirs, plugin_info
from plugins.plugin_discovery import load_single_plugin, unload_plugin
from plugins.commands.helpers.store_backend import GitStoreBackend


PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ALLOWED_ROOTS = {family for family, _prefix in PLUGIN_FAMILIES.values()} | {"helpers"}
RECEIPTS_DIR = PACKAGES_DIR / "receipts"
INTERNAL_IMPORTS = {"agent", "config", "events", "helpers", "installed_plugins", "paths", "pipeline", "plugins", "runtime", "sandbox_plugins", "state_machine", "templates", *ALLOWED_ROOTS}
PIP_NAMES = {"PIL": "Pillow", "bs4": "beautifulsoup4", "cv2": "opencv-python", "docx": "python-docx", "fitz": "PyMuPDF", "googleapiclient": "google-api-python-client", "pptx": "python-pptx", "sklearn": "scikit-learn", "yaml": "PyYAML"}


class PackageError(RuntimeError):
    """Raised for package validation or install failures."""


@dataclass
class PackageActionResult:
    """Structured package operation result."""
    ok: bool
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines) if self.lines else ("OK" if self.ok else "Failed")


def search_packages(root_dir: str | Path, query: str = "") -> list[dict]:
    """Return packages from the store index matching query."""
    items = GitStoreBackend(root_dir).get_index()
    q = (query or "").strip().lower()
    if not q:
        return sorted(items, key=lambda item: item.get("id", ""))
    def hay(item):
        return " ".join(str(item.get(k, "")) for k in ("id", "name", "description", "tags")).lower()
    return sorted([item for item in items if q in hay(item)], key=lambda item: item.get("id", ""))


def installed_packages() -> list[dict]:
    """Return installed package receipts."""
    return sorted((_load_receipt(path) for path in RECEIPTS_DIR.glob("*.json")), key=lambda receipt: receipt.get("id", "")) if RECEIPTS_DIR.exists() else []


def package_info(root_dir: str | Path, package_id: str) -> dict:
    """Return one package manifest."""
    _validate_package_id(package_id)
    manifest = GitStoreBackend(root_dir).get_manifest(package_id)
    return _validate_manifest(manifest)


def install_package(root_dir: str | Path, package_id: str, context=None, *, requested: bool = True) -> PackageActionResult:
    """Install a package and its dependencies."""
    backend = GitStoreBackend(root_dir)
    lines: list[str] = []
    installed: list[str] = []
    _install(package_id, backend, context, requested=requested, active=set(), installed=installed, lines=lines)
    if not installed:
        lines.append("Nothing installed.")
    return PackageActionResult(True, lines)


def uninstall_package(package_id: str, context=None) -> PackageActionResult:
    """Uninstall a package and prune unneeded auto-installed dependencies."""
    lines: list[str] = []
    pruned: list[str] = []
    _uninstall(package_id, context, lines=lines, pruned=pruned, explicit=True)
    if pruned:
        lines.append(f"Pruned auto-installed dependencies: {', '.join(pruned)}")
    return PackageActionResult(True, lines)


def _install(package_id: str, backend, context, *, requested: bool, active: set[str], installed: list[str], lines: list[str]):
    _validate_package_id(package_id)
    if package_id in active:
        raise PackageError(f"Dependency cycle includes {package_id}.")
    existing = _receipt_path(package_id)
    if existing.exists():
        if requested:
            raise PackageError(f"Package already installed: {package_id}")
        lines.append(f"Dependency already installed: {package_id}")
        return
    active.add(package_id)
    try:
        manifest = _validate_manifest(backend.get_manifest(package_id))
        if manifest["id"] != package_id:
            raise PackageError(f"Manifest id mismatch: requested {package_id}, got {manifest['id']}.")
        for dep in manifest["requires"]:
            _install(dep, backend, context, requested=False, active=active, installed=installed, lines=lines)

        files = _validated_files(manifest)
        entrypoints = _entrypoints(manifest, files)
        file_bytes = {rel: backend.get_file_bytes(package_id, rel) for rel in files}
        _preflight_collisions(package_id, file_bytes)

        written = []
        try:
            for rel, content in file_bytes.items():
                target = _target(rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                written.append(target)
            pip_installed = _install_missing_imports(file_bytes)
            loaded = _load_entrypoints(entrypoints, context)
            _reload_parser_service_if_needed(file_bytes, context, lines)
            receipt = {
                "id": package_id,
                "name": manifest.get("name", package_id),
                "description": manifest.get("description", ""),
                "installed_at": time.time(),
                "requested": bool(requested),
                "requires": manifest["requires"],
                "manifest_hash": _sha256(backend.get_manifest_bytes(package_id)),
                "files": [{"path": rel, "sha256": _sha256(content)} for rel, content in file_bytes.items()],
                "entrypoints": loaded,
                "pip_packages": pip_installed,
            }
            _write_receipt(receipt)
        except Exception:
            for target in reversed(written):
                target.unlink(missing_ok=True)
            _remove_empty_dirs()
            raise
        installed.append(package_id)
        if pip_installed:
            lines.append(f"Installed Python package(s): {', '.join(pip_installed)}")
        lines.append(f"Installed {package_id}: {len(files)} file(s), {len(loaded)} plugin entrypoint(s).")
    finally:
        active.remove(package_id)


def _uninstall(package_id: str, context, *, lines: list[str], pruned: list[str], explicit: bool):
    _validate_package_id(package_id)
    receipt_path = _receipt_path(package_id)
    if not receipt_path.exists():
        raise PackageError(f"Package is not installed: {package_id}")
    dependents = _dependents(package_id)
    if dependents:
        raise PackageError(f"Cannot uninstall {package_id}; required by: {', '.join(dependents)}")
    receipt = _load_receipt(receipt_path)
    cleanup = _cleanup_plan(receipt)
    approved_cleanup = _approve_cleanup(context, package_id, cleanup, lines)
    for ep in receipt.get("entrypoints", []):
        _unload_entrypoint(ep, context)
    removed_paths = [item["path"] for item in receipt.get("files", [])]
    for item in sorted(receipt.get("files", []), key=lambda f: f.get("path", ""), reverse=True):
        target = _target(item["path"])
        target.unlink(missing_ok=True)
    receipt_path.unlink(missing_ok=True)
    _remove_empty_dirs()
    _reload_parser_service_if_needed(removed_paths, context, lines)
    if approved_cleanup:
        _apply_cleanup(context, cleanup, lines)
    lines.append(f"Uninstalled {package_id}.")

    for dep in receipt.get("requires", []):
        dep_path = _receipt_path(dep)
        if not dep_path.exists() or _dependents(dep):
            continue
        dep_receipt = _load_receipt(dep_path)
        if dep_receipt.get("requested"):
            continue
        _uninstall(dep, context, lines=lines, pruned=pruned, explicit=False)
        pruned.append(dep)


def _load_entrypoints(entrypoints: list[str], context) -> list[dict]:
    loaded = []
    try:
        for rel in entrypoints:
            path = _target(rel)
            info, err = plugin_info(path)
            if err:
                raise PackageError(err)
            name, error = load_single_plugin(
                info.plugin_type,
                path,
                tool_registry=getattr(context, "tool_registry", None),
                orchestrator=getattr(context, "orchestrator", None),
                services=getattr(context, "services", None),
                config=getattr(context, "config", None),
                command_registry=getattr(context, "command_registry", None),
                frontend_manager=getattr(getattr(context, "runtime", None), "frontend_manager", None),
                runtime=getattr(context, "runtime", None),
            )
            if error:
                raise PackageError(f"Failed to load {rel}: {error}")
            loaded.append({"path": rel, "type": info.plugin_type, "name": name or ""})
            if info.plugin_type == "command":
                _refresh_commands(context)
    except Exception:
        for entrypoint in reversed(loaded):
            _unload_entrypoint(entrypoint, context)
        raise
    return loaded


def _unload_entrypoint(entrypoint: dict, context):
    unload_plugin(
        entrypoint.get("type", ""),
        entrypoint.get("name", ""),
        tool_registry=getattr(context, "tool_registry", None),
        orchestrator=getattr(context, "orchestrator", None),
        services=getattr(context, "services", None),
        source_path=str(_target(entrypoint["path"])),
        command_registry=getattr(context, "command_registry", None),
        frontend_manager=getattr(getattr(context, "runtime", None), "frontend_manager", None),
    )
    if entrypoint.get("type") == "command":
        _refresh_commands(context)


def _is_parser_helper(rel: str) -> bool:
    """Whether a package file is a parser-discovery helper (services/helpers/parse_*.py).

    Such files aren't plugin entrypoints — they register (extension, modality)
    parsers when the parser service scans its helper dirs — so installing or
    removing one only takes effect on a parser-service reload.
    """
    p = PurePosixPath(rel)
    return (
        len(p.parts) == 3
        and p.parts[0] == "services"
        and p.parts[1] == "helpers"
        and p.suffix == ".py"
        and p.name.startswith("parse_")
    )


def _reload_parser_service_if_needed(file_rels, context, lines: list[str]) -> None:
    """Reload the parser service so newly written/removed parser helpers take
    effect live, without an app restart. Non-fatal: a reload failure is noted
    but never fails the install/uninstall."""
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


def _refresh_commands(context):
    runtime = getattr(context, "runtime", None)
    registry = getattr(context, "command_registry", None)
    if runtime and registry and hasattr(registry, "to_callable_specs"):
        runtime.commands = registry.to_callable_specs()
    if runtime and hasattr(runtime, "refresh_session_specs"):
        runtime.refresh_session_specs()


def _install_missing_imports(file_bytes: dict[str, bytes]) -> list[str]:
    packages = _missing_pip_packages(file_bytes)
    if not packages:
        return []
    result = subprocess.run([sys.executable, "-m", "pip", "install", *packages], capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise PackageError(f"pip install failed for {', '.join(packages)}:\n{result.stderr or result.stdout}")
    return packages


def _missing_pip_packages(file_bytes: dict[str, bytes]) -> list[str]:
    roots = set()
    for rel, content in file_bytes.items():
        if rel.endswith(".py"):
            roots.update(_import_roots(content))
    stdlib = set(getattr(sys, "stdlib_module_names", set())) | set(sys.builtin_module_names)
    missing = []
    for root in sorted(roots):
        if root in stdlib or root in INTERNAL_IMPORTS or _module_available(root):
            continue
        missing.append(PIP_NAMES.get(root, root))
    return sorted(set(missing), key=str.lower)


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


def _cleanup_plan(receipt: dict) -> dict[str, list[str]]:
    owned_paths = {_target(item["path"]).resolve() for item in receipt.get("files", [])}
    owned = _declared_state(owned_paths)
    used_elsewhere = _declared_state(_other_plugin_files(owned_paths))
    settings = sorted(owned["settings"] - used_elsewhere["settings"])
    tables = sorted(owned["writes"] - (used_elsewhere["reads"] | used_elsewhere["writes"]))
    return {
        "settings": settings,
        "tables": tables,
        "kept_settings": sorted(owned["settings"] - set(settings)),
        "kept_tables": sorted(owned["writes"] - set(tables)),
    }


def _approve_cleanup(context, package_id: str, cleanup: dict, lines: list[str]) -> bool:
    if cleanup["kept_settings"]:
        lines.append(f"Kept config setting(s) still declared by other plugins: {', '.join(cleanup['kept_settings'])}")
    if cleanup["kept_tables"]:
        lines.append(f"Kept table(s) still used by remaining tasks; their data may now be stale: {', '.join(cleanup['kept_tables'])}")
    if not cleanup["settings"] and not cleanup["tables"]:
        return False
    ask = getattr(context, "request_user_input", None)
    if ask is None:
        lines.append("Cleanup available but no approval session is available; kept package config/table data.")
        return False
    prompt = "Delete package-owned data?\n\n"
    if cleanup["settings"]:
        prompt += "Config settings: " + ", ".join(cleanup["settings"]) + "\n"
    if cleanup["tables"]:
        prompt += "Tables: " + ", ".join(cleanup["tables"]) + "\n"
    req = ask(f"Uninstall {package_id}", prompt.strip(), type="boolean")
    if not req.wait(timeout=300.0) or not req.approved:
        lines.append("Kept package config/table data.")
        return False
    return True


def _apply_cleanup(context, cleanup: dict, lines: list[str]) -> None:
    if cleanup["settings"]:
        from config import config_manager
        saved = config_manager.load_plugin_config()
        for key in cleanup["settings"]:
            saved.pop(key, None)
            if getattr(context, "config", None) is not None:
                context.config.pop(key, None)
        config_manager.save_plugin_config(saved)
        lines.append(f"Deleted config setting(s): {', '.join(cleanup['settings'])}")
    if cleanup["tables"] and getattr(context, "db", None) is not None:
        with context.db.lock:
            for table in cleanup["tables"]:
                context.db._validate_identifier(table)
                context.db.conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            context.db.conn.commit()
        lines.append(f"Deleted table(s): {', '.join(cleanup['tables'])}")


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
                assigns = [item for item in getattr(node, "body", []) if isinstance(item, ast.Assign)]
                for assign in assigns:
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
    normalized = dict(manifest)
    normalized["requires"] = requires
    normalized["files"] = [_validate_rel_path(path) for path in files]
    if entrypoints is not None:
        normalized["entrypoints"] = [_validate_rel_path(path) for path in entrypoints]
    return normalized


def _validated_files(manifest: dict) -> list[str]:
    files = manifest["files"]
    if not files:
        raise PackageError("Manifest must include at least one file.")
    return files


def _entrypoints(manifest: dict, files: list[str]) -> list[str]:
    explicit = manifest.get("entrypoints")
    entrypoints = explicit if explicit is not None else [path for path in files if _is_entrypoint(path)]
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
    for family, prefix in PLUGIN_FAMILIES.values():
        if p.parts[0] == family and p.name.startswith(prefix):
            return True
    return False


def _validate_rel_path(path: str) -> str:
    p = PurePosixPath(path.replace("\\", "/"))
    if p.is_absolute() or not p.parts or any(part in {"", ".", ".."} for part in p.parts):
        raise PackageError(f"Invalid package file path: {path}")
    if p.parts[0] not in ALLOWED_ROOTS:
        raise PackageError(f"Package file path must start with one of {sorted(ALLOWED_ROOTS)}: {path}")
    return p.as_posix()


def _preflight_collisions(package_id: str, file_bytes: dict[str, bytes]):
    for rel, content in file_bytes.items():
        target = _target(rel)
        if not target.exists():
            continue
        owner = _file_owner(rel)
        if owner == package_id and _sha256(target.read_bytes()) == _sha256(content):
            continue
        raise PackageError(f"Refusing to overwrite existing file: {target}")


def _file_owner(rel_path: str) -> str | None:
    for receipt in installed_packages():
        if any(item.get("path") == rel_path for item in receipt.get("files", [])):
            return receipt.get("id")
    return None


def _dependents(package_id: str) -> list[str]:
    return sorted(receipt["id"] for receipt in installed_packages() if package_id in (receipt.get("requires") or []))


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


def _validate_package_id(package_id: str):
    if not isinstance(package_id, str) or not PACKAGE_ID_RE.match(package_id):
        raise PackageError(f"Invalid package id: {package_id!r}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
