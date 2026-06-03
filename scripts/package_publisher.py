"""Publish packages to the remote package-store branch."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.commands.helpers import package_manager

SKIP_DIRS = {".git", "__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


class StorePublishError(RuntimeError):
    """Raised when package publishing cannot proceed."""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "validate":
            if args.path:
                validate_store(Path(args.path))
            else:
                with_store_worktree(args.remote, args.branch, lambda path: validate_store(path))
            print("Store valid.")
            return 0
        publish_package(args)
        return 0
    except StorePublishError as e:
        print(f"Package publish failed: {e}", file=sys.stderr)
        return 1


def publish_package(args) -> None:
    def publish(worktree: Path):
        manifest = write_package(
            worktree,
            package_id=args.package_id,
            name=args.name,
            description=args.description,
            file_specs=args.file,
            requires=args.require,
            tags=args.tag,
            entrypoints=[] if args.no_entrypoints else args.entrypoint,
            update=args.update,
        )
        validate_store(worktree)
        status = git(worktree, "status", "--short")
        if not status:
            print(f"No changes for {args.package_id}.")
            return
        print(status)
        if args.dry_run:
            print("Dry run: not committing or pushing.")
            return
        git(worktree, "add", "packages")
        message = args.message or f"Publish package {args.package_id}"
        git(worktree, "commit", "-m", message)
        git(worktree, "push", args.remote, f"HEAD:refs/heads/{args.branch}")
        print(f"Published {manifest['id']} to {args.remote}/{args.branch}.")

    with_store_worktree(args.remote, args.branch, publish)


def write_package(
    store_root: Path,
    *,
    package_id: str,
    name: str,
    description: str,
    file_specs: list[str],
    requires: list[str],
    tags: list[str],
    entrypoints: list[str] | None,
    update: bool,
) -> dict:
    package_manager._validate_package_id(package_id)
    package_dir = store_root / "packages" / package_id
    if package_dir.exists():
        if not update:
            raise StorePublishError(f"Package already exists: {package_id}. Use --update to replace it.")
        shutil.rmtree(package_dir)
    files = expand_file_specs(file_specs)
    if not files and not requires:
        raise StorePublishError("A package needs at least one --file, or --require for a meta-package bundle.")
    package_files = package_dir / "files"
    for source, dest in files:
        target = package_files / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "id": package_id,
        "name": name,
        "description": description,
        "requires": sorted(set(requires)),
        "files": [dest.as_posix() for _source, dest in files],
    }
    if entrypoints is not None:
        manifest["entrypoints"] = [package_manager._validate_rel_path(path) for path in entrypoints]
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_index(store_root, package_id, name, description, tags)
    return manifest


def update_index(store_root: Path, package_id: str, name: str, description: str, tags: list[str]) -> None:
    index_path = store_root / "packages" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        items = data.get("packages", data) if isinstance(data, dict) else data
    else:
        items = []
    items = [item for item in items if item.get("id") != package_id]
    items.append({"id": package_id, "name": name, "description": description, "tags": sorted(set(tags))})
    items.sort(key=lambda item: item["id"])
    index_path.write_text(json.dumps({"packages": items}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expand_file_specs(specs: list[str]) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        source, dest = parse_file_spec(spec)
        if not source.exists():
            raise StorePublishError(f"Source does not exist: {source}")
        if source.is_dir():
            for child in sorted(p for p in source.rglob("*") if p.is_file() and not _skipped(p)):
                rel = child.relative_to(source).as_posix()
                files.append(_file_pair(child, f"{dest.as_posix().rstrip('/')}/{rel}", seen))
        else:
            target = dest / source.name if dest.as_posix().endswith("/") else dest
            files.append(_file_pair(source, target.as_posix(), seen))
    return files


def parse_file_spec(spec: str) -> tuple[Path, Path]:
    if "=" in spec:
        source, dest = spec.split("=", 1)
    elif ":" in spec and not (len(spec) > 1 and spec[1] == ":"):
        source, dest = spec.rsplit(":", 1)
    else:
        raise StorePublishError("File specs must use SOURCE=DEST.")
    return (ROOT / source).resolve() if not Path(source).is_absolute() else Path(source).resolve(), Path(package_manager._validate_rel_path(dest))


def validate_store(store_root: Path) -> None:
    packages_dir = store_root / "packages"
    index_path = packages_dir / "index.json"
    if not index_path.exists():
        raise StorePublishError("Missing packages/index.json.")
    data = json.loads(index_path.read_text(encoding="utf-8"))
    items = data.get("packages", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise StorePublishError("packages/index.json must contain a package list.")
    ids = [item.get("id") for item in items]
    if len(ids) != len(set(ids)):
        raise StorePublishError("Duplicate package id in index.")
    indexed = set(ids)
    actual = {path.name for path in packages_dir.iterdir() if path.is_dir()}
    if indexed != actual:
        raise StorePublishError(f"Index/package dirs differ. Index={sorted(indexed)} dirs={sorted(actual)}")
    manifests = {}
    for package_id in sorted(indexed):
        package_manager._validate_package_id(package_id)
        manifest_path = packages_dir / package_id / "manifest.json"
        if not manifest_path.exists():
            raise StorePublishError(f"Missing manifest for {package_id}.")
        manifest = package_manager._validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        if manifest["id"] != package_id:
            raise StorePublishError(f"Manifest id mismatch in {package_id}.")
        if package_id in manifest["requires"]:
            raise StorePublishError(f"Package requires itself: {package_id}.")
        files = package_manager._validated_files(manifest)
        package_manager._entrypoints(manifest, files)
        listed = set(files)
        actual_files = {
            path.relative_to(packages_dir / package_id / "files").as_posix()
            for path in (packages_dir / package_id / "files").rglob("*")
            if path.is_file()
        }
        if listed != actual_files:
            raise StorePublishError(f"Manifest files differ for {package_id}. Manifest={sorted(listed)} files={sorted(actual_files)}")
        manifests[package_id] = manifest
    for package_id, manifest in manifests.items():
        missing = [dep for dep in manifest["requires"] if dep not in manifests]
        if missing:
            raise StorePublishError(f"{package_id} requires missing package(s): {', '.join(missing)}")
    _check_cycles(manifests)


def with_store_worktree(remote: str, branch: str, callback) -> None:
    git(ROOT, "fetch", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}")
    ref = f"refs/remotes/{remote}/{branch}"
    with tempfile.TemporaryDirectory(prefix="second-brain-store-") as tmp:
        worktree = Path(tmp) / "store"
        git(ROOT, "worktree", "add", "--detach", str(worktree), ref)
        try:
            callback(worktree)
        finally:
            git(ROOT, "worktree", "remove", "--force", str(worktree))


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(cwd), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise StorePublishError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _file_pair(source: Path, dest: str, seen: set[str]) -> tuple[Path, Path]:
    dest = package_manager._validate_rel_path(dest)
    if dest in seen:
        raise StorePublishError(f"Duplicate destination path: {dest}")
    seen.add(dest)
    return source, Path(dest)


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts) or path.suffix in SKIP_SUFFIXES


def _check_cycles(manifests: dict[str, dict]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(package_id: str):
        if package_id in visiting:
            raise StorePublishError(f"Dependency cycle includes {package_id}.")
        if package_id in visited:
            return
        visiting.add(package_id)
        for dep in manifests[package_id]["requires"]:
            visit(dep)
        visiting.remove(package_id)
        visited.add(package_id)

    for package_id in manifests:
        visit(package_id)


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description="Publish Second Brain packages to origin/store.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate the store branch or a checked-out store path.")
    validate.add_argument("--remote", default="origin")
    validate.add_argument("--branch", default="store")
    validate.add_argument("--path")
    publish = sub.add_parser("publish", help="Write one package, commit it, and push it.")
    publish.add_argument("package_id")
    publish.add_argument("--name", required=True)
    publish.add_argument("--description", required=True)
    publish.add_argument("--file", action="append", default=[], help="SOURCE=DEST, repeatable. Directories are copied recursively. Omit for a meta-package bundle (requires-only).")
    publish.add_argument("--require", action="append", default=[])
    publish.add_argument("--tag", action="append", default=[])
    publish.add_argument("--entrypoint", action="append", default=None)
    publish.add_argument("--no-entrypoints", action="store_true", help="Write entrypoints: [] for file-only packages.")
    publish.add_argument("--update", action="store_true")
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--message")
    publish.add_argument("--remote", default="origin")
    publish.add_argument("--branch", default="store")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
