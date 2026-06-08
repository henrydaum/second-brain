"""Publish files to the tree-based package-store branch."""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.commands.helpers import package_manager  # noqa: E402

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
                with_store_worktree(args.remote, args.branch, validate_store)
            print("Store valid.")
            return 0
        publish_package(args)
        return 0
    except (StorePublishError, package_manager.PackageError) as e:
        print(f"Package publish failed: {e}", file=sys.stderr)
        return 1


def publish_package(args) -> None:
    def publish(worktree: Path):
        files = write_package(
            worktree,
            package_id=args.package_id,
            name=args.name,
            description=args.description,
            file_specs=args.file,
            requires=args.require,
            entrypoints=[] if args.no_entrypoints else args.entrypoint,
            pip=[] if args.no_pip else args.pip,
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
        git(worktree, "add", "-A")
        git(worktree, "commit", "-m", args.message or f"Publish {args.package_id}")
        git(worktree, "push", args.remote, f"HEAD:refs/heads/{args.branch}")
        print(f"Published {args.package_id}: {', '.join(files)}")

    with_store_worktree(args.remote, args.branch, publish)


def write_package(
    store_root: Path,
    *,
    package_id: str,
    name: str = "",
    description: str = "",
    file_specs: list[str],
    requires: list[str],
    entrypoints: list[str] | None = None,
    update: bool,
    pip: list[str] | None = None,
) -> list[str]:
    package_manager._validate_package_id(package_id)
    files = expand_file_specs(file_specs)
    if not files:
        raise StorePublishError("A tree-store package needs at least one --file.")
    written: list[str] = []
    for source, dest in files:
        target = store_root / dest
        if target.exists() and not update and target.read_bytes() != source.read_bytes():
            raise StorePublishError(f"Store file already exists with different bytes: {dest}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written.append(dest.as_posix())
    primary = _primary_file(package_id, written)
    if requires or pip is not None:
        _write_deps(store_root / primary, requires, [] if pip is None else pip)
    return written


def validate_store(store_root: Path) -> None:
    files = _tree_files(store_root)
    stems: dict[str, str] = {}
    for rel in files:
        stem = Path(rel).stem
        if stem in stems:
            raise StorePublishError(f"Duplicate install stem {stem}: {stems[stem]}, {rel}")
        stems[stem] = rel
    for rel in files:
        meta = package_manager.read_dependency_meta(rel, (store_root / rel).read_text(encoding="utf-8"))
        for dep in meta.dependencies_files:
            if dep not in files:
                raise StorePublishError(f"{rel} depends on missing file: {dep}")


def expand_file_specs(specs: list[str]) -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        source, dest = parse_file_spec(spec)
        if not source.exists():
            raise StorePublishError(f"Source does not exist: {source}")
        if source.is_dir():
            for child in sorted(p for p in source.rglob("*") if p.is_file() and not _skipped(p)):
                files.append(_file_pair(child, f"{dest.as_posix().rstrip('/')}/{child.relative_to(source).as_posix()}", seen))
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
    return (ROOT / source).resolve() if not Path(source).is_absolute() else Path(source).resolve(), Path(_validate_package_dest(dest))


def with_store_worktree(remote: str, branch: str, callback) -> None:
    git(ROOT, "fetch", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}")
    with tempfile.TemporaryDirectory(prefix="second-brain-store-") as tmp:
        worktree = Path(tmp) / "store"
        git(ROOT, "worktree", "add", "--detach", str(worktree), f"refs/remotes/{remote}/{branch}")
        try:
            callback(worktree)
        finally:
            git(ROOT, "worktree", "remove", "--force", str(worktree))


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(cwd), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise StorePublishError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _tree_files(store_root: Path) -> set[str]:
    return {
        path.relative_to(store_root).as_posix()
        for path in store_root.rglob("*.py")
        if path.is_file() and package_manager._is_valid_tree_rel(path.relative_to(store_root).as_posix())
    }


def _file_pair(source: Path, dest: str, seen: set[str]) -> tuple[Path, Path]:
    dest = package_manager._validate_rel_path(dest)
    if dest in seen:
        raise StorePublishError(f"Duplicate destination path: {dest}")
    seen.add(dest)
    return source, Path(dest)


def _validate_package_dest(path: str) -> str:
    p = Path(path.replace("\\", "/"))
    if p.is_absolute() or not p.parts or any(part in {"", ".", ".."} for part in p.parts):
        raise StorePublishError(f"Invalid package destination: {path}")
    if p.suffix == ".py":
        return package_manager._validate_rel_path(path)
    if p.parts[0] not in package_manager.TREE_ROOTS:
        raise StorePublishError(f"Package destination must start with one of {sorted(package_manager.TREE_ROOTS)}: {path}")
    return p.as_posix()


def _primary_file(package_id: str, files: list[str]) -> str:
    matches = [rel for rel in files if Path(rel).stem == package_id]
    if len(matches) == 1:
        return matches[0]
    if not matches and len(files) == 1:
        return files[0]
    raise StorePublishError(f"Could not choose primary file for {package_id}.")


def _write_deps(path: Path, deps_files: list[str], deps_pip: list[str]) -> None:
    deps_files = sorted(dict.fromkeys(package_manager._validate_rel_path(path) for path in deps_files))
    deps_pip = sorted(dict.fromkeys(deps_pip))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    remove = {
        node.lineno
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(getattr(t, "id", None) in package_manager.DEPENDENCY_FIELDS for t in getattr(node, "targets", [getattr(node, "target", None)]))
    }
    lines = [line for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if i not in remove]
    insert = 1
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        insert = tree.body[0].end_lineno + 1
    while insert <= len(lines) and not lines[insert - 1].strip():
        insert += 1
    while insert <= len(lines) and lines[insert - 1].startswith("from __future__ import "):
        insert += 1
    while insert <= len(lines) and not lines[insert - 1].strip():
        insert += 1
    block = [f"dependencies_files = {deps_files!r}", f"dependencies_pip = {deps_pip!r}", ""]
    lines[insert - 1:insert - 1] = block
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _skipped(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts) or path.suffix in SKIP_SUFFIXES


def _scan_package_dirs(store_root: Path) -> dict[str, Path]:
    return {Path(rel).stem: store_root / rel for rel in _tree_files(store_root)}


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description="Publish Second Brain package files to origin/store.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate the store branch or a checked-out store path.")
    validate.add_argument("--remote", default="origin")
    validate.add_argument("--branch", default="store")
    validate.add_argument("--path")
    publish = sub.add_parser("publish", help="Copy files into the store tree, commit, and push.")
    publish.add_argument("package_id", help="Stem of the primary plugin/helper file.")
    publish.add_argument("--name", default="")
    publish.add_argument("--description", default="")
    publish.add_argument("--file", action="append", default=[], help="SOURCE=DEST, repeatable. Directories are copied recursively.")
    publish.add_argument("--require", action="append", default=[], help="Store-relative .py dependency file, repeatable.")
    publish.add_argument("--entrypoint", action="append", default=None, help="Ignored; kept for old command lines.")
    publish.add_argument("--no-entrypoints", action="store_true", help="Ignored; tree store uses file stems.")
    publish.add_argument("--pip", action="append", default=None, help="PyPI dependency, repeatable.")
    publish.add_argument("--no-pip", action="store_true", help="Declare dependencies_pip = [].")
    publish.add_argument("--update", action="store_true")
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--message")
    publish.add_argument("--remote", default="origin")
    publish.add_argument("--branch", default="store")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
