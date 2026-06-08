"""One-time migration of the package-store branch to plugin-matched ids.

Renames every package on ``origin/store`` so its id matches the plugin/helper
module it ships, with underscores instead of hyphens:

- bundles gain a ``bundle_`` prefix (``all-parsers`` -> ``bundle_all_parsers``);
- helper packages drop their descriptive prefix and are named after the helper
  (``parser-pdf`` -> ``parse_pdf``, ``helper-search-result`` -> ``search_result``);
- ``parser-container`` is renamed to the task it actually ships
  (``task_extract_container``);
- every other package is just hyphen->underscore (``service-litellm`` ->
  ``service_litellm``).

Family subfolders are preserved; only the leaf dir, each manifest's ``id``/
``requires``, and ``index.json`` change. Runs in a throwaway worktree of
``origin/store`` and never touches your checkout or your local installed plugins.

    python scripts/migrate_store_ids.py --dry-run    # show the map + validate
    python scripts/migrate_store_ids.py              # commit + push
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.package_publisher import (  # noqa: E402
    StorePublishError,
    _scan_package_dirs,
    git,
    validate_store,
    with_store_worktree,
)

# Ids that don't follow a mechanical rule (a package named after a concept rather
# than the plugin it ships). Everything else is derived in ``new_id``.
SPECIAL = {"parser-container": "task_extract_container"}


def new_id(old: str, family: str) -> str:
    if old in SPECIAL:
        return SPECIAL[old]
    if family == "bundles":
        return "bundle_" + old.replace("-", "_")
    if old.startswith("parser-"):
        return "parse_" + old[len("parser-"):].replace("-", "_")
    if old.startswith("helper-"):
        return old[len("helper-"):].replace("-", "_")
    return old.replace("-", "_")


def build_map(store_root: Path) -> dict[str, str]:
    index = json.loads((store_root / "index.json").read_text(encoding="utf-8"))
    items = index.get("packages", index) if isinstance(index, dict) else index
    families = {item["id"]: item.get("family", "") for item in items}
    mapping = {old: new_id(old, fam) for old, fam in families.items()}
    collisions = [v for v in set(mapping.values()) if list(mapping.values()).count(v) > 1]
    if collisions:
        raise StorePublishError(f"Rename map is not injective; collisions: {sorted(collisions)}")
    return mapping


def migrate(store_root: Path, mapping: dict[str, str]) -> None:
    dirs = _scan_package_dirs(store_root)  # {old_id: package_dir}
    # Move leaf dirs first (within the same family folder), then rewrite manifests.
    for old, new in mapping.items():
        if old == new:
            continue
        src = dirs[old]
        dst = src.parent / new
        if dst.exists():
            raise StorePublishError(f"Target dir already exists: {dst}")
        shutil.move(str(src), str(dst))
    for old, new in mapping.items():
        family = dirs[old].parent.name
        manifest_path = store_root / family / new / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["id"] = new
        manifest["requires"] = sorted(mapping[dep] for dep in manifest.get("requires", []))
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rebuild_index(store_root)


def _rebuild_index(store_root: Path) -> None:
    items = []
    for pid, package_dir in _scan_package_dirs(store_root).items():
        manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
        items.append({
            "id": pid,
            "name": manifest.get("name", pid),
            "description": manifest.get("description", ""),
            "family": package_dir.parent.name,
        })
    items.sort(key=lambda item: item["id"])
    (store_root / "index.json").write_text(json.dumps({"packages": items}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args) -> None:
    def work(worktree: Path):
        mapping = build_map(worktree)
        renames = {old: new for old, new in mapping.items() if old != new}
        print(f"{len(renames)} package(s) to rename:")
        for old, new in sorted(renames.items()):
            print(f"  {old}  ->  {new}")
        migrate(worktree, mapping)
        validate_store(worktree)
        status = git(worktree, "status", "--short")
        if not status:
            print("No changes.")
            return
        if args.dry_run:
            print("\nStore valid. Dry run: not committing or pushing.")
            return
        git(worktree, "add", "-A")
        git(worktree, "commit", "-m", "Migrate store ids to plugin-matched names")
        git(worktree, "push", args.remote, f"HEAD:refs/heads/{args.branch}")
        print(f"\nPushed migrated store to {args.remote}/{args.branch}.")

    with_store_worktree(args.remote, args.branch, work)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rename store packages to plugin-matched ids.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="store")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except StorePublishError as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
