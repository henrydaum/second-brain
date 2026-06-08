"""One-time migration: empty the store's ``helpers/`` family folder.

Shared/standalone helper packages become **plugin-level helpers**:

- ``mcp_oauth`` (only ``service_mcp`` uses it) is folded into ``service_mcp``.
- ``email_context`` is co-owned by all four email tools — each ships its own
  identical copy under ``tools/helpers/``.
- ``search_result`` (``tools/helpers/SearchResult.py``) is co-owned by
  ``tool_lexical_search`` and ``tool_semantic_search``.
- the ``parse_*`` parser helpers (loaded by the kernel parser service, with no
  installable plugin host) move into a ``service_parser/`` store folder that
  ships only the helper files — no entrypoint. They still install to
  ``services/helpers/parse_*.py`` and are discovered there.

Every package's ``requires`` drops the now-inlined helper ids. The installer
co-owns byte-identical files and reference-counts them on uninstall, so the
duplicated helper copies are safe. Runs in a throwaway worktree of
``origin/store``; never touches your checkout or local installed plugins.

    python scripts/migrate_store_helpers.py --dry-run
    python scripts/migrate_store_helpers.py
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

# helper id -> (install rel path, non-bundle consumer ids that ship a copy)
INLINE_HELPERS = {
    "mcp_oauth": ("services/helpers/mcp_oauth.py", ["service_mcp"]),
    "email_context": ("tools/helpers/email_context.py",
                      ["tool_email_check", "tool_email_mark_read", "tool_email_modify_labels", "tool_email_send"]),
    "search_result": ("tools/helpers/SearchResult.py",
                      ["tool_lexical_search", "tool_semantic_search"]),
}
# Parser helper packages relocated out of helpers/ into their own store folder.
PARSER_IDS = ["parse_pdf", "parse_office", "parse_audio", "parse_video", "parse_gdoc", "parse_tabular"]
PARSER_FAMILY = "service_parser"
HELPER_IDS = set(INLINE_HELPERS)


def _read_manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict) -> None:
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def migrate(store: Path) -> list[str]:
    dirs = _scan_package_dirs(store)  # {id: package_dir}
    log: list[str] = []

    # 1. Inline each standalone helper into its consumer packages, then delete it.
    for helper_id, (rel, consumers) in INLINE_HELPERS.items():
        helper_bytes = (dirs[helper_id] / "files" / rel).read_bytes()
        for consumer in consumers:
            pkg = dirs[consumer]
            (pkg / "files" / rel).parent.mkdir(parents=True, exist_ok=True)
            (pkg / "files" / rel).write_bytes(helper_bytes)
            manifest = _read_manifest(pkg)
            manifest["files"] = sorted(set(manifest.get("files", []) + [rel]))
            _write_manifest(pkg, manifest)
            log.append(f"inlined {helper_id} -> {consumer}")
        shutil.rmtree(dirs[helper_id])
        log.append(f"removed standalone helper package {helper_id}")

    # 2. Relocate parser helper packages into the service_parser/ folder.
    (store / PARSER_FAMILY).mkdir(exist_ok=True)
    for pid in PARSER_IDS:
        dst = store / PARSER_FAMILY / pid
        shutil.move(str(dirs[pid]), str(dst))
        dirs[pid] = dst
        log.append(f"relocated {pid} -> {PARSER_FAMILY}/")

    # 3. Drop every inlined helper id from all remaining manifests' requires.
    for pid, pkg in _scan_package_dirs(store).items():
        manifest = _read_manifest(pkg)
        kept = [r for r in manifest.get("requires", []) if r not in HELPER_IDS]
        if kept != manifest.get("requires", []):
            manifest["requires"] = kept
            _write_manifest(pkg, manifest)
            log.append(f"dropped helper requires from {pid}")

    # 4. Remove the now-empty helpers/ family folder.
    helpers_dir = store / "helpers"
    if helpers_dir.exists() and not any(helpers_dir.iterdir()):
        helpers_dir.rmdir()

    _rebuild_index(store)
    return log


def _rebuild_index(store: Path) -> None:
    items = []
    for pid, pkg in _scan_package_dirs(store).items():
        manifest = _read_manifest(pkg)
        items.append({
            "id": pid,
            "name": manifest.get("name", pid),
            "description": manifest.get("description", ""),
            "family": pkg.parent.name,
        })
    items.sort(key=lambda item: item["id"])
    (store / "index.json").write_text(json.dumps({"packages": items}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args) -> None:
    def work(worktree: Path):
        for line in migrate(worktree):
            print(f"  {line}")
        validate_store(worktree)
        status = git(worktree, "status", "--short")
        if not status:
            print("No changes.")
            return
        if args.dry_run:
            print("\nStore valid. Dry run: not committing or pushing.")
            return
        git(worktree, "add", "-A")
        git(worktree, "commit", "-m", "Inline standalone helpers into plugins; empty helpers/ folder")
        git(worktree, "push", args.remote, f"HEAD:refs/heads/{args.branch}")
        print(f"\nPushed to {args.remote}/{args.branch}.")

    with_store_worktree(args.remote, args.branch, work)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inline store helpers into the plugins that need them.")
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
