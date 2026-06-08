"""Read package-store files from a git ref."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class StoreBackendError(RuntimeError):
    """Raised when the package store cannot be read."""


class GitStoreBackend:
    """Git-ref-backed store reader."""

    def __init__(self, root_dir: str | Path, ref: str = "origin/store"):
        self.root_dir = Path(root_dir)
        self.ref = ref
        self._family_map: dict[str, str] | None = None

    def get_index(self) -> list[dict]:
        data = json.loads(self._show_text("index.json"))
        return data.get("packages", data) if isinstance(data, dict) else data

    def _families(self) -> dict[str, str]:
        """Cached ``{package_id: family}`` from the index (family may be absent)."""
        if self._family_map is None:
            try:
                index = self.get_index()
            except StoreBackendError:
                index = []  # no index → every package resolves via the legacy path
            self._family_map = {
                entry["id"]: entry["family"]
                for entry in index
                if entry.get("id") and entry.get("family")
            }
        return self._family_map

    def _package_dir(self, package_id: str) -> str:
        """Locate a package's directory on the ref.

        Packages live at the store root grouped by family (``<family>/<id>``);
        a flat ``<id>`` is used when the index carries no ``family`` for the
        package, so this reader tolerates an unfamilied entry."""
        family = self._families().get(package_id)
        return f"{family}/{package_id}" if family else package_id

    def get_manifest(self, package_id: str) -> dict:
        return json.loads(self._show_text(f"{self._package_dir(package_id)}/manifest.json"))

    def get_manifest_bytes(self, package_id: str) -> bytes:
        return self.get_file_bytes(package_id, "manifest.json", base="")

    def get_file_bytes(self, package_id: str, rel_path: str, base: str = "files") -> bytes:
        prefix = f"{self._package_dir(package_id)}/"
        path = prefix + (f"{base.strip('/')}/" if base else "") + rel_path.replace("\\", "/")
        return self._show_bytes(path)

    def list_tree_files(self) -> list[str]:
        """Return every file path on the store ref."""
        proc = subprocess.run(
            ["git", "-C", str(self.root_dir), "ls-tree", "-r", "--name-only", self.ref],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode:
            raise StoreBackendError(proc.stderr.strip() or f"Could not list {self.ref}")
        return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]

    def list_python_files(self) -> list[str]:
        """Return tree-store plugin/helper Python files."""
        from plugins.commands.helpers import package_manager
        return [path for path in self.list_tree_files() if path.endswith(".py") and package_manager._is_valid_tree_rel(path)]

    def get_tree_file_bytes(self, rel_path: str) -> bytes:
        """Read a file from the tree-mirrored store layout."""
        return self._show_bytes(rel_path.replace("\\", "/"))

    def _show_text(self, path: str) -> str:
        return self._show_bytes(path).decode("utf-8")

    def _show_bytes(self, path: str) -> bytes:
        ref_path = f"{self.ref}:{path}"
        proc = subprocess.run(
            ["git", "-C", str(self.root_dir), "show", ref_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise StoreBackendError(f"Could not read {ref_path}: {err}")
        return proc.stdout
