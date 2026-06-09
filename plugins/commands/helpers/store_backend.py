"""Read package-store files from a git ref."""

from __future__ import annotations

import subprocess
from pathlib import Path


class StoreBackendError(RuntimeError):
    """Raised when the package store cannot be read."""


class GitStoreBackend:
    """Git-ref-backed store reader."""

    def __init__(self, root_dir: str | Path, ref: str = "origin/store"):
        self.root_dir = Path(root_dir)
        self.ref = ref

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
