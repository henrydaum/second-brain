"""Read package-store files from a local git branch."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


class StoreBackendError(RuntimeError):
    """Raised when the package store cannot be read."""


class GitStoreBackend:
    """Branch-backed store reader."""

    def __init__(self, root_dir: str | Path, ref: str = "store"):
        self.root_dir = Path(root_dir)
        self.ref = ref

    def get_index(self) -> list[dict]:
        data = json.loads(self._show_text("packages/index.json"))
        return data.get("packages", data) if isinstance(data, dict) else data

    def get_manifest(self, package_id: str) -> dict:
        return json.loads(self._show_text(f"packages/{package_id}/manifest.json"))

    def get_manifest_bytes(self, package_id: str) -> bytes:
        return self.get_file_bytes(package_id, "manifest.json", base="")

    def get_file_bytes(self, package_id: str, rel_path: str, base: str = "files") -> bytes:
        prefix = f"packages/{package_id}/"
        path = prefix + (f"{base.strip('/')}/" if base else "") + rel_path.replace("\\", "/")
        return self._show_bytes(path)

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
