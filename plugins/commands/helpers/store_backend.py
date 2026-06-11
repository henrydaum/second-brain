"""Read package-store files from a git ref."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger("Store")

# How long a successful-or-attempted fetch keeps the ref "fresh enough".
# Nothing else ever updates origin/store on this machine, so without this
# the store would stay frozen at whatever the user last fetched manually.
_FETCH_TTL_SECONDS = 300.0
_fetch_times: dict[tuple[str, str], float] = {}
_fetch_lock = threading.Lock()


class StoreBackendError(RuntimeError):
    """Raised when the package store cannot be read."""


class GitStoreBackend:
    """Git-ref-backed store reader."""

    def __init__(self, root_dir: str | Path, ref: str = "origin/store"):
        self.root_dir = Path(root_dir)
        self.ref = ref

    def refresh(self, force: bool = False) -> None:
        """Fetch the store branch so the local ref reflects the remote.

        Throttled by a TTL so browsing doesn't shell out to the network on
        every call; failures (offline, no such remote) degrade silently to
        the last-fetched ref.
        """
        if "/" not in self.ref:
            return  # local ref, nothing to fetch
        remote, branch = self.ref.split("/", 1)
        key = (str(self.root_dir), self.ref)
        with _fetch_lock:
            if not force and time.monotonic() - _fetch_times.get(key, 0.0) < _FETCH_TTL_SECONDS:
                return
            # Stamp before fetching so an unreachable remote doesn't retry
            # on every store read.
            _fetch_times[key] = time.monotonic()
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.root_dir), "fetch", "--quiet", remote,
                 f"refs/heads/{branch}:refs/remotes/{remote}/{branch}"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                check=False, timeout=30,
            )
            if proc.returncode:
                logger.debug("Store fetch failed (using local ref): %s", proc.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.debug("Store fetch failed (using local ref): %s", e)

    def resolve_commit(self) -> str | None:
        """Commit hash the store ref currently points at, for install
        provenance. None when the ref can't be resolved (no store yet)."""
        proc = subprocess.run(
            ["git", "-C", str(self.root_dir), "rev-parse", self.ref],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            check=False,
        )
        return proc.stdout.strip() or None if proc.returncode == 0 else None

    def list_tree_files(self) -> list[str]:
        """Return every file path on the store ref."""
        self.refresh()
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
