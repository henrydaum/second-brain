"""Shared filesystem-walking helpers for the grep and glob tools.

Walks are confined to the project root and the Second Brain data directory,
skip well-known junk directories, and never follow symlinks or Windows
junctions (reparse points), so a cycle cannot occur.
"""

dependencies_files = []
dependencies_pip = []

import os
import re
import stat
from pathlib import Path

from paths import ROOT_DIR, DATA_DIR

ALLOWED_ROOTS = {Path(ROOT_DIR).resolve(), Path(DATA_DIR).resolve()}

IGNORED_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__",
    ".venv", "venv", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
    "dist", "build", ".idea",
}

MAX_FILE_BYTES = 2_000_000   # grep skips files bigger than this
MAX_SCAN_FILES = 20_000      # enumeration bound per walk


def resolve_root(raw: str | None) -> tuple[Path | None, str | None]:
    """Resolve a user-supplied path against the project root and confine it.

    Returns (path, None) on success or (None, error) when the path escapes
    the allowed roots.
    """
    raw = (raw or "").strip()
    p = Path(raw).expanduser() if raw else Path(ROOT_DIR)
    p = (p if p.is_absolute() else Path(ROOT_DIR) / p).resolve()
    if any(p == root or root in p.parents for root in ALLOWED_ROOTS):
        return p, None
    return None, f"Path is outside the project root and data directory: {p}"


def _is_link(entry_path: str) -> bool:
    """True for symlinks and Windows reparse points (junctions)."""
    try:
        st = os.lstat(entry_path)
    except OSError:
        return True  # unreadable — treat as skippable
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attrs & reparse)


def iter_files(root: Path) -> tuple[list[Path], bool]:
    """Enumerate regular files under ``root``, pruning junk dirs and links.

    Returns (files, scan_truncated) where scan_truncated is True when the
    MAX_SCAN_FILES bound stopped the walk early.
    """
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not _is_link(os.path.join(dirpath, d))
        ]
        for name in filenames:
            full = os.path.join(dirpath, name)
            if _is_link(full):
                continue
            files.append(Path(full))
            if len(files) >= MAX_SCAN_FILES:
                return files, True
    return files, False


def compile_glob(pattern: str) -> re.Pattern:
    """Translate a glob into a regex over '/'-separated relative paths.

    Semantics: ``*`` and ``?`` never cross a path separator; a ``**`` segment
    matches any number of directories (including none). So ``*.py`` matches
    top-level files only, while ``**/*.py`` matches any depth.
    """
    segments = [s for s in pattern.replace("\\", "/").split("/") if s]
    parts: list[str] = []
    for seg in segments:
        if seg == "**":
            parts.append("(?:[^/]+/)*")
            continue
        piece = ""
        for ch in seg:
            if ch == "*":
                piece += "[^/]*"
            elif ch == "?":
                piece += "[^/]"
            else:
                piece += re.escape(ch)
        parts.append(piece + "/")
    body = "".join(parts)
    if body.endswith("/"):
        body = body[:-1]
    return re.compile(f"^{body}$", re.IGNORECASE)


def match_rel(path: Path, root: Path, compiled: re.Pattern) -> bool:
    """Match a compiled glob against ``path`` relative to ``root``."""
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return False
    return compiled.match(rel) is not None


def is_binary(path: Path) -> bool:
    """Null-byte sniff on the first KB; unreadable files count as binary."""
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(1024)
    except OSError:
        return True


def mtime_sorted(paths: list[Path]) -> list[Path]:
    """Newest-first by modification time; unreadable stats sort last."""
    def key(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
    return sorted(paths, key=key, reverse=True)
