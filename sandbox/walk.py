"""Tree walking and content search — the engine behind ``fs.list`` and ``fs.search``.

This is host code, and that is the point of it being here. Searching a project
tree properly needs three things sandboxed code cannot have: a walk that prunes
``.git`` and ``node_modules`` without asking about each one, a regex engine
applied to file contents without shipping every file across the wire, and
ripgrep when it happens to be installed. Doing it in the guest costs one round
trip per file, which over a subprocess boundary is not a slow search — it is no
search at all.

So the engine sits behind two existing Requests rather than becoming a new one.
``fs.search`` and ``fs.list`` grow *arguments*; the authorization surface —
which Request types exist, and what ``policy.classify`` says about each — is
untouched. A plugin that could search before can still search, and one that
could not still cannot.

Nothing here decides *whether* a path may be read. That stays with
``protected.py``, consulted by the handlers.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

# Well-known junk. Pruned from every walk: they are large, uninteresting, and
# the reason a naive ``**/*`` over a project takes minutes.
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__",
    ".venv", "venv", ".tox", ".eggs",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache",
    "dist", "build", ".idea",
})

# Content search skips files bigger than this. Deliberately smaller than the
# handler's ``MAX_READ_BYTES``: an 8 MB file is a reasonable thing to *read* on
# purpose and an unreasonable thing to sweep past while grepping a tree.
MAX_FILE_BYTES = 2_000_000

# Enumeration bound per walk. Reaching it is reported as ``scan_truncated``
# rather than failing — a partial answer with a flag beats no answer.
MAX_SCAN_FILES = 20_000

RG_TIMEOUT = 30
MAX_CONTEXT = 10

_UNSET = object()
_rg_cache = _UNSET

# rg content output: match lines ``path:line:text``, context lines
# ``path-line-text``, groups separated by lone ``--`` lines. Normalized to the
# Python path's ``rel:lineno: text`` / ``rel:lineno- text`` format.
_RG_MATCH_RE = re.compile(r"^(.+?):(\d+):(.*)$")
_RG_CONTEXT_RE = re.compile(r"^(.+?)-(\d+)-(.*)$")


# ──────────────────────────────────────────────────────────────────────
# Walking.
# ──────────────────────────────────────────────────────────────────────

def is_link(entry_path) -> bool:
    """True for symlinks and Windows reparse points (junctions).

    Both are how a walk turns into a cycle, so both are skipped. An unreadable
    entry counts as a link: the walk cannot inspect it, and refusing to descend
    is the safe reading of that.
    """
    try:
        st = os.lstat(entry_path)
    except OSError:
        return True
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attrs & reparse)


def iter_files(root: Path) -> tuple[list[Path], bool]:
    """Enumerate regular files under ``root``, pruning junk dirs and links.

    Returns ``(files, scan_truncated)``.
    """
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not is_link(os.path.join(dirpath, d))
        ]
        for name in filenames:
            full = os.path.join(dirpath, name)
            if is_link(full):
                continue
            files.append(Path(full))
            if len(files) >= MAX_SCAN_FILES:
                return files, True
    return files, False


def iter_entries(root: Path) -> tuple[list[Path], bool]:
    """Like :func:`iter_files`, but directories are included too.

    ``fs.list`` answers about a directory's contents, and a recursive listing
    that silently dropped every subdirectory would be a different question than
    the one asked.
    """
    entries: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True,
                                                followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORED_DIRS and not is_link(os.path.join(dirpath, d))
        ]
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            if is_link(full):
                continue
            entries.append(Path(full))
            if len(entries) >= MAX_SCAN_FILES:
                return entries, True
    return entries, False


def compile_glob(pattern: str) -> re.Pattern:
    """Translate a glob into a regex over '/'-separated relative paths.

    ``*`` and ``?`` never cross a path separator; a ``**`` segment matches any
    number of directories including none. So ``*.py`` matches top-level files
    only, while ``**/*.py`` matches any depth — which is the distinction
    ``Path.glob`` blurs and callers keep tripping over.
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


def relative(path: Path, root: Path) -> str:
    """Root-relative posix path for display, falling back to the absolute one."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def is_binary(path: Path) -> bool:
    """Null-byte sniff on the first KB; unreadable files count as binary."""
    try:
        with open(path, "rb") as handle:
            return b"\x00" in handle.read(1024)
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


# ──────────────────────────────────────────────────────────────────────
# ripgrep fast path.
#
# Any rg failure — a missing binary, exit 2 because the Rust regex engine
# rejects a backreference, a timeout, output we cannot parse — returns None and
# the caller falls back to the Python path. So the fast path can never make a
# search fail; at worst it makes one slower.
#
# Accepted divergences from the Python path: rg does its own binary and UTF-16
# detection rather than our null-byte sniff, it has no MAX_SCAN_FILES analogue
# (``scan_truncated`` stays False), and in multiline content mode it reports
# every line of a match rather than just its first.
# ──────────────────────────────────────────────────────────────────────

def rg_path():
    """Path to ripgrep, or None. Resolved once per process."""
    global _rg_cache
    if _rg_cache is _UNSET:
        import shutil
        _rg_cache = shutil.which("rg")
    return _rg_cache


def reset_rg_cache(value=_UNSET):
    """Test seam: forget (or force) the resolved ripgrep path."""
    global _rg_cache
    _rg_cache = value


def run_ripgrep(rg, pattern, root, raw_glob, mode, case_insensitive,
                multiline, context_lines):
    """Run rg and parse its output into the Python path's result shapes.

    Returns the full (unlimited) result list, or None to signal fallback.
    """
    cmd = [rg, "--no-config", "--no-ignore", "--hidden", "--no-messages",
           "--sortr", "modified", "--max-filesize", str(MAX_FILE_BYTES)]
    for d in sorted(IGNORED_DIRS):
        # ripgrep's glob anchoring differs at the search root on Windows: the
        # recursive form prunes nested matches but can leave ``root/d`` alive.
        # Name both shapes, then filter results as a final backend-independent
        # guard in fs_net.
        cmd += ["-g", f"!/{d}/**", "-g", f"!**/{d}/**"]
    if raw_glob:
        g = raw_glob.replace("\\", "/").lstrip("/")
        # Our '*.py' means top level only; a bare rg glob matches basenames at
        # any depth, so anchor it with a leading '/'.
        cmd += ["-g", g if g.startswith("**") else "/" + g,
                "--glob-case-insensitive"]
    if case_insensitive:
        cmd.append("-i")
    if multiline:
        cmd += ["-U", "--multiline-dotall"]
    if mode == "files":
        cmd.append("-l")
    elif mode == "count":
        cmd.append("--count-matches")
    else:
        cmd.append("-n")
        if context_lines:
            cmd += ["-C", str(context_lines)]
    # -e plus '--': a pattern starting with '-' must not parse as a flag.
    cmd += ["-e", pattern, "--"]
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=RG_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if proc.returncode == 1:
        return []
    if proc.returncode != 0:
        return None
    return parse_rg(proc.stdout, mode, with_context=context_lines > 0)


def parse_rg(stdout, mode, with_context=False):
    """Normalize rg stdout into the Python path's result shapes."""
    lines = stdout.splitlines()
    if mode == "files":
        return [_rg_rel(line) for line in lines if line]
    if mode == "count":
        results = []
        for line in lines:
            if not line:
                continue
            path, _, n = line.rpartition(":")
            if not path or not n.isdigit():
                return None
            results.append([_rg_rel(path), int(n)])
        return results
    # Content without context: rg emits no '--' separators, so each match line
    # is its own result — which is exactly the Python path's grouping.
    if not with_context:
        results = []
        for line in lines:
            if not line:
                continue
            m = _RG_MATCH_RE.match(line)
            if not m:
                return None
            results.append(f"{_rg_rel(m.group(1))}:{m.group(2)}: {m.group(3)}")
        return results
    # Content with context: split blocks on lone '--' separator lines. A
    # context line always carries a 'path-line-' prefix, so a bare '--' is
    # never content. rg merges contiguous hits into one group where the Python
    # path emits one group per hit — accepted divergence.
    results, block = [], []
    for line in lines + ["--"]:
        if line == "--":
            if block:
                results.append("\n".join(block))
                block = []
            continue
        m = _RG_MATCH_RE.match(line)
        if m:
            block.append(f"{_rg_rel(m.group(1))}:{m.group(2)}: {m.group(3)}")
            continue
        m = _RG_CONTEXT_RE.match(line)
        if m:
            block.append(f"{_rg_rel(m.group(1))}:{m.group(2)}- {m.group(3)}")
            continue
        return None  # unparseable line — fall back to the Python path
    return results


def _rg_rel(path: str) -> str:
    """Normalize an rg-emitted relative path to posix form."""
    path = path.replace("\\", "/")
    return path[2:] if path.startswith("./") else path


# ──────────────────────────────────────────────────────────────────────
# The Python search path.
# ──────────────────────────────────────────────────────────────────────

def content_matches(regex, multiline, text, rel, context_lines, results, limit):
    """Append content-mode result groups for one file."""
    lines = text.splitlines()
    if multiline:
        hit_lines = sorted({text.count("\n", 0, m.start())
                            for m in regex.finditer(text)})
    else:
        hit_lines = [i for i, line in enumerate(lines) if regex.search(line)]
    for i in hit_lines:
        lo = max(0, i - context_lines)
        hi = min(len(lines), i + context_lines + 1)
        results.append("\n".join(
            f"{rel}:{n + 1}{':' if n == i else '-'} {lines[n]}"
            for n in range(lo, hi)))
        if len(results) >= limit:
            return


def file_matches(regex, multiline, text) -> bool:
    """Whether the file matches, honoring line-scoped vs multiline mode."""
    if multiline:
        return regex.search(text) is not None
    return any(regex.search(line) for line in text.splitlines())
