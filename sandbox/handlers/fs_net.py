"""Filesystem, network and process handlers — the stdlib-facing effects.

These are the handlers that need nothing from the kernel: a path, a URL, an
argv. Everything else in this package reaches into ``ctx``.
"""

from __future__ import annotations

import base64
import binascii
import itertools
import json
import os
import re
import shlex
import shutil
import stat as stat_module
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .. import provenance, walk
from ..guest import protocol
from ..guest.codes import ERROR_NOT_FOUND, ERROR_NOT_PERMITTED
from ..guest.requests import (ENV_READ, FS_DELETE, FS_LIST, FS_MOVE, FS_READ,
                              FS_READ_BYTES, FS_SEARCH, FS_TEMP, FS_WRITE,
                              FS_STAT, FS_WRITE_BYTES, NET_HTTP, PROC_LIST,
                              PROC_RUN,
                              PROC_START, PROC_STATUS, PROC_STOP,
                              SECRET_REVEAL, Result)
from ..protected import MAX_TEXT_READ_BYTES, is_protected, reason_for
from ..credentials import lookup_from, redact, resolve
from .args import float_arg, int_arg

MAX_READ_BYTES = MAX_TEXT_READ_BYTES
# Binary reads get their own cap: the things that need them are media files
# headed for a model or a chat transport, and a 10 MB video is ordinary where a
# 10 MB text file is a mistake.
#
# The number is *derived*, because it was guessed before and the guess was
# wrong in a way nothing noticed. It read 32 MB with a comment claiming base64
# made that the real ceiling on one frame; base64 inflates by 4/3, so anything
# over ~12 MB blew past ``protocol.MAX_MESSAGE_BYTES`` and surfaced as an
# unsendable-result *fault* — a crash-shaped answer to an ordinary request.
# Deriving it from the wire means the two cannot drift apart again.
#
# It is applied uniformly, including to in-process boxes that have no wire to
# overflow. A limit that holds in one isolation mode and not the other is the
# worst kind: code passes every local test and fails once it is subprocessed,
# which is precisely when nobody is watching.
MAX_READ_BINARY = (protocol.MAX_MESSAGE_BYTES - 1024 * 1024) * 3 // 4
MAX_SEARCH_HITS = 500
HTTP_TIMEOUT = 30.0
# Ten minutes, because the command that most needs the headroom is the one a
# user just approved a dialog for — ``pip install`` of something that builds
# from source. A box waiting here is *blocked*, not running, so the watchdog
# does not charge it (see ``Execution.running_for``); the caller's own
# ``timeout`` is what normally decides, and this only caps it.
PROC_TIMEOUT = 600.0

# The only schemes ``net.http`` will open. ``urllib`` speaks several others,
# and two of them make this Request a way around controls it sits beside:
# ``file://`` reads any path — including the ones ``protected.py`` exists to
# keep from crossing — and ``data:`` is not egress at all. Neither is what
# anybody means by "make an HTTP request", and both were reachable because the
# scheme was never looked at.
#
# This is sharper than it looks. ``net.http`` is UNSAFE and normally prompts,
# but a command declaring it in ``requests`` puts it in ``chain.approved``, and
# the policy function then returns SAFE — so ``file:///…/config.json`` returned
# every API key in plaintext with no dialog at all.
ALLOWED_SCHEMES = frozenset({"http", "https"})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Return redirects instead of silently changing network destination."""

    def redirect_request(self, request, file_pointer, code, message, headers,
                         new_url):
        return None


_HTTP_OPENER = urllib.request.build_opener(_NoRedirect)


def _fs_read(ctx, args: dict) -> Result:
    """Read a file as text."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.read requires a path")
    path = Path(raw)
    if (why := reason_for(path)):
        return Result.refusal(f"{raw} is not readable: {why}",
                              code=ERROR_NOT_PERMITTED)
    try:
        if not path.is_file():
            return Result.failure(f"not a file: {raw}",
                                  code=ERROR_NOT_FOUND)
        if path.stat().st_size > MAX_READ_BYTES:
            return Result.failure(f"file exceeds {MAX_READ_BYTES} bytes")
        return Result(data=path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return Result.failure(f"read failed: {exc}", retryable=True)


def _fs_read_bytes(ctx, args: dict) -> Result:
    """Read a file as raw bytes, base64-encoded for the wire.

    The guest decodes back to ``bytes``, so a plugin never sees the encoding.
    It exists because the things that need bytes — an image on its way to a
    vision model, audio a provider ingests natively — are exactly the things
    ``fs.read``'s ``errors="replace"`` would silently corrupt.

    ``offset``/``length`` read one window instead of the whole file. They are
    what makes a file larger than one message readable at all: the cap below is
    on a single *answer*, not on the file, and a caller that wants a 50 MB video
    asks for it a window at a time. Which bytes may leave is decided before
    either is looked at, so windowing is not a way around anything.
    """
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.read_bytes requires a path")
    path = Path(raw)
    # Same answer as fs.read: the encoding a caller asks for cannot be a way
    # around which bytes may leave.
    if (why := reason_for(path)):
        return Result.refusal(f"{raw} is not readable: {why}",
                              code=ERROR_NOT_PERMITTED)
    try:
        offset = max(0, int(args.get("offset") or 0))
        length = max(0, int(args.get("length") or 0))
    except (TypeError, ValueError):
        return Result.failure("fs.read_bytes offset and length must be whole "
                              "numbers of bytes")
    try:
        if not path.is_file():
            return Result.failure(f"not a file: {raw}",
                                  code=ERROR_NOT_FOUND)
        too_big = Result.failure(
            f"a single fs.read_bytes answer is capped at {MAX_READ_BINARY} "
            f"bytes; read it in windows with offset= and length=")
        if (length or max(0, path.stat().st_size - offset)) > MAX_READ_BINARY:
            return too_big
        with open(path, "rb") as handle:
            if offset:
                handle.seek(offset)
            # An explicit window is read exactly: it has already been checked,
            # and one byte more would quietly hand back more than was asked
            # for. Without one, read a byte past the cap instead — the size
            # came from a stat, and a file that grew since then should be
            # refused rather than truncated into a plausible-looking answer.
            chunk = (handle.read(length) if length
                     else handle.read(MAX_READ_BINARY + 1))
        if len(chunk) > MAX_READ_BINARY:
            return too_big
        return Result(data=base64.b64encode(chunk).decode("ascii"))
    except OSError as exc:
        return Result.failure(f"read failed: {exc}", retryable=True)


def _fs_stat(ctx, args: dict) -> Result:
    """Return metadata for exactly one path."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.stat requires a path")
    path = Path(raw)
    if (why := reason_for(path)):
        return Result.refusal(f"{raw} is not readable: {why}",
                              code=ERROR_NOT_PERMITTED)
    try:
        info = path.stat()
        return Result(data={
            "path": str(path),
            "name": path.name,
            "is_file": stat_module.S_ISREG(info.st_mode),
            "is_dir": stat_module.S_ISDIR(info.st_mode),
            "is_symlink": path.is_symlink(),
            "mtime": info.st_mtime_ns,
            "size": info.st_size,
        })
    except FileNotFoundError:
        if args.get("missing_ok"):
            return Result(data=None)
        return Result.failure(f"no such file or directory: {raw}",
                              code=ERROR_NOT_FOUND)
    except OSError as exc:
        return Result.failure(f"stat failed: {exc}", retryable=True)


def _guard_write(*paths) -> Result | None:
    """Refuse a write aimed at a file the kernel owns absolutely.

    Reads have always consulted this list; writes relied on the approval
    dialog instead, on the reasoning that a write outside scratch is UNSAFE
    and therefore asked about. That reasoning has a hole in it: a grant is
    *type-level*, so one "yes" to a command declaring ``fs.write`` covers every
    write it makes, and clobbering ``config.json`` or the live database is not
    something a person meant to authorise by approving a command.

    The kernel edits these files through its own code, never through a
    Request, so nothing legitimate is lost.
    """
    for raw in paths:
        if not raw:
            continue
        if (why := reason_for(raw)):
            return Result.refusal(f"{raw} is not writable: {why}",
                                  code=ERROR_NOT_PERMITTED)
    return None


def _fs_write_bytes(ctx, args: dict) -> Result:
    """Write raw bytes, supplied base64-encoded."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.write_bytes requires a path")
    if (refused := _guard_write(raw)) is not None:
        return refused
    encoded = args.get("data")
    if not isinstance(encoded, str):
        return Result.failure("fs.write_bytes requires base64 string data")
    try:
        # validate=True so silently-dropped junk becomes an error rather than
        # a truncated file.
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        return Result.failure(f"fs.write_bytes data is not valid base64: {exc}")
    mode = "ab" if args.get("mode") == "append" else "wb"
    path = Path(raw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode) as handle:
            handle.write(data)
        return Result(data={"path": str(path), "bytes": len(data)})
    except OSError as exc:
        return Result.failure(f"write failed: {exc}", retryable=True)


def _fs_write(ctx, args: dict) -> Result:
    """Create, overwrite, or append to a file."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.write requires a path")
    if (refused := _guard_write(raw)) is not None:
        return refused
    data = args.get("data")
    if not isinstance(data, str):
        return Result.failure("fs.write requires string data")
    mode = "a" if args.get("mode") == "append" else "w"
    path = Path(raw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding="utf-8") as handle:
            handle.write(data)
        return Result(data={"path": str(path), "bytes": len(data)})
    except OSError as exc:
        return Result.failure(f"write failed: {exc}", retryable=True)


# Arguments that switch fs.list out of its original shape. Presence, not
# truthiness: a caller passing ``limit=0`` still means "answer in the new
# shape", and a caller passing nothing at all must get byte-identical
# behaviour to before these existed.
_LIST_EXTRAS = ("recursive", "files_only", "sort", "limit")


def _fs_list(ctx, args: dict) -> Result:
    """List a directory, optionally filtered by glob pattern.

    Two shapes, chosen by whether the caller passed any of ``_LIST_EXTRAS``.
    Without them this is the original: a flat ``Path.glob`` returning a bare
    list. With them it walks, prunes junk directories, sorts, and caps —
    because ``**/*.py`` across a project through ``Path.glob`` descends into
    ``.git`` and ``node_modules`` and answers with tens of thousands of paths
    nobody asked for.
    """
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.list requires a path")
    pattern = args.get("pattern") or "*"
    path = Path(raw)
    extended = any(key in args for key in _LIST_EXTRAS)
    # Validated here rather than in ``_walked_entries``, which answers with a
    # tuple and has no way to report a bad argument.
    limit, bad = int_arg(args, "limit", 0, lo=0)
    if bad is not None:
        return bad
    args = {**args, "limit": limit}
    try:
        # A file answers for itself. Asking "what do you know about this one
        # path" is the common case behind a stat, and routing it through
        # "list the parent, then filter" made callers build a glob out of a
        # filename — which breaks the moment the name contains [ or *.
        if path.is_file():
            entries, scan_truncated, truncated = [path], False, False
        elif not path.is_dir():
            return Result.failure(f"no such directory or file: {raw}",
                                  code=ERROR_NOT_FOUND)
        elif not extended:
            entries = sorted(path.glob(pattern))
            return Result(data=_list_payload(entries, bool(args.get("details"))))
        else:
            entries, scan_truncated, truncated = _walked_entries(path, pattern, args)
    except OSError as exc:
        return Result.failure(f"list failed: {exc}", retryable=True)

    if not extended:
        return Result(data=_list_payload(entries, bool(args.get("details"))))
    return Result(data={
        "root": str(path),
        "entries": _list_payload(entries, bool(args.get("details"))),
        "truncated": truncated,
        "scan_truncated": scan_truncated,
    })


def _list_payload(entries, details: bool):
    """Entries as metadata dicts or as plain path strings."""
    if details:
        return [_entry_details(entry) for entry in entries]
    return [str(entry) for entry in entries]


def _walked_entries(root: Path, pattern: str, args: dict):
    """The pruning-walk half of ``fs.list``; ``(entries, scan_truncated, truncated)``."""
    files_only = bool(args.get("files_only"))
    if args.get("recursive"):
        entries, scan_truncated = (walk.iter_files(root) if files_only
                                   else walk.iter_entries(root))
    else:
        entries, scan_truncated = sorted(root.iterdir()), False
        if files_only:
            entries = [e for e in entries if e.is_file()]

    # A bare '*' is "everything here", which both branches above already
    # produced — and running it through compile_glob would silently re-narrow
    # a recursive walk to its top level.
    if pattern and pattern != "*":
        compiled = walk.compile_glob(pattern)
        entries = [e for e in entries if walk.match_rel(e, root, compiled)]

    if args.get("sort") == "mtime":
        entries = walk.mtime_sorted(entries)
    else:
        entries = sorted(entries)

    limit = args.get("limit") or 0
    truncated = bool(limit) and len(entries) > limit
    if truncated:
        entries = entries[:limit]
    return entries, scan_truncated, truncated


def _entry_details(entry: Path) -> dict:
    """One directory entry's metadata.

    ``mtime`` is ``st_mtime_ns`` — an int, so it survives JSON exactly, where
    a float seconds value would round and make "changed since I looked"
    unreliable at sub-millisecond resolution. Compare it with ``!=`` rather
    than ``<``: a file restored to an older version has also changed.

    A stat that fails leaves the fields ``None`` rather than dropping the
    entry, so a listing never silently omits a file the caller can see.
    """
    try:
        info = entry.stat()
        mtime, size = info.st_mtime_ns, info.st_size
    except OSError:
        mtime, size = None, None
    return {
        "path": str(entry),
        "name": entry.name,
        "is_dir": entry.is_dir(),
        "mtime": mtime,
        "size": size,
    }


# As with fs.list: presence of any of these switches fs.search into the
# grep-shaped answer. Without them the original substring scan is returned
# unchanged, so nothing built on it moves.
_SEARCH_EXTRAS = ("regex", "case_insensitive", "multiline", "mode",
                  "context_lines", "limit")
SEARCH_MODES = ("content", "files", "count")
DEFAULT_SEARCH_LIMIT = 100


def _fs_search(ctx, args: dict) -> Result:
    """Search file contents beneath a root.

    Derivable from list + read, and a separate Request anyway: doing it by
    hand costs one round trip per file. That argument gets stronger the more
    the search can do, which is why regex, output modes and junk-directory
    pruning live here rather than in whichever plugin wanted them first.
    """
    needle = args.get("pattern")
    root = Path(args.get("root") or ".")
    if not needle:
        return Result.failure("fs.search requires a pattern")
    if any(key in args for key in _SEARCH_EXTRAS):
        return _fs_search_extended(needle, root, args)

    glob = args.get("glob") or "**/*"
    hits = []
    try:
        for path in root.glob(glob):
            if not path.is_file() or path.stat().st_size > MAX_READ_BYTES:
                continue
            # Skipped rather than refused: a search over a whole tree should
            # not fail because a protected file happens to sit in it. It
            # matters as much as fs.read here — hits carry matching *lines*,
            # so pattern="secret_" would otherwise do the job on its own.
            if is_protected(path):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if needle in line:
                    hits.append({"path": str(path), "line": number,
                                 "text": line[:400]})
                    if len(hits) >= MAX_SEARCH_HITS:
                        return Result(data=hits)
    except OSError as exc:
        return Result.failure(f"search failed: {exc}", retryable=True)
    return Result(data=hits)


def _fs_search_extended(needle: str, root: Path, args: dict) -> Result:
    """The grep-shaped search: regex, output modes, pruning, ripgrep."""
    mode = args.get("mode") or "content"
    if mode not in SEARCH_MODES:
        return Result.failure(
            f"unknown fs.search mode {mode!r}; expected one of {list(SEARCH_MODES)}")
    multiline = bool(args.get("multiline"))
    case_insensitive = bool(args.get("case_insensitive"))
    context_lines, bad = int_arg(args, "context_lines", 0,
                                 lo=0, hi=walk.MAX_CONTEXT)
    if bad is not None:
        return bad
    limit, bad = int_arg(args, "limit", DEFAULT_SEARCH_LIMIT,
                         lo=1, hi=MAX_SEARCH_HITS)
    if bad is not None:
        return bad
    raw_glob = (args.get("glob") or "").strip()
    if raw_glob == "**/*":
        raw_glob = ""  # the default means "no filter", not a literal pattern

    flags = re.IGNORECASE if case_insensitive else 0
    if multiline:
        flags |= re.DOTALL
    # A literal search is a regex over the escaped pattern, so one engine
    # serves both and 'C++' cannot arrive as an invalid quantifier.
    source = needle if args.get("regex") else re.escape(needle)
    try:
        regex = re.compile(source, flags)
    except re.error as exc:
        return Result.failure(f"invalid regex: {exc}")

    if not root.exists():
        return Result.failure(f"no such directory or file: {root}",
                                  code=ERROR_NOT_FOUND)

    # ripgrep only ever runs against a literal regex over a directory: an
    # escaped literal round-trips through Rust's engine fine, but a
    # user-written Python pattern may not, and that fallback is already
    # handled by run_ripgrep returning None.
    rg = walk.rg_path() if root.is_dir() else None
    if rg:
        found = walk.run_ripgrep(rg, source, root, raw_glob, mode,
                                 case_insensitive, multiline, context_lines)
        if found is not None:
            # rg knows nothing about protected.py, and content hits carry
            # matching *lines* — so an unfiltered fast path would hand back
            # exactly the config lines the slow path exists to withhold. The
            # filter is applied before the limit, so a dropped hit cannot
            # push a real one off the end.
            found = _drop_protected(found, root, mode)
            truncated = len(found) > limit
            return Result(data=_search_payload(
                root, mode, found[:limit], truncated, False, 0, 0, "ripgrep"))

    return _search_python(regex, root, raw_glob, mode, multiline,
                          context_lines, limit)


def _search_python(regex, root: Path, raw_glob, mode, multiline,
                   context_lines, limit) -> Result:
    """The always-available path: walk, prune, read, match."""
    scan_truncated = False
    if root.is_file():
        files, base = [root], root.parent
    else:
        try:
            files, scan_truncated = walk.iter_files(root)
        except OSError as exc:
            return Result.failure(f"search failed: {exc}", retryable=True)
        base = root
        if raw_glob:
            compiled = walk.compile_glob(raw_glob)
            files = [f for f in files if walk.match_rel(f, base, compiled)]
        files = walk.mtime_sorted(files)

    results, skipped_binary, skipped_large, truncated = [], 0, 0, False
    for path in files:
        try:
            if path.stat().st_size > walk.MAX_FILE_BYTES:
                skipped_large += 1
                continue
        except OSError:
            continue
        # Same reasoning as the substring path: skipped, not refused, because
        # one protected file in a tree must not fail the whole search — and it
        # matters as much here, since hits carry matching lines.
        if is_protected(path):
            continue
        if walk.is_binary(path):
            skipped_binary += 1
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = walk.relative(path, base)

        if mode == "files":
            if walk.file_matches(regex, multiline, text):
                results.append(rel)
        elif mode == "count":
            if multiline:
                n = sum(1 for _ in regex.finditer(text))
            else:
                n = sum(len(regex.findall(line)) for line in text.splitlines())
            if n:
                results.append([rel, n])
        else:
            walk.content_matches(regex, multiline, text, rel, context_lines,
                                 results, limit)

        if len(results) >= limit:
            truncated = len(results) > limit or path is not files[-1]
            results = results[:limit]
            break

    return Result(data=_search_payload(root, mode, results, truncated,
                                       scan_truncated, skipped_binary,
                                       skipped_large, "python"))


def _drop_protected(results, root: Path, mode: str) -> list:
    """Remove ripgrep results belonging to protected or ignored files.

    Each shape names its file differently: ``files`` is the path itself,
    ``count`` is a ``[path, n]`` pair, and a ``content`` group is one or more
    ``rel:lineno: text`` lines that all came from the same file — so the first
    line's prefix identifies the group. A content line whose prefix will not
    parse is dropped rather than kept: an unrecognised shape is not evidence
    that the file behind it is safe to return.
    """
    kept = []
    for item in results:
        if mode == "files":
            rel = item
        elif mode == "count":
            rel = item[0]
        else:
            head = item.split("\n", 1)[0]
            match = _CONTENT_PREFIX_RE.match(head)
            if match is None:
                continue
            rel = match.group(1)
        parts = Path(str(rel).replace("\\", "/")).parts
        if any(part in walk.IGNORED_DIRS for part in parts):
            continue
        if not is_protected(root / rel):
            kept.append(item)
    return kept


# The 'rel:lineno: ' / 'rel:lineno- ' prefix parse_rg writes. Non-greedy, so a
# relative path containing a colon still yields at the line-number boundary.
_CONTENT_PREFIX_RE = re.compile(r"^(.+?):\d+[:-] ")


def _search_payload(root, mode, results, truncated, scan_truncated,
                    skipped_binary, skipped_large, backend) -> dict:
    """The extended search answer, identical across both backends."""
    return {"root": str(root), "mode": mode, "results": results,
            "truncated": truncated, "scan_truncated": scan_truncated,
            "skipped_binary": skipped_binary, "skipped_large": skipped_large,
            "backend": backend}


def _fs_delete(ctx, args: dict) -> Result:
    """Remove a file or a tree."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.delete requires a path")
    if (refused := _guard_write(raw)) is not None:
        return refused
    path = Path(raw)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            return Result.failure(f"no such path: {raw}",
                                  code=ERROR_NOT_FOUND)
        return Result(data={"deleted": str(path)})
    except OSError as exc:
        return Result.failure(f"delete failed: {exc}")


def _fs_move(ctx, args: dict) -> Result:
    """Copy or move one path to another."""
    src, dst = args.get("src"), args.get("dst")
    if not src or not dst:
        return Result.failure("fs.move requires src and dst")
    # Both ends: moving the database away is as destructive as writing over
    # it, and moving something *onto* it is the same act as a write.
    if (refused := _guard_write(src, dst)) is not None:
        return refused
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if args.get("copy"):
            if Path(src).is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        return Result(data={"src": str(src), "dst": str(dst)})
    except (OSError, shutil.Error) as exc:
        return Result.failure(f"move failed: {exc}")


def _fs_temp(ctx, args: dict) -> Result:
    """Allocate scratch space.

    Always safe, and separate from ``fs.write`` for that reason: "I need
    somewhere to put this" should never need a policy decision.
    """
    try:
        import trees

        scratch = Path(trees.tree("workspace").path) / "temp"
        scratch.mkdir(parents=True, exist_ok=True)
        prefix = provenance.scratch_prefix()
        if args.get("directory"):
            return Result(data=tempfile.mkdtemp(prefix=prefix, dir=scratch))
        handle, path = tempfile.mkstemp(prefix=prefix,
                                        suffix=args.get("suffix") or "",
                                        dir=scratch)
        os.close(handle)
        return Result(data=path)
    except OSError as exc:
        return Result.failure(f"temp failed: {exc}", retryable=True)


def _net_http(ctx, args: dict) -> Result:
    """Perform an outbound HTTP request.

    The one place secret handles are swapped back for real credentials: the
    sandbox never held the key, and substitution happens after the policy
    function has already decided.
    """
    url = args.get("url")
    if not url:
        return Result.failure("net.http requires a url")

    lookup = lookup_from(ctx)
    url = resolve(url, lookup)
    params = resolve(args.get("params"), lookup)
    if params is not None:
        try:
            encoded = urllib.parse.urlencode(params, doseq=True)
            parts = urllib.parse.urlsplit(url)
            query = "&".join(part for part in (parts.query, encoded) if part)
            url = urllib.parse.urlunsplit(
                (parts.scheme, parts.netloc, parts.path, query, parts.fragment))
        except (TypeError, ValueError) as exc:
            return Result.failure(f"net.http params are invalid: {exc}")
    # After substitution, so a handle cannot smuggle a scheme in, and before
    # anything is opened.
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        named = scheme or "no scheme"
        return Result.refusal(
            f"net.http speaks http and https, not {named}; "
            f"read files with sdk.fs.read", code=ERROR_NOT_PERMITTED)
    headers = resolve(dict(args.get("headers") or {}), lookup)
    body = resolve(args.get("body"), lookup)
    if "json" in args:
        if body is not None:
            return Result.failure("net.http body and json are mutually exclusive")
        try:
            body = json.dumps(resolve(args.get("json"), lookup))
        except (TypeError, ValueError) as exc:
            return Result.failure(f"net.http json is not serializable: {exc}")
        if not any(str(name).lower() == "content-type" for name in headers):
            headers["Content-Type"] = "application/json"
    method = (args.get("method") or "GET").upper()

    request = urllib.request.Request(
        url, method=method, headers=headers,
        data=body.encode("utf-8") if isinstance(body, str) else body,
    )
    try:
        with _HTTP_OPENER.open(request, timeout=HTTP_TIMEOUT) as response:
            return Result(data=_answer(response))
    except urllib.error.HTTPError as exc:
        # An error *status* is an answer, not a failed request. The body is
        # where an API explains itself — which rate limit, which parameter was
        # wrong, how long to wait — and collapsing it to ``http 429`` threw
        # away the only part the caller could act on. That loss was survivable
        # only because callers reached around it: the store's web-search
        # service carried a ``_read_http_error_body`` helper and its *tool*
        # called that private method, which is what a discarded answer costs.
        # The connection was made and the server replied, so this is a Result,
        # and the guest branches on ``status`` exactly as it would on a 200.
        return Result(data=_answer(exc))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # No reply at all — DNS, refused, timed out. Nothing to hand back but
        # the reason, so this stays a failure.
        return Result.failure(f"request failed: {exc}", retryable=True)


def _answer(response) -> dict:
    """One HTTP reply as plain data.

    ``HTTPError`` is itself a readable response object, which is why the
    success and error paths share this: the same three keys either way, so
    nothing downstream has to know which branch produced it.

    The body is UTF-8-decoded with replacement, so this Request answers about
    text and text only. Binary egress is deliberately absent — the things that
    want it (model weights, media) are foreign libraries doing their own I/O
    inside their own box, already outside the kernel's reach and documented as
    such.
    """
    try:
        payload = response.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        payload = ""
    try:
        headers = {str(k).lower(): str(v)
                   for k, v in (response.headers or {}).items()}
    except Exception:
        headers = {}
    status = getattr(response, "status", None) or getattr(response, "code", 0)
    return {"status": int(status or 0), "body": payload, "headers": headers}


# ── running commands ──────────────────────────────────────────────────
#
# Building the invocation is host work, not guest work, and the reason is
# Windows. ``cmd.exe`` does not understand the backslash-escaped quotes that
# ``subprocess``'s list-to-command-line conversion produces, so a guest that
# wrapped its own command as ``["cmd", "/c", command]`` would have every
# embedded quote silently mangled — ``git commit -m "two words"`` arrives as
# something else entirely. Passing the string with ``shell=True`` hands it to
# the shell verbatim, which is the only form that round-trips. So the guest
# names *which shell* and the kernel builds the call.

_SHELLS = (None, "default", "powershell", "cmd")


def _command_line(argv) -> str:
    """An argv rendered as something a shell will parse back the same way."""
    if isinstance(argv, str):
        return argv
    parts = [str(part) for part in (argv or [])]
    return (subprocess.list2cmdline(parts) if os.name == "nt"
            else shlex.join(parts))


def _invocation(args: dict):
    """Resolve a shell Request's arguments into what ``subprocess`` wants.

    Answers ``(cmd, use_shell, rendered)``, or a :class:`Result` failure.
    """
    argv = args.get("argv")
    if not argv:
        return Result.failure("a command is required")
    shell = args.get("shell")
    if shell not in _SHELLS:
        return Result.failure(
            f"unknown shell {shell!r}; expected one of "
            f"{[s for s in _SHELLS if s]} or none")

    rendered = _command_line(argv)
    if shell is None:
        # No shell: the argv is executed as given. A string is split the way a
        # shell would, but without one — no globbing, no pipes, no
        # substitution. This is the right default for a caller that built the
        # list itself.
        exact = shlex.split(argv) if isinstance(argv, str) else [
            str(part) for part in argv]
        return exact, False, rendered
    if shell == "powershell":
        # PowerShell parses its own arguments with rules ``list2cmdline``
        # matches, so the list form is safe here and gives us -NoProfile.
        #
        # The binary has two names and they are not interchangeable.
        # ``powershell`` is Windows PowerShell 5.1 and exists only on Windows;
        # PowerShell Core installs as ``pwsh`` everywhere, which is the only
        # thing a Mac or Linux box could have. Naming the wrong one turned a
        # supported shell into "could not run: No such file or directory".
        binary = "powershell" if os.name == "nt" else "pwsh"
        return ([binary, "-NoProfile", "-Command", rendered], False, rendered)
    if shell == "cmd" and os.name != "nt":
        return Result.failure("shell 'cmd' is only available on Windows")
    return rendered, True, rendered


def _proc_run(ctx, args: dict) -> Result:
    """Run a command to completion."""
    built = _invocation(args)
    if isinstance(built, Result):
        return built
    cmd, use_shell, rendered = built
    timeout, bad = float_arg(args, "timeout", PROC_TIMEOUT,
                             lo=0.0, hi=PROC_TIMEOUT)
    if bad is not None:
        return bad
    try:
        done = subprocess.run(cmd, shell=use_shell, capture_output=True,
                              text=True, errors="replace", timeout=timeout,
                              cwd=args.get("cwd") or None)
        return Result(data={"code": done.returncode,
                            "stdout": (done.stdout or "")[-100_000:],
                            "stderr": (done.stderr or "")[-100_000:],
                            "command": rendered})
    except subprocess.TimeoutExpired:
        return Result.failure(f"timed out after {timeout:.0f}s", retryable=True)
    except (OSError, ValueError) as exc:
        return Result.failure(f"could not run: {exc}")


# ── processes that outlive the Request that started them ──────────────
#
# The registry is in memory, and deliberately: a ``Popen`` handle is not
# serializable, so nothing here survives a restart. What survives is the log
# file. ``main.pyw`` exits through ``os._exit``, so a process started here and
# not stopped is orphaned rather than killed when Second Brain goes down —
# which is why ``proc.stop`` is classified safe, and why the agent prompt says
# to use it.

_PROCESSES: dict = {}
_NEXT_ID = itertools.count(1)
_PROCESS_LIMIT = 64


def _reap() -> None:
    """Drop the oldest finished entries once the registry gets long.

    Nothing else forgets them: an exited process stays listed so its output is
    still readable, which is the point, but not forever.
    """
    while len(_PROCESSES) > _PROCESS_LIMIT:
        for key, entry in sorted(_PROCESSES.items()):
            if entry["popen"].poll() is not None:
                _PROCESSES.pop(key, None)
                break
        else:
            return  # everything still running; nothing to reap


def _tail(path: str, limit: int) -> str:
    """The last ``limit`` characters a process wrote."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return "(log unavailable)"
    if not text:
        return "(no output yet)"
    return text[-limit:] if limit and limit > 0 else text


def _described(key: int, entry: dict, tail: int = 0) -> dict:
    """One registry entry as the guest sees it."""
    code = entry["popen"].poll()
    described = {"id": key, "pid": entry["popen"].pid,
                 "command": entry["command"], "label": entry["label"],
                 "cwd": entry["cwd"], "log": entry["log"],
                 "started_at": entry["started_at"],
                 "running": code is None, "code": code}
    if tail:
        described["output"] = _tail(entry["log"], tail)
    return described


def _proc_start(ctx, args: dict) -> Result:
    """Start a command and leave it running."""
    built = _invocation(args)
    if isinstance(built, Result):
        return built
    cmd, use_shell, rendered = built
    cwd = args.get("cwd") or None

    handle = None
    try:
        descriptor, log = tempfile.mkstemp(prefix="sb-proc-", suffix=".log")
        handle = open(descriptor, "w", encoding="utf-8", errors="replace")
        handle.write(f"$ {rendered}\n# cwd: {cwd or os.getcwd()}\n\n")
        handle.flush()
        popen = subprocess.Popen(
            cmd, shell=use_shell, cwd=cwd,
            stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # POSIX: its own process group, so stopping it reaches the
            # children a shell command line usually has. Windows gets the
            # same reach from ``taskkill /T``.
            start_new_session=os.name != "nt")
    except (OSError, ValueError) as exc:
        return Result.failure(f"could not start: {exc}")
    finally:
        # The child holds its own duplicate of the descriptor.
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    key = next(_NEXT_ID)
    _PROCESSES[key] = {"popen": popen, "command": rendered, "log": log,
                       "cwd": str(cwd) if cwd else os.getcwd(),
                       "label": args.get("label") or "",
                       "started_at": time.time()}
    _reap()
    return Result(data=_described(key, _PROCESSES[key]))


def _proc_status(ctx, args: dict) -> Result:
    """Report on a started process, with the tail of what it has written."""
    key = args.get("id")
    entry = _PROCESSES.get(key)
    if entry is None:
        return Result.failure(f"no process {key!r}; ask proc.list")
    tail = args.get("tail")
    return Result(data=_described(key, entry,
                                  tail=int(tail) if tail else 4000))


def _signal_group(popen, sig) -> None:
    """Send a signal to a POSIX child's whole process group.

    The group is why ``start_new_session`` is set: a shell command line
    usually *is* several processes, and signalling only the shell leaves the
    server it launched running with nothing tracking it.
    """
    try:
        os.killpg(os.getpgid(popen.pid), sig)
    except OSError:
        # Already gone, or never got a group. Fall back to the child alone,
        # which is still the right thing to try.
        try:
            popen.send_signal(sig)
        except OSError:
            pass


def _end(popen) -> int | None:
    """End a running child as firmly as the platform requires.

    Windows and POSIX are asymmetric here and the asymmetry matters.
    ``taskkill /T /F`` is already a hard kill of the whole tree, so it either
    works or the process was unkillable anyway. POSIX ``SIGTERM`` is a
    *request*, which is the right thing to send a dev server first — it gets
    to close its socket — but a process that traps or ignores it survives.
    Stopping has to be escalated, or ``proc.stop`` reports success on a
    process that is still running and no longer tracked, which is the worst
    of both.
    """
    if popen.poll() is not None:
        return popen.returncode
    if os.name == "nt":
        try:
            # With shell=True the tracked pid is the shell's, so only a tree
            # kill reaches what was actually asked for.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(popen.pid)],
                           capture_output=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            try:
                popen.kill()
            except OSError:
                pass
    else:
        import signal as signals

        _signal_group(popen, signals.SIGTERM)
        try:
            return popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _signal_group(popen, signals.SIGKILL)
    try:
        return popen.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return None


def _proc_stop(ctx, args: dict) -> Result:
    """End a started process and forget it."""
    key = args.get("id")
    entry = _PROCESSES.get(key)
    if entry is None:
        return Result.failure(f"no process {key!r}; ask proc.list")
    code = _end(entry["popen"])
    _PROCESSES.pop(key, None)
    # ``code is None`` means it outlived even a kill. Say so rather than
    # reporting a clean stop: it is now running untracked, which somebody
    # needs to know.
    return Result(data={"id": key, "code": code, "command": entry["command"],
                        "log": entry["log"], "stopped": code is not None,
                        "pid": entry["popen"].pid})


def _proc_list(ctx, args: dict) -> Result:
    """Every process this system started and still remembers."""
    return Result(data=[_described(key, entry)
                        for key, entry in sorted(_PROCESSES.items())])


def _env_read(ctx, args: dict) -> Result:
    """Read an environment variable, redacting credentials."""
    name = args.get("name")
    if not name:
        return Result.failure("env.read requires a name")
    value = os.environ.get(name)
    if value is None:
        return Result(data=None)
    # Nothing declares an environment variable, so the name is all there is.
    return Result(data=redact(name, value, guess=True))


def _secret_reveal(ctx, args: dict) -> Result:
    """Hand over a credential in plaintext.

    The handle mechanism works when the *kernel* performs the effect, because
    it can substitute on the way out. A plugin driving a foreign library —
    an OAuth client, a provider SDK — performs its own I/O, so there is no
    such moment and it genuinely needs the value.

    That is the same foreign-library limit the security contract already
    names, not a separate one. The answer is the answer to everything else
    here: not forbidden, gated. This Request is always unsafe, so the user
    sees which secret, which plugin, and what chain asked for it, and the
    ledger keeps the record.
    """
    name = args.get("name")
    if not name:
        return Result.failure("secret.reveal requires a name")
    value = lookup_from(ctx)(name)
    if value is None:
        return Result.failure(f"no secret named {name!r}")
    return Result(data=value)


HANDLERS = {
    FS_READ: _fs_read,
    SECRET_REVEAL: _secret_reveal,
    FS_WRITE: _fs_write,
    FS_READ_BYTES: _fs_read_bytes,
    FS_STAT: _fs_stat,
    FS_WRITE_BYTES: _fs_write_bytes,
    FS_LIST: _fs_list,
    FS_SEARCH: _fs_search,
    FS_DELETE: _fs_delete,
    FS_MOVE: _fs_move,
    FS_TEMP: _fs_temp,
    NET_HTTP: _net_http,
    PROC_RUN: _proc_run,
    PROC_START: _proc_start,
    PROC_STATUS: _proc_status,
    PROC_STOP: _proc_stop,
    PROC_LIST: _proc_list,
    ENV_READ: _env_read,
}
