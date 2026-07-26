"""Filesystem, network and process handlers — the stdlib-facing effects.

These are the handlers that need nothing from the kernel: a path, a URL, an
argv. Everything else in this package reaches into ``ctx``.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from ..guest.requests import (ENV_READ, FS_DELETE, FS_LIST, FS_MOVE, FS_READ,
                              FS_SEARCH, FS_TEMP, FS_WRITE, NET_HTTP, PROC_RUN,
                              Result)
from ..secrets import lookup_from, redact, resolve

MAX_READ_BYTES = 8 * 1024 * 1024
MAX_SEARCH_HITS = 500
HTTP_TIMEOUT = 30.0
PROC_TIMEOUT = 120.0


def _fs_read(ctx, args: dict) -> Result:
    """Read a file as text."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.read requires a path")
    path = Path(raw)
    try:
        if not path.is_file():
            return Result.failure(f"not a file: {raw}")
        if path.stat().st_size > MAX_READ_BYTES:
            return Result.failure(f"file exceeds {MAX_READ_BYTES} bytes")
        return Result(data=path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return Result.failure(f"read failed: {exc}", retryable=True)


def _fs_write(ctx, args: dict) -> Result:
    """Create, overwrite, or append to a file."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.write requires a path")
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


def _fs_list(ctx, args: dict) -> Result:
    """List a directory, optionally filtered by glob pattern."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.list requires a path")
    pattern = args.get("pattern") or "*"
    path = Path(raw)
    try:
        if not path.is_dir():
            return Result.failure(f"not a directory: {raw}")
        return Result(data=sorted(str(p) for p in path.glob(pattern)))
    except OSError as exc:
        return Result.failure(f"list failed: {exc}", retryable=True)


def _fs_search(ctx, args: dict) -> Result:
    """Search file contents beneath a root.

    Derivable from list + read, and a separate Request anyway: doing it by
    hand costs one round trip per file.
    """
    needle = args.get("pattern")
    root = Path(args.get("root") or ".")
    if not needle:
        return Result.failure("fs.search requires a pattern")
    glob = args.get("glob") or "**/*"
    hits = []
    try:
        for path in root.glob(glob):
            if not path.is_file() or path.stat().st_size > MAX_READ_BYTES:
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


def _fs_delete(ctx, args: dict) -> Result:
    """Remove a file or a tree."""
    raw = args.get("path")
    if not raw:
        return Result.failure("fs.delete requires a path")
    path = Path(raw)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        else:
            return Result.failure(f"no such path: {raw}")
        return Result(data={"deleted": str(path)})
    except OSError as exc:
        return Result.failure(f"delete failed: {exc}")


def _fs_move(ctx, args: dict) -> Result:
    """Copy or move one path to another."""
    src, dst = args.get("src"), args.get("dst")
    if not src or not dst:
        return Result.failure("fs.move requires src and dst")
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
        if args.get("directory"):
            return Result(data=tempfile.mkdtemp(prefix="sb-box-"))
        handle, path = tempfile.mkstemp(prefix="sb-box-",
                                        suffix=args.get("suffix") or "")
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
    headers = resolve(dict(args.get("headers") or {}), lookup)
    body = resolve(args.get("body"), lookup)
    method = (args.get("method") or "GET").upper()

    request = urllib.request.Request(
        url, method=method, headers=headers,
        data=body.encode("utf-8") if isinstance(body, str) else body,
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = response.read().decode("utf-8", errors="replace")
            return Result(data={"status": response.status, "body": payload})
    except urllib.error.HTTPError as exc:
        return Result.failure(f"http {exc.code}", retryable=exc.code >= 500)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Result.failure(f"request failed: {exc}", retryable=True)


def _proc_run(ctx, args: dict) -> Result:
    """Run a command to completion."""
    argv = args.get("argv")
    if isinstance(argv, str):
        argv = shlex.split(argv)
    if not argv:
        return Result.failure("proc.run requires argv")
    timeout = min(float(args.get("timeout") or PROC_TIMEOUT), PROC_TIMEOUT)
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, cwd=args.get("cwd") or None)
        return Result(data={"code": done.returncode,
                            "stdout": done.stdout[-100_000:],
                            "stderr": done.stderr[-100_000:]})
    except subprocess.TimeoutExpired:
        return Result.failure(f"timed out after {timeout:.0f}s", retryable=True)
    except (OSError, ValueError) as exc:
        return Result.failure(f"could not run: {exc}")


def _env_read(ctx, args: dict) -> Result:
    """Read an environment variable, redacting credentials."""
    name = args.get("name")
    if not name:
        return Result.failure("env.read requires a name")
    value = os.environ.get(name)
    if value is None:
        return Result(data=None)
    return Result(data=redact(name, value))


HANDLERS = {
    FS_READ: _fs_read,
    FS_WRITE: _fs_write,
    FS_LIST: _fs_list,
    FS_SEARCH: _fs_search,
    FS_DELETE: _fs_delete,
    FS_MOVE: _fs_move,
    FS_TEMP: _fs_temp,
    NET_HTTP: _net_http,
    PROC_RUN: _proc_run,
    ENV_READ: _env_read,
}
