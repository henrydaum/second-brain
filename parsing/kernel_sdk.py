"""The `sdk` a parser gets when the *kernel* is the one calling it.

A parser has two callers and one signature. Inside a box it receives the real
SDK and every effect is a Request; here it receives this, and effects happen
directly — because the kernel is already inside the boundary. Mediating the
kernel against itself would be theatre, and worse, it would mean a parse the
kernel needs could be denied.

So this is deliberately a *stand-in*, not a fake: the same handful of names a
parser actually uses, doing the obvious thing. It is small on purpose. If a
parser needs something that is not here, that is a signal it is doing
something a parser should not.

The point of the shared signature is that a parser stops caring. It reads
through ``sdk.fs.read``, reaches peers through ``sdk.services.call``, and logs
through ``sdk.log`` — and whether those are Requests or direct calls is
somebody else's problem. That is what makes the same file importable into a
box without a second contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

# The exception vocabulary is part of what a parser can *observe*, so it comes
# from the guest rather than being restated here — a stand-in that raises
# lookalikes would let `except sdk.Failed` work in a box and miss kernel-side.
from sandbox.guest.codes import ERROR_NOT_FOUND
from sandbox.guest.requests import (SERVICE_CALL, SERVICE_LOAD,
                                    SERVICE_UNLOAD, Denied, RequestFailed,
                                    Result)

logger = logging.getLogger("Parsing")


class _Fs:
    """Reading files, the way the kernel does it: directly."""

    def read(self, path, encoding: str = "utf-8") -> str:
        """Read a file as text, falling back to latin-1 like the parsers do."""
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as handle:
                return handle.read()

    def read_bytes(self, path, offset: int = 0, length: int = 0) -> bytes:
        """Read all bytes or one window, matching the guest SDK."""
        with open(path, "rb") as handle:
            if offset:
                handle.seek(max(0, int(offset)))
            return handle.read(max(0, int(length))) if length else handle.read()

    def iter_bytes(self, path, chunk_size: int = 4 * 1024 * 1024,
                   offset: int = 0, limit=None):
        """Yield a binary file in the same windows as the guest SDK."""
        chunk_size = int(chunk_size)
        offset = int(offset)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if offset < 0:
            raise ValueError("offset must not be negative")
        remaining = None if limit is None else int(limit)
        if remaining is not None and remaining < 0:
            raise ValueError("limit must not be negative")

        while remaining is None or remaining:
            length = (chunk_size if remaining is None
                      else min(chunk_size, remaining))
            chunk = self.read_bytes(path, offset=offset, length=length)
            if not chunk:
                break
            yield chunk
            offset += len(chunk)
            if remaining is not None:
                remaining -= len(chunk)
            if len(chunk) < length:
                break

    def stat(self, path) -> dict:
        """Return metadata for one path, matching the guest SDK shape."""
        target = Path(path)
        info = target.stat()
        return {
            "path": str(target),
            "name": target.name,
            "is_file": target.is_file(),
            "is_dir": target.is_dir(),
            "is_symlink": target.is_symlink(),
            "mtime": info.st_mtime_ns,
            "size": info.st_size,
        }

    def exists(self, path) -> bool:
        """Whether a path exists."""
        return Path(path).exists()

    def write(self, path, data, mode: str = "overwrite") -> dict:
        """Create, overwrite, or append to a file, creating parents."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a" if mode == "append" else "w",
                  encoding="utf-8") as handle:
            handle.write(data)
        return {"path": str(target), "bytes": len(data)}

    def write_bytes(self, path, data, mode: str = "overwrite") -> dict:
        """Write raw bytes, creating parents.

        A ``str`` is encoded rather than refused, matching the real SDK — the
        two must agree on everything a parser can observe, or a file works one
        way and breaks the other.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "ab" if mode == "append" else "wb") as handle:
            handle.write(data)
        return {"path": str(target), "bytes": len(data)}

    def list(self, path, pattern: str = "*") -> list:
        """Everything in a directory matching a glob."""
        target = Path(path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        return sorted(str(p) for p in target.glob(pattern))

    def search(self, pattern: str, root=".", glob: str = "**/*") -> list:
        """Lines matching a pattern under a root."""
        import re

        expression = re.compile(pattern)
        hits = []
        for candidate in sorted(Path(root).glob(glob)):
            if not candidate.is_file():
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if expression.search(line):
                    hits.append({"path": str(candidate), "line": number,
                                 "text": line})
        return hits

    def delete(self, path) -> bool:
        """Remove a file or a tree."""
        import shutil

        target = Path(path)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        return True

    def move(self, src, dst, copy: bool = False) -> str:
        """Copy or move one path to another."""
        import shutil

        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if copy:
            if Path(src).is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        return str(dst)

    def mkdir(self, path, exist_ok: bool = True) -> str:
        """Create a directory and any missing parents.

        Rarely needed here for the same reason it is rarely needed in a box:
        :meth:`write` already makes the folders above the file.
        """
        Path(path).mkdir(parents=True, exist_ok=bool(exist_ok))
        return str(path)

    def temp(self, directory: bool = False, suffix: str = "") -> str:
        """Scratch space, for parsers that must hand a path to a library.

        Named after whoever asked, exactly as the ``fs.temp`` handler is, and
        this is the copy that mattered: a parser the *kernel* calls reaches
        this stand-in rather than the Request, so every extracted archive in
        ``workspace/temp`` was named here. Both had to change or the folder
        stays unreadable in the one case that fills it.

        The chain is reachable even though nothing here is a Request, because
        ``Interpreter._execute`` marks the thread for the whole handler and
        ``parsing.parse`` runs synchronously inside it. Imported late to keep
        this module's import graph as small as its docstring claims.
        """
        import os
        import tempfile

        import trees
        from sandbox import provenance

        scratch = Path(trees.tree("workspace").path) / "temp"
        scratch.mkdir(parents=True, exist_ok=True)
        prefix = provenance.scratch_prefix()
        if directory:
            return tempfile.mkdtemp(prefix=prefix, dir=scratch)
        handle, path = tempfile.mkstemp(prefix=prefix, suffix=suffix,
                                        dir=scratch)
        os.close(handle)
        return path


class _Services:
    """Reaching a peer service, for parsers that delegate.

    This replaced the ``services`` dict that used to be threaded through every
    parser call — one uniform way to reach a peer instead of a parameter every
    parser had to accept and most ignored.
    """

    def __init__(self, services: dict | None = None):
        self._services = services if services is not None else {}

    def bind(self, services: dict | None) -> None:
        """Point at the live registry."""
        self._services = services if services is not None else {}

    def call(self, name: str, method: str, **kwargs):
        """Invoke a method on a loaded service.

        Every failure is a ``RequestFailed``, with the messages and codes
        ``handlers.kernel._service_call`` uses. A parser guards this call with
        ``except sdk.Failed`` — the delegating ones all do, since "not
        installed", "not loaded" and "it broke" are one answer to them — and
        raising ``LookupError`` here meant that guard caught nothing kernel-side
        while catching everything in a box.
        """
        service = self._services.get(name)
        if service is None:
            raise RequestFailed(
                Result.failure(f"service {name!r} is not loaded",
                               code=ERROR_NOT_FOUND), SERVICE_CALL)

        exports = getattr(service, "exports", None)
        if exports is not None and method not in exports:
            raise RequestFailed(Result.failure(
                f"{name}.{method} is not exported; {sorted(exports)} are"),
                SERVICE_CALL)

        fn = getattr(service, method or "", None)
        if not callable(fn):
            raise RequestFailed(
                Result.failure(f"{name} has no method {method!r}"),
                SERVICE_CALL)
        try:
            return fn(**kwargs)
        except Exception as exc:
            # Foreign code, so it is guarded — and the traceback is kept,
            # because the message only says whose bug it is.
            logger.exception("service_call failed")
            raise RequestFailed(
                Result.failure(f"{name}.{method} failed: {exc}"),
                SERVICE_CALL) from exc

    def list(self) -> dict:
        """Loaded services and whether each is ready, as ``sdk.services.list``.

        Parsers that delegate check this before calling, so it has to answer
        the same shape the real SDK does — a stand-in that is missing a method
        the guest has breaks the one promise this class exists to keep.
        """
        return {name: bool(getattr(service, "loaded", False))
                for name, service in self._services.items()}

    def get(self, name: str):
        """The service instance, or None. For parsers that check first."""
        return self._services.get(name)

    def load(self, name: str):
        """Parsers may inspect/call peers, never change their lifecycle.

        A refusal is a ``Denied``, for the same reason ``call`` raises
        ``RequestFailed``: this is policy, and policy is the one thing a guest
        already knows how to catch.
        """
        raise Denied(Result.refusal(
            f"parsers cannot load service {name!r}"), SERVICE_LOAD)

    def unload(self, name: str):
        """Parsers may inspect/call peers, never change their lifecycle."""
        raise Denied(Result.refusal(
            f"parsers cannot unload service {name!r}"), SERVICE_UNLOAD)


class KernelSDK:
    """What a parser sees when the kernel calls it."""

    #: Raised when the kernel refused. Subclasses ``Failed``, as in the guest.
    Denied = Denied
    #: Raised when something broke. The two names a parser writes in an
    #: ``except`` clause, so they are not optional: a missing one makes the
    #: clause itself an ``AttributeError``, which masks the failure it was
    #: written to handle.
    Failed = RequestFailed

    def __init__(self, services: dict | None = None):
        self.fs = _Fs()
        self.services = _Services(services)

    def log(self, message: str, level: str = "info") -> None:
        """Log through the sdk, the way sandboxed code has to."""
        getattr(logger, level, logger.info)(message)

    def ok(self, data, **extras):
        """Present so a parser can use the same idiom in both worlds."""
        return data

    def fail(self, error: str, retryable: bool = False):
        """Ditto — a parser normally returns ParseResult.failed instead.

        It *returns*, matching the guest. Raising was the same parity bug one
        method over: ``return sdk.fail(...)`` answered in a box and blew up
        kernel-side.
        """
        return Result.failure(error, retryable=retryable)


# One instance, because the kernel is one process and this holds no per-call
# state. ``bind_services`` points its service lookup at the live registry.
KERNEL_SDK = KernelSDK()
