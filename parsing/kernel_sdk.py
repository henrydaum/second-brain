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

    def read_bytes(self, path) -> bytes:
        """Read a file as bytes — what a foreign decoder should be handed."""
        with open(path, "rb") as handle:
            return handle.read()

    def write(self, path, data, mode: str = "overwrite") -> dict:
        """Create, overwrite, or append to a file, creating parents."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a" if mode == "append" else "w",
                  encoding="utf-8") as handle:
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

    def temp(self, directory: bool = False, suffix: str = "") -> str:
        """Scratch space, for parsers that must hand a path to a library."""
        import os
        import tempfile

        if directory:
            return tempfile.mkdtemp()
        handle, path = tempfile.mkstemp(suffix=suffix)
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
        """Invoke a method on a loaded service."""
        service = self._services.get(name)
        if service is None:
            raise LookupError(f"service {name!r} is not loaded")
        fn = getattr(service, method, None)
        if not callable(fn):
            raise AttributeError(f"{name} has no method {method!r}")
        return fn(**kwargs)

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


class KernelSDK:
    """What a parser sees when the kernel calls it."""

    def __init__(self, services: dict | None = None):
        self.fs = _Fs()
        self.services = _Services(services)

    def log(self, message: str, level: str = "info") -> None:
        """Log through the sdk, the way sandboxed code has to."""
        getattr(logger, level, logger.info)(message)

    def ok(self, data, **extras):
        """Present so a parser can use the same idiom in both worlds."""
        return data

    def fail(self, error: str):
        """Ditto — a parser normally returns ParseResult.failed instead."""
        raise RuntimeError(error)


# One instance, because the kernel is one process and this holds no per-call
# state. ``bind_services`` points its service lookup at the live registry.
KERNEL_SDK = KernelSDK()
