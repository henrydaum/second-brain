"""The sandbox — mediating the boundary between the kernel and arbitrary code.

Sandboxed code cannot act. It can only *ask*, by making a typed Request that
the kernel classifies, executes, and answers. Each Request is therefore a
capability of the kernel, and the complete catalogue of them is the complete
list of what any plugin can ever do.

The package is split along the boundary it enforces:

**Guest** (``sandbox.guest``) — runs *inside*. Stdlib-only and self-contained:
the Request vocabulary, the wire format, the SDK, and the child entry point.
It is the shippable unit; a container image copies this directory alone.

**Host** — runs *outside*, and is never importable from the guest:

- ``policy``      — ``classify``: the one place a security level is decided
- ``handlers``    — the only code that actually touches the world
- ``interpreter`` — the drive loop: serial gate, parallel execution
- ``runner``      — in-process execution on a worker thread
- ``runner_subprocess`` — the same code behind a process boundary

Both runners share one gate, so policy, provenance and the ledger are
identical whichever way code is run.

See ``docs/SECURITY_CONTRACT_APPENDIX.md`` for the full Request catalogue.
"""

import sys as _sys

from . import guest as _guest

# Plugin source must read identically whichever runner executes it. The child
# runs with ``sandbox/`` as its working directory, so it imports ``guest``;
# in-process there is no such directory on the path. Aliasing the package here
# makes ``from guest.bases import BaseTool`` resolve in both, without putting
# ``sandbox/`` on sys.path (which would expose the host modules as top-level
# names and risk shadowing).
_sys.modules.setdefault("guest", _guest)

# Aliasing the package is not enough. Python imports ``guest.bases`` and
# ``sandbox.guest.bases`` as *separate* module objects, giving two distinct
# ``BasePlugin`` classes — so whether ``issubclass`` works depends on which
# name happened to be imported first. Pinning every submodule under both
# names keeps one object per module however it is reached.
import importlib as _importlib  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

for _found in sorted(_Path(_guest.__file__).parent.glob("*.py")):
    if _found.stem in ("__init__", "child"):
        continue          # child is only ever the subprocess entry point
    _sys.modules.setdefault(
        f"guest.{_found.stem}",
        _importlib.import_module(f".guest.{_found.stem}", __name__))

from .guest.channel import Terminated  # noqa: E402
from .guest.requests import Request, Result
from .guest.sdk import SDK
from .interpreter import (Execution, Interpreter, InterpreterChannel,  # noqa: E402
                          clamp_timeout)
from .boxes import BoxError, PersistentBox, open_box
from .facade import Run, Sandbox
from .policy import SAFE, UNSAFE, Chain, Decision, classify
from .runner import run_in_process
from .runner_subprocess import run_in_subprocess

__all__ = [
    "BoxError", "Chain", "Decision", "Execution", "Interpreter",
    "InterpreterChannel", "PersistentBox", "Request", "Result", "Run",
    "SDK", "SAFE", "UNSAFE", "Sandbox", "Terminated", "classify",
    "clamp_timeout", "open_box", "run_in_process", "run_in_subprocess",
]
