"""The guest boundary, pinned executably.

``sandbox/guest/`` must stay self-contained: stdlib-only, importing nothing
from the host and nothing from the wider repo. Two things depend on it —

1. **Security.** If guest code can import ``sandbox.handlers`` it can call a
   handler directly and skip the gate entirely.
2. **Containers.** The guest is the shippable unit. An image copies that
   directory alone, so anything it reaches outside itself is a file that would
   have to ship too.

This is the sandbox's counterpart to ``test_kernel_boundary.py``: widening the
boundary fails the suite until the allowlist here is changed deliberately.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

GUEST_DIR = Path(__file__).resolve().parent.parent / "sandbox" / "guest"

# Everything the guest is permitted to import. Stdlib only, plus its own
# siblings by relative import.
ALLOWED_ABSOLUTE = {
    "__future__", "base64", "importlib", "importlib.util", "json", "re",
    "sys", "traceback", "pathlib", "time", "typing", "dataclasses",
    "resource",
    # The loader re-hashes a file before executing it, so the bytes that ran
    # are the bytes the validator passed. Pure computation over a string.
    "hashlib",
    # Pure path *arithmetic* for sdk.path — string manipulation with no cwd,
    # no stat, no environment. Named directly rather than reached through
    # ``os.path``, which is one of these two under an alias that also imports
    # ``os`` — the module this boundary exists to keep out.
    "ntpath", "posixpath",
}

HOST_MODULES = {"policy", "handlers", "interpreter", "runner",
                "runner_subprocess"}


def _guest_files():
    """Every Python file that ships inside the boundary."""
    return sorted(GUEST_DIR.glob("*.py"))


def test_guest_directory_is_not_empty():
    """Guard against the glob silently matching nothing."""
    assert len(_guest_files()) >= 5


@pytest.mark.parametrize("path", _guest_files(), ids=lambda p: p.name)
def test_guest_imports_only_stdlib_and_siblings(path):
    """No host module, no repo module, no third-party dependency."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert alias.name in ALLOWED_ABSOLUTE or root in ALLOWED_ABSOLUTE, (
                    f"{path.name} imports {alias.name!r}, which is not stdlib")

        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative: must stay within guest/ (level 1) and must not
                # climb out to the host package.
                assert node.level == 1, (
                    f"{path.name} climbs out of guest/ with a level-"
                    f"{node.level} relative import")
                if node.module:
                    assert node.module.split(".")[0] not in HOST_MODULES, (
                        f"{path.name} imports host module {node.module!r}")
                continue

            module = node.module or ""
            root = module.split(".")[0]
            assert root != "sandbox", (
                f"{path.name} imports {module!r} — the guest must never "
                f"reach the host package")
            assert module in ALLOWED_ABSOLUTE or root in ALLOWED_ABSOLUTE, (
                f"{path.name} imports {module!r}, which is not stdlib")


def test_host_modules_are_unreachable_from_the_child_at_runtime():
    """The static check, confirmed against a real child process.

    The child runs with ``sandbox/`` as its working directory, so ``guest`` is
    a top-level package and the host is not on its import path at all.
    """
    probe = (
        "import guest, sys; "
        "assert 'sandbox' not in sys.modules; "
        "import importlib; "
        "ok = False\n"
        "try:\n"
        "    importlib.import_module('handlers')\n"
        "except ImportError:\n"
        "    ok = True\n"
        "print('unreachable' if ok else 'REACHABLE')"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(GUEST_DIR.parent), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "unreachable"


def test_no_host_module_shadows_a_stdlib_module():
    """``sandbox/`` is the child's cwd, so its top-level names are the child's.

    That cuts both ways. The guest boundary keeps guest code from *reaching*
    the host; this keeps the host from *displacing* the stdlib underneath it.
    ``sandbox/secrets.py`` did exactly that: nothing in the sandbox imports
    stdlib ``secrets``, so it looked harmless, while inside every box
    ``import secrets`` returned a host module with no ``token_hex``. litellm
    calls it, so every model call failed — and the traceback pointed at the
    library, not at the file that broke it.
    """
    stdlib = set(sys.stdlib_module_names)
    host = GUEST_DIR.parent
    candidates = [p for p in host.glob("*.py")]
    candidates += [p for p in host.iterdir()
                   if p.is_dir() and (p / "__init__.py").exists()]
    offenders = sorted(
        p.name for p in candidates
        if p.stem in stdlib and p.stem != "__init__"
    )
    assert not offenders, (
        f"{offenders} shadow stdlib modules for every sandboxed process. "
        "Rename them (sandbox/secrets.py became sandbox/credentials.py)."
    )


def test_guest_carries_the_whole_sdk():
    """The shippable unit must actually be complete."""
    names = {p.name for p in _guest_files()}
    assert {"__init__.py", "requests.py", "protocol.py",
            "channel.py", "sdk.py", "child.py"} <= names
