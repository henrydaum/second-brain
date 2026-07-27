"""Session-wide pytest configuration for the kernel test suite.

Redirects pytest's temporary-directory *root* into a repo-local, gitignored
folder (`.pytest_tmp/`) instead of the shared system temp.

Why this exists: on Windows the global `%TEMP%\\pytest-of-<user>` root is shared
across every tool on the machine. If it ever gets created or touched by an
elevated / different security context, a normal non-elevated `pytest` run can no
longer even `scandir` it -- and because *every* test that requests the `tmp_path`
fixture goes through `getbasetemp()`, the whole suite then errors out at setup
with `PermissionError: [WinError 5] Access is denied` (one error per test, no
code actually run). Once poisoned, that directory can't be removed without
administrator rights, so the failure is sticky and recurs on every invocation.

Pointing the temp root at a directory we own sidesteps the problem entirely and
keeps test artifacts next to the code that produced them. pytest reads
`PYTEST_DEBUG_TEMPROOT` before constructing its `tmp_path_factory`
(see `_pytest/tmpdir.py::TempPathFactory.getbasetemp`), so setting it at
conftest import time -- the earliest point a root conftest runs -- is sufficient.

`setdefault` is used so an explicit `PYTEST_DEBUG_TEMPROOT` or `--basetemp` (e.g.
CI or ad-hoc runs) still takes precedence.
"""

import os
from pathlib import Path

import pytest

_TEMP_ROOT = Path(__file__).parent / ".pytest_tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMP_ROOT))


@pytest.fixture(autouse=True)
def _isolate_llm_registry():
    """Empty the LLM registry around every test.

    The registry is module-global, which is right for a kernel with one set of
    configured models and wrong for a test suite where each test configures
    its own. Without this, a test that saves an LLM profile leaves a brain
    behind, and the next test's model resolution silently prefers it — the
    failure lands somewhere unrelated and only under a full run, which is the
    worst kind to chase.

    Boxes are closed on the way out, so a leaked one cannot become an orphaned
    process for the rest of the session.
    """
    from llm import registry

    registry._BRAINS.clear()
    registry._BACKENDS.clear()
    try:
        yield
    finally:
        for brain in list(registry._BRAINS.values()):
            try:
                brain.unload()
            except Exception:
                pass
        registry._BRAINS.clear()
        registry._BACKENDS.clear()
