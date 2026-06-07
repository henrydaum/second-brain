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

_TEMP_ROOT = Path(__file__).parent / ".pytest_tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_TEMP_ROOT))
