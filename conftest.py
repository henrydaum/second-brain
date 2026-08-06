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
def _isolate_config_files(tmp_path_factory):
    """Point config writes at a scratch file, for every test, always.

    ``config_manager.save`` resolves ``path=None`` to a module-level default,
    and a great deal of code saves without naming a path — ``config.write``'s
    handler, the approval dialog's "always allow", ``rehome_kernel_keys``. So
    any test that reaches one of those writes the *developer's real config*,
    silently and with no ledger row, because the ledger database is not wired
    outside bootstrap.

    That is not hypothetical: a hook test that exercised ``config.write``
    end-to-end overwrote a live ``sync_directories`` with its fixture value,
    and the damage surfaced later as a settings-changed notification in the
    running app naming keys nobody had touched. Reading the value back is not
    a defence either — the whole point of such a test is to write.

    Autouse and session-scoped path, so it costs one directory and no test has
    to remember. A test that genuinely wants the real file can patch these
    back deliberately, which is a visible act rather than an omission.

    Two details are not optional. The files are **created**, because a policy
    that refuses to read the config is tested by reading it — pointed at a path
    that does not exist, those tests get ``not_found`` instead of ``denied``
    and pass for the wrong reason or fail for one. And
    ``sandbox.protected.reset()`` is called on both edges, because
    ``protected_paths`` is ``lru_cache``d and reads these same constants at
    first call: without the reset the protected set is whatever the first test
    to touch it happened to see, which makes a *security* check depend on test
    ordering.
    """
    from config import config_manager
    from sandbox import protected

    scratch = tmp_path_factory.mktemp("config")
    saved_core = config_manager._DEFAULT_CONFIG_PATH
    saved_plugin = config_manager._DEFAULT_PLUGIN_CONFIG_PATH
    core = scratch / "config.json"
    plugin = scratch / "plugin_config.json"
    for path in (core, plugin):
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
    config_manager._DEFAULT_CONFIG_PATH = str(core)
    config_manager._DEFAULT_PLUGIN_CONFIG_PATH = str(plugin)
    protected.reset()
    try:
        yield scratch
    finally:
        config_manager._DEFAULT_CONFIG_PATH = saved_core
        config_manager._DEFAULT_PLUGIN_CONFIG_PATH = saved_plugin
        protected.reset()


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
