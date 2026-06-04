"""Stateful fuzzer for the package store (install / uninstall).

The store is the kernel's next big surface (browse → install → uninstall against
a cloud catalog), and it's exactly where *ordering* bugs live: install a
dependency, install something that needs it, refuse to uninstall the dependency,
prune it once its dependents go away. So it gets the syzkaller treatment too.

Approach mirrors ``fuzz_runtime`` but against ``package_manager``:

- A small fake catalog with files + ``requires`` edges (incl. a meta-package /
  bundle that ships no files) backs a stub store backend — no git, no network.
- The install root, receipts dir, and plugin roots are redirected to a tempdir
  (the same redirection ``tests/test_package_store.py::_patch_install_root``
  uses), restored on teardown.
- Rules: ``install`` and ``uninstall`` arbitrary catalog packages.
- **Receipts on disk are the ground truth** — the oracle re-derives state from
  them every step and checks store integrity:
    * every file a receipt claims exists on disk, and
    * every file on disk is owned by some receipt (no orphans), and
    * every ``requires`` edge of an installed receipt points at another
      installed package (no dangling dependency).
- Per-op assertions cross-check the documented contract: install pulls the full
  dependency closure; uninstall refuses while dependents remain and is rejected
  for not-installed ids.

Run: ``pytest stress/fuzz_packages.py -q``
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
import hypothesis.strategies as st

from plugins.commands.helpers import package_manager as pm
from plugins import plugin_discovery
import plugins.helpers.plugin_paths as plugin_paths
from stress.invariants import Violation


# ── fake catalog ────────────────────────────────────────────────────────
# id -> (files, requires). A file path must live under an allowed plugin root.
CATALOG: dict[str, tuple[list[str], list[str]]] = {
    "base":    (["helpers/base.txt"], []),
    "lib":     (["helpers/lib.txt"], []),
    "tool-a":  (["tools/tool_a.py"], ["base"]),
    "tool-b":  (["tools/tool_b.py"], ["base"]),
    "tool-c":  (["tools/tool_c.py"], ["lib"]),
    "bundle":  ([], ["tool-a", "tool-c"]),          # meta-package, no files
    "mega":    ([], ["bundle", "tool-b"]),          # bundle-of-bundles
}
PACKAGE_IDS = sorted(CATALOG)


def _transitive_requires(pid: str, seen: set[str] | None = None) -> set[str]:
    seen = seen if seen is not None else set()
    for dep in CATALOG[pid][1]:
        if dep not in seen:
            seen.add(dep)
            _transitive_requires(dep, seen)
    return seen


class _FakeBackend:
    """Stub of GitStoreBackend over CATALOG (matches the test's _Backend)."""

    def _manifest(self, pid: str) -> dict:
        if pid not in CATALOG:
            raise pm.PackageError(f"missing manifest: {pid}")
        files, requires = CATALOG[pid]
        return {"id": pid, "name": pid, "description": "", "requires": list(requires), "files": list(files)}

    def get_manifest(self, pid: str) -> dict:
        return self._manifest(pid)

    def get_manifest_bytes(self, pid: str) -> bytes:
        return json.dumps(self._manifest(pid), sort_keys=True).encode()

    def get_file_bytes(self, pid: str, rel_path: str) -> bytes:
        # Trivial, import-free content so the pip auto-detect finds nothing.
        return f"# {pid}:{rel_path}\nVALUE = 1\n".encode()


def _context(root_dir: Path):
    registry = SimpleNamespace(
        tools={},
        register=lambda tool: None,
        unregister=lambda name: None,
    )
    return SimpleNamespace(
        root_dir=root_dir, tool_registry=registry, orchestrator=None,
        services={}, config={}, command_registry=None, runtime=None,
        request_user_input=None, db=None,
    )


class PackageStoreStateMachine(RuleBasedStateMachine):
    """Random install/uninstall sequences against the fake store."""

    def __init__(self):
        super().__init__()
        self._tmp = tempfile.TemporaryDirectory(prefix="sb_pkgfuzz_")
        tmp = Path(self._tmp.name)
        self.installed_root = tmp / "installed_plugins"
        self.receipts = tmp / "packages" / "receipts"
        self.installed_root.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)

        # Redirect the package manager + discovery at the tempdir, saving
        # originals for teardown. Same surface as tests/_patch_install_root.
        roots = (plugin_paths.PluginRoot("installed", self.installed_root, "installed_plugins"),)
        config = dict(plugin_paths.PLUGIN_CONFIG)
        config["tool"] = (plugin_paths.PluginDir(roots[0], "tool", "tools", "tool_"),)
        config["task"] = (plugin_paths.PluginDir(roots[0], "task", "tasks", "task_"),)
        self._saved = {
            (pm, "INSTALLED_PLUGINS"): pm.INSTALLED_PLUGINS,
            (pm, "RECEIPTS_DIR"): pm.RECEIPTS_DIR,
            (pm, "GitStoreBackend"): pm.GitStoreBackend,
            (plugin_paths, "PLUGIN_ROOTS"): plugin_paths.PLUGIN_ROOTS,
            (plugin_paths, "PLUGIN_CONFIG"): plugin_paths.PLUGIN_CONFIG,
            (plugin_discovery, "PLUGIN_ROOTS"): plugin_discovery.PLUGIN_ROOTS,
        }
        pm.INSTALLED_PLUGINS = self.installed_root
        pm.RECEIPTS_DIR = self.receipts
        pm.GitStoreBackend = lambda _root: _FakeBackend()
        plugin_paths.PLUGIN_ROOTS = roots
        plugin_paths.PLUGIN_CONFIG = config
        plugin_discovery.PLUGIN_ROOTS = roots

        self.root_dir = tmp

    # ── helpers ──────────────────────────────────────────────────────

    def _receipts(self) -> dict[str, dict]:
        return {r["id"]: r for r in pm.installed_packages()}

    # ── rules ────────────────────────────────────────────────────────

    @rule(pid=st.sampled_from(PACKAGE_IDS))
    def install(self, pid):
        before = self._receipts()
        if pid in before:
            # Re-installing an already-installed package (as an explicit request)
            # is refused, and must be a no-op.
            try:
                pm.install_package(self.root_dir, pid, _context(self.root_dir))
                raised = False
            except pm.PackageError:
                raised = True
            assert raised, f"re-installing already-installed {pid} should raise"
            assert self._receipts().keys() == before.keys()
            return
        result = pm.install_package(self.root_dir, pid, _context(self.root_dir))
        assert result.ok, f"install {pid} failed: {result.text}"
        receipts = self._receipts()
        # The package and its entire dependency closure must be installed.
        assert pid in receipts
        for dep in _transitive_requires(pid):
            assert dep in receipts, f"install {pid} left dependency {dep} uninstalled"

    @rule(pid=st.sampled_from(PACKAGE_IDS))
    def uninstall(self, pid):
        before = self._receipts()
        dependents = sorted(d for d in before if pid in (CATALOG[d][1]))
        if pid not in before:
            try:
                pm.uninstall_package(pid, _context(self.root_dir))
                raised = False
            except pm.PackageError:
                raised = True
            assert raised, f"uninstalling not-installed {pid} should raise"
            return
        if dependents:
            try:
                pm.uninstall_package(pid, _context(self.root_dir))
                raised = False
            except pm.PackageError:
                raised = True
            assert raised, f"uninstall {pid} should be refused; depended on by {dependents}"
            # Refusal must be a no-op.
            assert self._receipts().keys() == before.keys()
            return
        result = pm.uninstall_package(pid, _context(self.root_dir))
        assert result.ok
        assert pid not in self._receipts()

    # ── oracle ───────────────────────────────────────────────────────

    @invariant()
    def store_is_consistent(self):
        violations = self._check_store()
        assert not violations, "Store invariant(s) broken:\n" + "\n".join(map(str, violations))

    def _check_store(self) -> list[Violation]:
        out: list[Violation] = []
        receipts = self._receipts()

        owned: dict[str, str] = {}
        for pid, receipt in receipts.items():
            for f in receipt.get("files", []):
                rel = f["path"] if isinstance(f, dict) else f
                # No two receipts may own the same file.
                if rel in owned:
                    out.append(Violation("store.double_owned", f"{rel} owned by {owned[rel]} and {pid}"))
                owned[rel] = pid
                if not (self.installed_root / rel).exists():
                    out.append(Violation("store.missing_file", f"{pid} receipt claims absent file {rel}"))
            # Every requires edge must resolve to an installed package.
            for dep in receipt.get("requires", []):
                if dep not in receipts:
                    out.append(Violation("store.dangling_requires", f"{pid} requires absent {dep}"))

        # No file on disk without an owning receipt.
        for path in self.installed_root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(self.installed_root).as_posix()
                if rel not in owned:
                    out.append(Violation("store.orphan_file", f"{rel} on disk owns no receipt"))
        return out

    def teardown(self):
        for (module, attr), value in self._saved.items():
            setattr(module, attr, value)
        try:
            self._tmp.cleanup()
        except Exception:
            import shutil
            shutil.rmtree(self._tmp.name, ignore_errors=True)


PackageStoreStateMachine.TestCase.settings = settings(
    max_examples=50,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

TestPackageStoreFuzz = PackageStoreStateMachine.TestCase
