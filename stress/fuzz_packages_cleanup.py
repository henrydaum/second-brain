"""Adversarial fuzzer for the package-store *uninstall cleanup* flow.

This targets the hairiest path in the store: the multi-step uninstall form
(action → package → "remove config? / tables? / pip?" each all/none/specific →
per-package plugin-by-plugin prompts when "specific") and the cleanup engine
behind it (`build_uninstall_plan`, `_cleanup_plan`, `_safe_pip_removals`,
`execute_uninstall_plan`).

It drives the **real `PackagesCommand` form** the way the form engine does —
calling `form(args)` repeatedly and filling each new step — so form/run drift is
exercised, not bypassed. Then it asserts the two safety properties that, if
violated, mean data loss or a broken kernel:

1. **Shared state is never collateral.** Uninstalling a package must never
   delete a config setting / drop a SQL table / pip-uninstall a library that a
   *still-installed* package also declares.
2. **Kernel requirements are never pip-removed.** A dependency listed in
   `requirements.txt` (e.g. ``litellm``) must never be uninstalled.

The fake catalog deliberately wires shared declarations (`shared_key`,
`shared_tbl`, `shared-lib`) and a kernel-req pip (`litellm`) across packages so
those keep-paths are actually hit.

Run: ``pytest stress/fuzz_packages_cleanup.py -q``
"""

from __future__ import annotations

import random
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
import hypothesis.strategies as st

from plugins.commands.helpers import package_manager as pm
from plugins.commands.command_packages import PackagesCommand
from plugins import plugin_discovery
import plugins.helpers.plugin_paths as plugin_paths
import config.config_manager as config_manager


def _src(settings=(), reads=(), writes=()) -> bytes:
    parts = ["VALUE = 1"]
    if settings:
        parts.append("config_settings = [" + ", ".join(f'("T", {k!r}, "d", "x", {{}})' for k in settings) + "]")
    if reads:
        parts.append(f"reads = {list(reads)!r}")
    if writes:
        parts.append(f"writes = {list(writes)!r}")
    return ("\n".join(parts) + "\n").encode()


# id -> dict(files={rel: bytes}, requires=[], pip=[], settings=set, tables=set)
def _pkg(rel, *, settings=(), reads=(), writes=(), requires=(), pip=()):
    return {
        "files": {rel: _src(settings, reads, writes)},
        "requires": list(requires),
        "pip": list(pip),
        "settings": set(settings),
        "tables": set(writes),  # only writes create/drop tables
    }


CATALOG = {
    "p-alpha": _pkg("tools/tool_alpha.py", settings=["alpha_key"], writes=["alpha_tbl"], pip=["alpha-lib"]),
    "p-beta":  _pkg("tasks/task_beta.py", settings=["beta_key", "shared_key"], writes=["shared_tbl"], pip=["beta-lib", "shared-lib"]),
    "p-gamma": _pkg("tasks/task_gamma.py", settings=["shared_key"], reads=["shared_tbl"], pip=["shared-lib"]),
    "p-delta": _pkg("tools/tool_delta.py", settings=["delta_key"], writes=["delta_tbl"], pip=["watchdog"]),  # watchdog is a real kernel requirement
    # A bundle whose member shares nothing, to mix pruning into the flow.
    "p-bundle": {"files": {}, "requires": ["p-alpha"], "pip": [], "settings": set(), "tables": set()},
}
PACKAGE_IDS = sorted(CATALOG)
CLEANUP_MODES = ["all", "none", "specific"]


class _FakeBackend:
    def get_manifest(self, pid):
        if pid not in CATALOG:
            raise pm.PackageError(f"missing manifest: {pid}")
        c = CATALOG[pid]
        return {"id": pid, "name": pid, "description": "", "requires": list(c["requires"]),
                "files": list(c["files"]), "pip": list(c["pip"])}

    def get_manifest_bytes(self, pid):
        import json
        return json.dumps(self.get_manifest(pid), sort_keys=True).encode()

    def get_file_bytes(self, pid, rel):
        return CATALOG[pid]["files"][rel]


class PackageCleanupStateMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self._tmp = tempfile.TemporaryDirectory(prefix="sb_pkgclean_")
        tmp = Path(self._tmp.name)
        self.installed_root = tmp / "installed_plugins"
        self.receipts = tmp / "packages" / "receipts"
        self.installed_root.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)
        self.root_dir = tmp

        # In-memory plugin config + db + recorded pip calls stand in for the
        # global side effects the cleanup engine would otherwise perform.
        self.plugin_config: dict[str, str] = {}
        self.db = _MemDb()
        self.pip_uninstalled: list[str] = []

        roots = (plugin_paths.PluginRoot("installed", self.installed_root, "installed_plugins"),)
        pc = dict(plugin_paths.PLUGIN_CONFIG)
        pc["tool"] = (plugin_paths.PluginDir(roots[0], "tool", "tools", "tool_"),)
        pc["task"] = (plugin_paths.PluginDir(roots[0], "task", "tasks", "task_"),)
        self._saved = {
            (pm, "INSTALLED_PLUGINS"): pm.INSTALLED_PLUGINS,
            (pm, "RECEIPTS_DIR"): pm.RECEIPTS_DIR,
            (pm, "GitStoreBackend"): pm.GitStoreBackend,
            (pm, "_install_python_packages"): pm._install_python_packages,
            (pm, "subprocess"): pm.subprocess,
            (plugin_paths, "PLUGIN_ROOTS"): plugin_paths.PLUGIN_ROOTS,
            (plugin_paths, "PLUGIN_CONFIG"): plugin_paths.PLUGIN_CONFIG,
            (plugin_discovery, "PLUGIN_ROOTS"): plugin_discovery.PLUGIN_ROOTS,
            (config_manager, "load_plugin_config"): config_manager.load_plugin_config,
            (config_manager, "save_plugin_config"): config_manager.save_plugin_config,
        }
        pm.INSTALLED_PLUGINS = self.installed_root
        pm.RECEIPTS_DIR = self.receipts
        pm.GitStoreBackend = lambda _root: _FakeBackend()
        pm._install_python_packages = lambda packages, progress=None: None  # no real pip
        pm.subprocess = _PipRecorder(self.pip_uninstalled)
        plugin_paths.PLUGIN_ROOTS = roots
        plugin_paths.PLUGIN_CONFIG = pc
        plugin_discovery.PLUGIN_ROOTS = roots
        config_manager.load_plugin_config = lambda path=None: dict(self.plugin_config)
        config_manager.save_plugin_config = lambda data, path=None: self.plugin_config.clear() or self.plugin_config.update(data)

    # ── helpers ──────────────────────────────────────────────────────

    def _context(self):
        registry = SimpleNamespace(tools={}, register=lambda t: None, unregister=lambda n: None)
        return SimpleNamespace(
            root_dir=self.root_dir, tool_registry=registry, orchestrator=None,
            services={}, config=dict(self.plugin_config), command_registry=None,
            runtime=None, request_user_input=None, db=self.db,
        )

    def _installed(self) -> dict[str, dict]:
        return {r["id"]: r for r in pm.installed_packages()}

    def _seed_state_for(self, pid):
        """Mirror the side effects a real install of pid would have produced."""
        for key in CATALOG[pid]["settings"]:
            self.plugin_config.setdefault(key, "v")
        for tbl in CATALOG[pid]["tables"]:
            self.db.create_table(tbl)

    def _declared_by_remaining(self, kind: str) -> set[str]:
        """Union of settings/tables/pip declared by all currently-installed packages."""
        out: set[str] = set()
        for pid in self._installed():
            if pid in CATALOG:
                if kind == "settings":
                    out |= CATALOG[pid]["settings"]
                elif kind == "tables":
                    out |= CATALOG[pid]["tables"]
                elif kind == "pip":
                    out |= {pm._normalize_pip(n) for n in CATALOG[pid]["pip"]}
        return out

    # ── rules ────────────────────────────────────────────────────────

    @rule(pid=st.sampled_from(PACKAGE_IDS))
    def install(self, pid):
        if pid in self._installed():
            return
        result = pm.install_package(self.root_dir, pid, self._context())
        assert result.ok, f"install {pid} failed: {result.text}"
        # Seed the side-effect state for everything that became installed.
        for installed_id in self._installed():
            if installed_id in CATALOG:
                self._seed_state_for(installed_id)

    @rule(pid=st.sampled_from(PACKAGE_IDS),
          mode_cfg=st.sampled_from(CLEANUP_MODES),
          mode_tbl=st.sampled_from(CLEANUP_MODES),
          mode_pip=st.sampled_from(CLEANUP_MODES),
          seed=st.integers(0, 2**16))
    def uninstall_via_form(self, pid, mode_cfg, mode_tbl, mode_pip, seed):
        if pid not in self._installed():
            return
        # Packages with installed dependents can't be uninstalled standalone;
        # refusal-ordering is covered by fuzz_packages.py. This fuzzer targets
        # the cleanup flow, so skip that combination here.
        if any(pid in CATALOG[d]["requires"] for d in self._installed() if d in CATALOG):
            return
        rng = random.Random(seed)
        pip_before = len(self.pip_uninstalled)
        cmd = PackagesCommand()
        modes = {"cleanup_config": mode_cfg, "cleanup_tables": mode_tbl, "cleanup_pip": mode_pip}

        # Drive the real dynamic form: fill each new step until stable.
        args = {"action": "uninstall", "package_id": pid}
        for _ in range(12):
            steps = cmd.form(args, self._context())
            missing = [s for s in steps if s.name not in args]
            if not missing:
                break
            for s in missing:
                if s.name in modes:
                    args[s.name] = modes[s.name]
                elif s.name.startswith(("cleanup_config__", "cleanup_tables__", "cleanup_pip__")):
                    args[s.name] = rng.choice([True, False])
                else:
                    args[s.name] = args.get(s.name)

        result = cmd.run(args, self._context())
        assert isinstance(result, str)
        assert pid not in self._installed(), f"{pid} still installed after uninstall: {result}"

        # Per-operation pip safety: nothing removed by *this* uninstall may be a
        # kernel requirement or a pip still needed by a package left installed.
        removed_now = {pm._normalize_pip(n) for n in self.pip_uninstalled[pip_before:]}
        kernel = pm._kernel_requirements()
        assert not (removed_now & kernel), f"kernel-req pip removed: {removed_now & kernel}"
        still_needed = self._declared_by_remaining("pip")
        assert not (removed_now & still_needed), \
            f"pip still needed by an installed package was removed: {removed_now & still_needed}"

    # ── oracle ───────────────────────────────────────────────────────

    @invariant()
    def shared_state_is_never_collateral(self):
        # Every config key / table declared by a still-installed package must
        # still exist (pip safety is checked per-operation in the rule, since the
        # recorded-removals list is cumulative across re-installs).
        for key in self._declared_by_remaining("settings"):
            assert key in self.plugin_config, f"config '{key}' deleted but still declared by an installed package"
        for tbl in self._declared_by_remaining("tables"):
            assert self.db.has_table(tbl), f"table '{tbl}' dropped but still declared by an installed package"

    def teardown(self):
        for (module, attr), value in self._saved.items():
            setattr(module, attr, value)
        try:
            self._tmp.cleanup()
        except Exception:
            import shutil
            shutil.rmtree(self._tmp.name, ignore_errors=True)


class _MemDb:
    """Minimal Database stand-in: a real in-memory sqlite + the bits cleanup uses."""
    _validate_identifier = staticmethod(lambda name: None)

    def __init__(self):
        import threading
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.lock = threading.Lock()

    def create_table(self, name):
        with self.lock:
            self.conn.execute(f'CREATE TABLE IF NOT EXISTS "{name}" (id INTEGER)')
            self.conn.commit()

    def has_table(self, name) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
        return row is not None


class _PipRecorder:
    """Stands in for package_manager.subprocess; records pip uninstall targets."""
    def __init__(self, sink: list[str]):
        self._sink = sink

    def run(self, cmd, **kwargs):
        if "uninstall" in cmd:
            i = cmd.index("-y") + 1 if "-y" in cmd else len(cmd)
            self._sink.extend(cmd[i:])
        return subprocess.CompletedProcess(cmd, 0, "", "")


PackageCleanupStateMachine.TestCase.settings = settings(
    max_examples=60,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

TestPackageCleanupFuzz = PackageCleanupStateMachine.TestCase
