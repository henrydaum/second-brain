"""Boot a full kernel headlessly for stress testing.

This mirrors ``main.pyw``'s composition root but:

- targets a **throwaway SQLite DB** (so fuzzing never touches real data),
- starts **no frontends** and (by default) **no background watcher/event-trigger
  threads** (we drive turns synchronously via ``runtime.iterate_agent_turn``),
- lets the caller inject an **LLM** — a network-free fake for fuzzing, or
  ``None`` to use whatever real backend the on-disk config has configured.

The returned :class:`Kernel` is a small bag of the wired references plus a
``close()`` that tears everything down cleanly so the invariant checker can
assert nothing leaked.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Settle the runtime<->state_machine package-init cycle before importing the
# runtime (state_machine/__init__ pulls the runtime in). Mirrors the tests.
import state_machine  # noqa: F401

from config import config_manager
from pipeline.database import Database
from pipeline.orchestrator import Orchestrator
from agent.tool_registry import ToolRegistry
from plugins.BaseService import should_autoload_service
from plugins.plugin_discovery import (
    discover_services,
    discover_tasks,
    discover_tools,
    get_plugin_settings,
)
from runtime.bootstrap import _conversation_runtime

_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Kernel:
    """Wired, headless kernel under test."""

    runtime: Any
    db: Any
    orchestrator: Any
    tool_registry: Any
    services: dict
    config: dict
    root_dir: Path
    _tempdir: Any = None
    _started_orchestrator: bool = False
    _extra_closers: list = field(default_factory=list)

    def close(self) -> None:
        """Tear down in reverse order of construction. Best-effort, never raises."""
        for fn in reversed(self._extra_closers):
            _safely(fn)
        if self._started_orchestrator and self.orchestrator is not None:
            _safely(self.orchestrator.stop)
        for svc in self.services.values():
            if getattr(svc, "loaded", False):
                _safely(svc.unload)
        # Close the SQLite handle before removing the tempdir — Windows refuses
        # to unlink a file with an open handle (WinError 32).
        if self.db is not None and getattr(self.db, "conn", None) is not None:
            _safely(self.db.conn.close)
        if self._tempdir is not None:
            # ignore_errors: WAL sidecar files can linger briefly on Windows.
            import shutil
            _safely(lambda: shutil.rmtree(self._tempdir.name, ignore_errors=True))
            self._tempdir._finalizer.detach()


def _safely(fn) -> None:
    try:
        fn()
    except Exception:
        pass


def boot_kernel(
    *,
    llm: Any = None,
    use_real_config: bool = False,
    start_orchestrator: bool = False,
    autoload_services: bool = True,
    config_overrides: dict | None = None,
    data_dir: str | Path | None = None,
) -> Kernel:
    """Boot a headless kernel.

    Args:
        llm: an LLM instance to inject as the ``llm`` service (e.g. a
            :class:`stress.fake_llm.MonkeyLLM`). When provided, the LLM-profile
            config is neutralised so the agent loop resolves straight to it.
            When ``None`` the real configured backend is used (network).
        use_real_config: start from the on-disk config (real LLM profiles, etc.)
            instead of schema defaults. Always overridden to a temp DB.
        start_orchestrator: start the pipeline orchestrator's background loop.
            Off by default — the kernel ships zero pipeline tasks, and leaving it
            off keeps the thread census deterministic for the invariant checker.
        autoload_services: run the same autoload pass main.pyw does.
        config_overrides: shallow-merged over the resolved config last.
        data_dir: when given, use this directory for the DB/sync surface and do
            NOT auto-delete it. This makes a kernel **persistent across process
            invocations** (the driver relies on it to drive turns one shell call
            at a time). When omitted, a throwaway tempdir is used and cleaned up.
    """
    if data_dir is not None:
        tempdir = None
        tmp = Path(data_dir)
        tmp.mkdir(parents=True, exist_ok=True)
    else:
        tempdir = tempfile.TemporaryDirectory(prefix="sb_stress_")
        tmp = Path(tempdir.name)

    if use_real_config:
        config = config_manager.load()
        config_manager.load_plugin_config_early(config)
    else:
        config = dict(config_manager.DEFAULTS)

    # Always isolate persistence and sync surface.
    config["db_path"] = str(tmp / "stress.db")
    config["sync_directories"] = [str(tmp / "sync")]
    (tmp / "sync").mkdir(parents=True, exist_ok=True)
    config["_root"] = str(_ROOT)

    if llm is not None:
        # Route the agent loop straight to the injected fake (see
        # runtime.agent_scope.resolve_agent_llm: empty default + no profiles
        # falls back to services["llm"]).
        config["llm_profiles"] = {}
        config["default_llm_profile"] = ""

    if config_overrides:
        config.update(config_overrides)

    db = Database(config["db_path"])
    services = discover_services(_ROOT, config)

    if llm is not None:
        services["llm"] = llm
    elif autoload_services:
        for name, svc in services.items():
            if should_autoload_service(name, svc, config):
                _safely(svc.load)

    orchestrator = Orchestrator(db, config, services)
    discover_tasks(_ROOT, orchestrator, config)

    tool_registry = ToolRegistry(db, config, services)
    tool_registry.orchestrator = orchestrator
    orchestrator.tool_registry = tool_registry
    discover_tools(_ROOT, tool_registry, config)

    config_manager.reconcile_plugin_config(config, get_plugin_settings())

    started = False
    if start_orchestrator:
        orchestrator.start()
        started = True

    scaffold = _Scaffold(orchestrator=orchestrator, db=db)

    def _noop():
        return None

    runtime = _conversation_runtime(
        scaffold, _noop, tool_registry, services, config, _ROOT
    )

    # Bind runtime into services the way main.pyw's _bind_runtime_services does,
    # so service-driven code paths (compactor, watcher) can reach it.
    for svc in services.values():
        if hasattr(svc, "bind_runtime"):
            _safely(lambda s=svc: s.bind_runtime(
                tool_registry=tool_registry,
                orchestrator=orchestrator,
                runtime=runtime,
                command_registry=getattr(runtime, "command_registry", None),
                frontend_manager=None,
            ))

    return Kernel(
        runtime=runtime,
        db=db,
        orchestrator=orchestrator,
        tool_registry=tool_registry,
        services=services,
        config=config,
        root_dir=_ROOT,
        _tempdir=tempdir,
        _started_orchestrator=started,
    )


@dataclass
class _Scaffold:
    """Minimal stand-in for main.pyw's Scaffold (only what bootstrap reads)."""
    orchestrator: Any = None
    db: Any = None
    restart: Any = None
