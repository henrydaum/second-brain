"""Warm disposable subprocess pool for canvas technique execution."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

from plugins.BaseService import BaseService

logger = logging.getLogger("TechniqueWorkerPool")
READY_TIMEOUT_S = 8.0


class TechniqueWorkerBusy(RuntimeError):
    """Raised when all warm technique workers are occupied too long."""


class TechniqueWorkerJobTimeout(RuntimeError):
    """Raised when a worker job exceeds the per-technique timeout."""


@dataclass
class TechniqueWorkerResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    memory_killed: bool = False
    peak_bytes: int = 0


class TechniqueWorkerPoolService(BaseService):
    model_name = "technique_worker_pool"
    shared = True
    config_settings = [
        ("Technique Worker Pool Enabled", "technique_worker_pool_enabled", "Use prewarmed disposable subprocesses for canvas techniques.", True, {"type": "bool"}),
        ("Technique Worker Min Idle", "technique_worker_min_idle", "Warm sandbox workers kept ready.", 1, {"type": "integer"}),
        ("Technique Worker Max Workers", "technique_worker_max_workers", "Maximum warm/active technique sandbox processes.", 2, {"type": "integer"}),
        ("Technique Worker Queue Timeout (s)", "technique_worker_queue_timeout_s", "Seconds a render waits for a technique worker before failing busy.", 10, {"type": "integer"}),
    ]

    def __init__(self, config: dict):
        super().__init__()
        self.enabled = bool(config.get("technique_worker_pool_enabled", True))
        self.min_idle = max(0, int(config.get("technique_worker_min_idle", 1)))
        self.max_workers = max(1, int(config.get("technique_worker_max_workers", 2)))
        self.min_idle = min(self.min_idle, self.max_workers - 1) if self.max_workers > 1 else 0
        self.queue_timeout_s = max(0.1, float(config.get("technique_worker_queue_timeout_s", 10)))
        self.active_limit = max(1, self.max_workers - self.min_idle)
        self._idle: queue.Queue[subprocess.Popen] = queue.Queue()
        self._active_sem = threading.BoundedSemaphore(self.active_limit)
        self._lock = threading.Lock()
        self._all: set[subprocess.Popen] = set()
        self._active = 0
        self._warming = 0
        self._stopping = False
        self._entry, self._env, self._cwd = self._worker_context()

    def _load(self) -> bool:
        self._stopping = False
        self._loaded = True
        if self.enabled:
            self._ensure_idle()
        return True

    def unload(self):
        with self._lock:
            self._stopping = True
            procs = list(self._all)
            while True:
                try: procs.append(self._idle.get_nowait())
                except queue.Empty: break
        for proc in procs:
            self._kill(proc)
        with self._lock:
            self._all.clear()
        self._loaded = False

    def run_job(self, *, job_path: str, timeout_s: float, memory_mb: int) -> TechniqueWorkerResult:
        if not self.enabled:
            raise TechniqueWorkerBusy("technique worker pool is disabled")
        if not self.loaded:
            self.load()
        deadline = time.monotonic() + self.queue_timeout_s
        sem_timeout = max(0.0, deadline - time.monotonic())
        if not self._active_sem.acquire(timeout=sem_timeout):
            raise TechniqueWorkerBusy("renderer busy; retry shortly")
        proc = None
        try:
            with self._lock:
                self._active += 1
            proc = self._take_idle(deadline)
            self._ensure_idle()
            return self._run_checked_out(proc, job_path, timeout_s, memory_mb)
        finally:
            if proc is not None:
                self._forget(proc)
            with self._lock:
                self._active = max(0, self._active - 1)
            try: self._active_sem.release()
            except ValueError: pass
            self._ensure_idle()

    def _take_idle(self, deadline: float):
        self._ensure_idle()
        timeout = max(0.0, deadline - time.monotonic())
        try:
            return self._idle.get(timeout=timeout)
        except queue.Empty:
            raise TechniqueWorkerBusy("renderer busy; no warm technique worker became ready")

    def _ensure_idle(self):
        if not self.enabled or not self.loaded:
            return
        with self._lock:
            while (
                not self._stopping
                and self._idle.qsize() + self._warming < self.min_idle
                and self._active + self._idle.qsize() + self._warming < self.max_workers
            ):
                self._warming += 1
                threading.Thread(target=self._warm_one, daemon=True, name="technique-worker-warm").start()

    def _warm_one(self):
        proc = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-B", str(self._entry), "--worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=self._env, cwd=self._cwd,
            )
            with self._lock:
                self._all.add(proc)
            line = self._readline_timeout(proc, READY_TIMEOUT_S)
            if line.strip() != b"READY":
                raise RuntimeError(f"worker did not become ready: {line!r}")
            with self._lock:
                if self._stopping:
                    raise RuntimeError("pool stopping")
            self._idle.put(proc)
            proc = None
        except Exception as e:
            logger.debug("technique worker warm failed: %s", e)
            if proc is not None:
                self._kill(proc)
        finally:
            with self._lock:
                self._warming = max(0, self._warming - 1)
            self._ensure_idle()

    def _run_checked_out(self, proc, job_path: str, timeout_s: float, memory_mb: int) -> TechniqueWorkerResult:
        stop = threading.Event()
        mem = {"killed": False, "peak": 0}
        wd = self._watch_memory(proc, max(64, int(memory_mb)) * 1024 * 1024, stop, mem)
        try:
            try:
                stdout, stderr = proc.communicate(input=(job_path + "\n").encode("utf-8"), timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._kill(proc)
                raise TechniqueWorkerJobTimeout(f"exceeded {timeout_s:.0f}s timeout")
        finally:
            stop.set()
            if wd is not None:
                wd.join(timeout=1.0)
        return TechniqueWorkerResult(proc.returncode, stdout or b"", stderr or b"", mem["killed"], mem["peak"])

    def _watch_memory(self, proc, cap: int, stop: threading.Event, mem: dict):
        if psutil is None:
            return None
        def run():
            try: p = psutil.Process(proc.pid)
            except psutil.Error: return
            while not stop.wait(0.2):
                try:
                    rss = p.memory_info().rss + sum((c.memory_info().rss for c in p.children(recursive=True)), 0)
                except psutil.Error:
                    return
                mem["peak"] = max(mem["peak"], rss)
                if rss > cap:
                    mem["killed"] = True
                    self._kill(proc)
                    return
        t = threading.Thread(target=run, daemon=True, name=f"technique-worker-mem-{proc.pid}")
        t.start()
        return t

    def _readline_timeout(self, proc, timeout: float) -> bytes:
        out: queue.Queue[bytes] = queue.Queue(maxsize=1)
        threading.Thread(target=lambda: out.put(proc.stdout.readline() if proc.stdout else b""), daemon=True).start()
        try:
            return out.get(timeout=timeout)
        except queue.Empty:
            self._kill(proc)
            return b""

    def _forget(self, proc):
        with self._lock:
            self._all.discard(proc)

    def _kill(self, proc):
        try:
            if proc.poll() is None:
                proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    def _worker_context(self):
        root = Path(__file__).resolve().parents[2]
        entry = root / "plugins" / "techniques" / "helpers" / "technique_sandbox_entry.py"
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
        return entry, env, str(root)


def build_services(config: dict) -> dict:
    return {"technique_worker_pool": TechniqueWorkerPoolService(config or {})}
