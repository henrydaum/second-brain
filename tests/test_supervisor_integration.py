"""End-to-end: intentionally broken tools and tasks must not break the kernel.

Drives the REAL ToolRegistry and Orchestrator execution paths with plugins that
hang and crash, and asserts the kernel (1) survives every crash, (2) abandons a
hang instead of wedging, (3) releases the worker slot a hang would otherwise
hold forever, and (4) quarantines a repeat offender after three strikes.

Run with ``-s`` to watch the narrative.
"""

import threading
import time

import pytest

from agent.tool_registry import ToolRegistry
from events.event_bus import bus
from events.event_channels import PLUGIN_QUARANTINE_REQUESTED
from pipeline.orchestrator import Orchestrator
from plugins.BaseTask import BaseTask, TaskResult
from plugins.BaseTool import BaseTool, ToolResult
from runtime.supervisor import supervisor

# Non-built-in paths → quarantine-eligible (a real sandbox/installed plugin).
TOOL_SRC = r"C:\fake\sandbox_plugins\tools\tool_broken.py"
TASK_SRC = r"C:\fake\sandbox_plugins\tasks\task_broken.py"

# Lets an abandoned "hang" thread exit promptly at test teardown instead of
# blocking interpreter shutdown for the full sleep.
_hang_release = threading.Event()


@pytest.fixture(autouse=True)
def _enabled():
    supervisor.configure({"plugin_supervisor": True})
    _hang_release.clear()
    yield
    _hang_release.set()
    supervisor.health._strikes.clear()
    supervisor.health._quarantined.clear()


def _capture_quarantine():
    events = []
    return events, bus.subscribe(PLUGIN_QUARANTINE_REQUESTED, lambda p: events.append(p))


# ── Broken tools ────────────────────────────────────────────────────

class HangTool(BaseTool):
    """Blocks forever — would wedge the kernel without the timeout."""
    name = "hang_tool"

    def run(self, context, **kwargs):
        _hang_release.wait(timeout=30)
        return ToolResult(llm_summary="never reached")


class CrashTool(BaseTool):
    """Raises on every call."""
    name = "crash_tool"

    def run(self, context, **kwargs):
        raise RuntimeError("boom")


def test_hanging_tool_is_abandoned_not_blocking():
    reg = ToolRegistry(None, {"tool_timeout": 1}, {})
    tool = HangTool()
    tool._source_path = TOOL_SRC
    reg.register(tool)

    t0 = time.time()
    res = reg.call("hang_tool")
    elapsed = time.time() - t0

    assert not res.success and "timed out" in res.error
    assert elapsed < 5, "kernel was wedged by the hanging tool"
    print(f"\n[hang_tool] abandoned after {elapsed:.2f}s -> {res.error}")


def test_crashing_tool_quarantined_after_three_strikes():
    reg = ToolRegistry(None, {"tool_timeout": 5}, {})
    tool = CrashTool()
    tool._source_path = TOOL_SRC
    reg.register(tool)

    events, unsub = _capture_quarantine()
    try:
        for i in range(3):
            res = reg.call("crash_tool")
            assert not res.success  # every crash is caught; kernel survives
            print(f"[crash_tool] call {i + 1}: {res.error!r}; quarantine_events={len(events)}")
        assert len(events) == 1
        assert events[0]["name"] == "crash_tool" and events[0]["plugin_type"] == "tool"
        print(f"[crash_tool] quarantined: {events[0]['reason']}")
    finally:
        unsub()


# ── Broken tasks ────────────────────────────────────────────────────

class _FakeDB:
    """Minimal DB surface the orchestrator's failure path touches."""
    def __init__(self):
        self.failures = []

    def register_task(self, **kwargs):
        pass

    def ensure_output_table(self, *a, **k):
        pass

    def fail_task(self, path, name, error):
        self.failures.append((path, name, error))

    def get_user_config(self, user_id):
        return {}


class HangTask(BaseTask):
    """Blocks forever — without a wall-clock kill this permanently leaks a worker."""
    name = "hang_task"
    timeout = 1

    def run(self, paths, context):
        _hang_release.wait(timeout=30)
        return [TaskResult() for _ in paths]


class CrashTask(BaseTask):
    """Raises on every batch."""
    name = "crash_task"
    timeout = 5

    def run(self, paths, context):
        raise RuntimeError("task boom")


def test_hanging_task_releases_worker_slot():
    db = _FakeDB()
    orch = Orchestrator(db, {"max_workers": 1})
    try:
        task = HangTask()
        task._source_path = TASK_SRC
        orch.register_task(task)
        sem = orch.task_semaphores["hang_task"]

        # Simulate dispatch: claim the only worker slot, then run the wrapper
        # exactly as the dispatch loop does.
        assert sem.acquire(blocking=False)
        t0 = time.time()
        orch._execute_wrapper(task, ["/file"], sem)
        elapsed = time.time() - t0

        assert elapsed < 5, "hang was not abandoned"
        assert db.failures and db.failures[0][1] == "hang_task"
        # The decisive check: the slot came back. A hang did NOT permanently
        # consume the worker (the bug this whole effort closes).
        assert sem.acquire(blocking=False), "worker slot leaked by the hang"
        print(f"\n[hang_task] slot released after {elapsed:.2f}s; failures={db.failures}")
    finally:
        orch.stop()


def test_crashing_task_quarantined_after_three_strikes():
    db = _FakeDB()
    orch = Orchestrator(db, {"max_workers": 1})
    events, unsub = _capture_quarantine()
    try:
        task = CrashTask()
        task._source_path = TASK_SRC
        orch.register_task(task)
        for i in range(3):
            orch._execute(task, ["/file"])
            print(f"[crash_task] run {i + 1}: failures={len(db.failures)}, quarantine_events={len(events)}")
        assert len(events) == 1
        assert events[0]["name"] == "crash_task" and events[0]["plugin_type"] == "task"
        print(f"[crash_task] quarantined: {events[0]['reason']}")
    finally:
        unsub()
        orch.stop()
