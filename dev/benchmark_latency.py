"""Offline microbenchmarks for docs/PERFORMANCE_SURVEY.md.

Run from the repository root: python dev/benchmark_latency.py
Uses synthetic content and a temporary database; no providers or live config.
These component timings are not end-to-end UI measurements.
"""
import json
import math
import platform
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from state_machine.conversation import ConversationState  # settle package imports
from events.event_bus import EventBus
from llm.registry import Brain
from pipeline.database import Database
from runtime.conversation_loop import ConversationLoop
from runtime.token_stripper import ModelTextFilter
from runtime.subagents import SubagentRegistry
from sandbox.boxes import InProcessBox
from tests.support import FakeLLM, FakeRegistry, agent_state


def measure(fn, count=30):
    fn()
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {"median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(samples[math.ceil(len(samples) * .95) - 1], 3)}


def main():
    results = {"python": platform.python_version(), "platform": platform.platform()}
    with tempfile.TemporaryDirectory(prefix="sb-perf-") as scratch:
        db = Database(str(Path(scratch) / "bench.db"))
        try:
            cid = db.create_conversation()
            for size in (10, 1000, 10000):
                history = [{"role": "user" if i % 2 == 0 else "assistant",
                            "content": "x" * 1000} for i in range(size)]
                results[f"replace_{size}_rows"] = measure(
                    lambda: db.replace_conversation_messages(cid, history), 10)
                loop = ConversationLoop(FakeLLM(), FakeRegistry([]), {}, "system")
                results[f"messages_{size}_rows"] = measure(lambda: loop._messages(history))
            results["append_message"] = measure(lambda: db.save_message(cid, "user", "x" * 1000))
            def turn():
                loop = ConversationLoop(FakeLLM(), FakeRegistry([]), {}, "system")
                loop.drive(agent_state(), "agent", [{"role": "user", "content": "hi"}])
            results["fake_turn_no_db_frontend_hooks"] = measure(turn)
        finally:
            db.conn.close()

    box = InProcessBox(object(), "benchmark", manage_lifecycle=False)
    results["inprocess_invoke_noop"] = measure(lambda: box._invoke(lambda sdk: True, (), {}, 5), 100)
    for size in (10000, 100000, 1000000):
        def filter_region():
            stream = ModelTextFilter()
            stream.feed("<think>")
            for _ in range(size // 20):
                stream.feed("x" * 20)
            stream.feed("</think>answer")
            stream.flush()
        results[f"filter_region_{size}_chars"] = measure(filter_region, 3)

    bus = EventBus()
    bus.subscribe("delta", lambda _: time.sleep(.002))
    results["100_deltas_with_2ms_subscriber"] = measure(
        lambda: [bus.emit("delta", "x") for _ in range(100)], 3)

    brain = Brain.__new__(Brain)
    brain.config = {"max_concurrent_subagents": 1}
    brain._lock = threading.Lock()
    brain._boxes = [SimpleNamespace(alive=True), SimpleNamespace(alive=True)]
    brain._idle = []
    leased = brain._lease()
    brain._release(brain._boxes[1])
    results["pool_probe"] = {
        "overflow_selected_box_zero": leased is brain._boxes[0],
        "other_box_idle_but_existing_lease_unchanged": bool(brain._idle) and leased is brain._boxes[0]}

    registry = SubagentRegistry()
    child = SimpleNamespace(deadline=time.time() + 60, finished=False, collected=False)
    registry.pending_for = lambda owner: [child]
    registry._deliver = lambda session, handles: bool(handles)
    registry.forget = lambda owner: None
    session = SimpleNamespace(key="benchmark", lock=threading.Lock(),
                              cancel_event=threading.Event(), pending_user_inputs=[])
    def complete_child():
        child.finished = True
        registry.wake(session.key)
    timer = threading.Timer(.05, complete_child)
    start = time.perf_counter()
    timer.start()
    registry._barrier(session)
    timer.join()
    results["barrier_child_finishes_after_50ms"] = {"elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
