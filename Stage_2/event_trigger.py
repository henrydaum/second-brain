"""
EventTrigger.

The bus-side analog of the file Watcher. Where the watcher notices files
and asks the orchestrator to enqueue path-keyed work, EventTrigger
subscribes to declared bus channels and enqueues run-id-keyed work in
the task_runs table.

An event task declares:
    trigger          = "event"
    trigger_channels = ["some.channel", ...]

When any of those channels fires, EventTrigger creates a PENDING row in
task_runs and notifies the orchestrator, which picks it up on its next
dispatch tick.

Manual/tool-driven runs are just bus.emit(channel, payload) — no
separate code path needed.
"""

import json
import logging
from uuid import uuid4

from event_bus import bus

logger = logging.getLogger("EventTrigger")


class EventTrigger:
    def __init__(self, orchestrator, db, config: dict):
        self.orchestrator = orchestrator
        self.db = db
        self.config = config
        self._unsubs: list = []
        self._started = False
        self._subscriptions: dict[tuple[str, str], callable] = {}
        self.orchestrator.event_trigger = self

    def start(self):
        self._started = True
        self.refresh()

    def refresh(self):
        if not self._started:
            return

        self.stop(clear_started=False)

        for task in self.orchestrator.tasks.values():
            if getattr(task, "trigger", "path") != "event":
                continue
            channels = getattr(task, "trigger_channels", []) or []
            if not channels:
                logger.warning(
                    f"Event task '{task.name}' has no trigger_channels — it will never fire."
                )
                continue
            for channel in channels:
                unsub = bus.subscribe(channel, self._make_handler(task, channel))
                self._unsubs.append(unsub)
                self._subscriptions[(task.name, channel)] = unsub
                logger.info(f"'{task.name}' subscribed to '{channel}'")

    def _make_handler(self, task, channel):
        def handler(payload):
            run_id = f"{task.name}:{uuid4().hex[:12]}"
            try:
                self.db.create_run(
                    run_id,
                    task.name,
                    triggered_by=channel,
                    payload_json=json.dumps(payload or {}, default=str),
                )
            except Exception as e:
                logger.error(f"Failed to enqueue run for '{task.name}' on '{channel}': {e}")
                return
            self.orchestrator.on_run_enqueued(run_id, task.name)
        return handler

    def stop(self, clear_started: bool = True):
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs.clear()
        self._subscriptions.clear()
        if clear_started:
            self._started = False
        logger.info("EventTrigger stopped.")
