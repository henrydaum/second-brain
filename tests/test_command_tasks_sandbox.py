"""Structured task requests and sandboxed ``/tasks`` coverage."""

import json
from types import SimpleNamespace

from sandbox import Sandbox
from sandbox.handlers.kernel import _task_list
from sandbox.guest.requests import Request
from sandbox.policy import Chain, SAFE, UNSAFE, classify


class FakeDb:
    def __init__(self):
        self.reset = []
        self.runs = []

    def get_system_stats(self):
        return {"tasks": {"index": {"DONE": 3}}}

    def get_run_stats(self):
        return {"notify": {"FAILED": 1}}

    def reset_task(self, name):
        self.reset.append((name, False))

    def reset_failed_tasks(self, name):
        self.reset.append((name, True))

    def create_run(self, run_id, name, **kwargs):
        self.runs.append((run_id, name, kwargs))


class PathTask:
    trigger = "path"
    requires_services = ["embed"]
    trigger_channels = []
    event_payload_schema = {}
    config_settings = [
        ("Batch size", "task_batch_test", "Rows per batch.", 10,
         {"type": "slider"}),
        ("Secret", "task_secret_test", "Hidden.", "", {"hidden": True}),
    ]


class EventTask:
    trigger = "event"
    requires_services = []
    trigger_channels = ["mail.received"]
    event_payload_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["subject"],
    }
    config_settings = []


def _context():
    db = FakeDb()
    notified = []
    orchestrator = SimpleNamespace(
        tasks={"index": PathTask(), "notify": EventTask()},
        paused={"index"},
        clear_skip_cache=lambda name: notified.append(("clear", name)),
        dependency_pipeline_graph=lambda: "index -> notify",
        on_run_enqueued=lambda run_id, name:
            notified.append(("run", run_id, name)),
    )
    timekeeper = SimpleNamespace(
        loaded=True,
        list_jobs=lambda: {
            "mail": {"channel": "mail.received"},
            "other": {"channel": "other"},
        },
    )
    context = SimpleNamespace(
        orchestrator=orchestrator,
        db=db,
        services={"timekeeper": timekeeper},
        config={"task_batch_test": 25},
        runtime=None,
        user_id=1,
        session_key="chat",
    )
    return context, db, notified


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_tasks.py",
            "TasksCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_task_details_are_structured_and_hide_private_settings():
    context, _, _ = _context()

    result = _task_list(context, {"details": True})

    index, notify = result.data
    assert index["name"] == "index"
    assert index["counts"] == {"DONE": 3}
    assert index["paused"] is True
    assert [item["key"] for item in index["config_settings"]] == [
        "task_batch_test"]
    assert notify["schedule_count"] == 1


def test_tasks_forms_cover_path_event_payload_and_setting_quicklink():
    context, _, _ = _context()

    initial = _run(context, {}, method="form")
    path = _run(context, {"task_name": "index"}, method="form")
    event = _run(
        context,
        {"task_name": "notify", "action": "trigger"},
        method="form",
    )
    setting = _run(
        context,
        {
            "task_name": "index",
            "action": "edit_setting:task_batch_test",
        },
        method="form",
    )

    assert initial.data[0]["enum"] == ["index", "notify", "Show pipeline"]
    assert path.data[1]["enum"] == [
        "pause", "unpause", "reset", "retry",
        "edit_setting:task_batch_test",
    ]
    assert [step["name"] for step in event.data[-2:]] == [
        "subject", "count"]
    assert setting.data[-1]["type"] == "integer"


def test_tasks_listing_and_graph_match_established_output():
    context, _, _ = _context()

    listing = _run(context, {})
    graph = _run(context, {"task_name": "Show pipeline"})

    assert listing.data == (
        "Tasks:\n\n"
        "**Path-driven tasks**\n\n"
        "| Task | Pending | Running | Done | Failed | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| index | 0 | 0 | 3 | 0 | paused; needs: ['embed'] |\n\n"
        "**Event-driven tasks**\n\n"
        "| Task | Pending | Running | Done | Failed | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| notify | 0 | 0 | 0 | 1 | listens on: mail.received |"
    )
    assert graph.data == "index -> notify"


def test_task_mutations_and_manual_trigger():
    context, db, notified = _context()

    paused = _run(context, {"task_name": "notify", "action": "pause"})
    unpaused = _run(
        context,
        {"task_name": "index", "action": "unpause"},
        approve=lambda *_: True,
    )
    reset = _run(
        context,
        {"task_name": "index", "action": "reset"},
        approve=lambda *_: True,
    )
    retry = _run(
        context,
        {"task_name": "index", "action": "retry"},
        approve=lambda *_: True,
    )
    triggered = _run(
        context,
        {
            "task_name": "notify",
            "action": "trigger",
            "subject": "hello",
            "count": 2,
            "ignored": "no",
        },
    )

    assert paused.data == "Paused task: notify"
    assert unpaused.data == "Unpaused task: index"
    assert reset.data == "Reset task: index"
    assert retry.data == "Retried failed entries for task: index"
    assert triggered.data.startswith("Triggered task: notify (notify:")
    assert db.reset == [("index", False), ("index", True)]
    assert json.loads(db.runs[0][2]["payload_json"]) == {
        "subject": "hello", "count": 2}
    assert ("clear", "index") in notified
    assert any(item[0] == "run" for item in notified)


def test_task_policy_distinguishes_narrowing_from_widening():
    chain = Chain()

    pause = classify(
        Request("task.pause", {"name": "index", "paused": True}), chain)
    unpause = classify(
        Request("task.pause", {"name": "index", "paused": False}), chain)
    reset = classify(
        Request("task.reset", {"name": "index"}), chain)

    assert pause.level == SAFE
    assert unpause.level == UNSAFE
    assert reset.level == UNSAFE
