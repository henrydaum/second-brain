"""Slash command plugin for `/tasks`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


PIPELINE = "Show pipeline"
ALL_TASKS = "All tasks"

# What each trigger kind can do, minus the pause/unpause pair, which depends on
# the task's live state and is added by ``_actions_for``.
PATH_ACTIONS = [("reset", "Reset it"), ("retry", "Retry failures")]
EVENT_ACTIONS = [("trigger", "Trigger it"), ("schedule", "Schedule it")]

STATUSES = ("PENDING", "PROCESSING", "DONE", "FAILED")


def _actions_for(task):
    """Only the half of the pause toggle that does anything.

    The mechanism was already here — this branched on ``trigger`` to pick
    between two action lists — but nothing read ``paused``, so both halves of
    the one genuinely stateful pair were always offered. It also passed the
    raw action strings as labels, so the menu read "pause / unpause / reset /
    retry" in lowercase machine words.
    """
    paused = bool(task.get("paused"))
    rest = (EVENT_ACTIONS if task.get("trigger", "path") == "event"
            else PATH_ACTIONS)
    pairs = [("unpause", "Resume it") if paused else ("pause", "Pause it")]
    pairs += rest
    return [name for name, _ in pairs], [label for _, label in pairs]


class TasksCommand(BaseCommand):
    """Inspect and manage registered pipeline tasks."""

    name = "tasks"
    description = "Pick a task — pause, unpause, reset, retry, or trigger"
    category = "Automation"
    # Up front, before the body runs, rather than mid-run from the gate: this
    # command was the only bundled one declaring a mutating Request with no
    # gate at all, which is the shape that once deadlocked ``/packages``. A
    # literal tuple, because this is read by AST.
    approval_actions = ("reset", "retry", "unpause", "schedule",
                        "reset_all", "retry_all")
    approval_actor_id = "user"
    requests = [
        "task.list", "task.graph", "task.pause", "task.reset",
        "task.trigger", "cron.create", "service.call", "config.write",
    ]

    def form(self, sdk, args):
        """Build task, action, payload, and setting-value steps."""
        tasks = sdk.tasks.list(details=True)
        # The status table goes in the *prompt*, the way ``/services`` and
        # ``/schedule`` do it. This step is required, so ``run``'s no-name
        # branch — which renders exactly this — was unreachable from the menu.
        steps = [FormStep(
            "task_name",
            _show(sdk, tasks) + "\n\nSelect a task, or an action for all of "
            "them.",
            True,
            enum=[*[task["name"] for task in tasks], ALL_TASKS, PIPELINE],
            columns=2,
        )]
        if args.get("task_name") == PIPELINE:
            return steps
        if args.get("task_name") == ALL_TASKS:
            # The cost is on the options themselves. "Reset all tasks" is one
            # click away from re-running the whole pipeline over every indexed
            # file, so the number of rows it would queue belongs in the thing
            # being chosen, not in a message afterwards.
            steps.append(FormStep(
                "action", _bulk_prompt(sdk, tasks), True,
                enum=["retry_all", "reset_all"],
                enum_labels=_bulk_labels(tasks)))
            return steps
        task = _find(tasks, args.get("task_name"))
        if task:
            actions, action_labels = _actions_for(task)
            links, labels = sdk.forms.setting_actions(
                task.get("config_settings"))
            steps.append(FormStep(
                "action",
                "What do you want to do with this task?\n\n"
                + _describe(sdk, task),
                True,
                enum=actions + links,
                enum_labels=action_labels + labels,
            ))
        action = args.get("action")
        if task and action in ("trigger", "schedule"):
            if action == "schedule":
                steps.append(FormStep(
                    "job_name", "Name this scheduled job.", True))
                steps.append(FormStep(
                    "cron", "When should it run? (cron: m h dom mon dow)",
                    True))
            steps += sdk.forms.from_schema(
                task.get("event_payload_schema"),
                prompt_optional=True,
            )
        setting = sdk.forms.setting_for_action(
            (task or {}).get("config_settings"), action)
        if setting:
            steps.append(sdk.forms.setting_value_step(setting))
        return steps

    def run(self, sdk, args):
        """Execute the selected task-management action."""
        action = args.get("action")
        name = args.get("task_name")
        tasks = sdk.tasks.list(details=True)
        if not name:
            return _show(sdk, tasks)
        if name == ALL_TASKS:
            return _run_bulk(sdk, tasks, action)
        if name == PIPELINE:
            # An empty pipeline answers successfully with nothing, so the
            # Failed branch alone is not enough — a falsy result would be
            # dropped by the frontend and print as silence.
            try:
                graph = sdk.tasks.graph()
            except sdk.Failed:
                return "Pipeline unavailable."
            if not graph:
                return "No pipeline tasks are registered."
            # Fenced, because a rich renderer folds consecutive non-blank
            # lines into one paragraph — which turned the whole dependency
            # tree into a single unreadable line. The REPL is unaffected
            # either way: ``render_plain`` strips the fence markers back off.
            return f"```\n{graph}\n```"
        task = _find(tasks, name)
        if not task:
            return "Unknown task."

        setting = sdk.forms.setting_for_action(
            task.get("config_settings"), action)
        if setting:
            sdk.config.write(setting["key"], args.get("value"))
            return (
                f"Set {setting['key']} = "
                f"{sdk.text.value(args.get('value'))}"
            )
        if action == "pause":
            sdk.tasks.pause(name, True)
            return f"Paused task: {name}"
        if action == "unpause":
            sdk.tasks.pause(name, False)
            return f"Unpaused task: {name}"
        if action == "reset":
            if task.get("trigger", "path") == "event":
                return "Only path-driven tasks can be reset."
            sdk.tasks.reset(name)
            return f"Reset task: {name}"
        if action == "retry":
            if task.get("trigger", "path") == "event":
                return "Only path-driven tasks can be retried."
            sdk.tasks.reset(name, failed_only=True)
            return f"Retried failed entries for task: {name}"
        if action == "schedule":
            return _schedule(sdk, task, args)
        if action == "trigger":
            if task.get("trigger", "path") != "event":
                return "Only event-driven tasks can be triggered manually."
            properties = (
                task.get("event_payload_schema") or {}
            ).get("properties") or {}
            payload = {
                key: args[key] for key in properties if key in args
            }
            try:
                run_id = sdk.tasks.trigger(name, payload)
            except sdk.Failed as exc:
                if "task runs" in exc.error.lower():
                    return "No database is available for task runs."
                raise
            return f"Triggered task: {name} ({run_id})"
        return f"Unknown action: {action}"


def _find(tasks, name):
    return next(
        (task for task in tasks if task["name"] == name),
        None,
    )


def _counts(task):
    """A task's run counts, zero-filled. Absent rows mean none, not unknown."""
    counts = task.get("counts") or {}
    return {status: int(counts.get(status) or 0) for status in STATUSES}


def _show(sdk, tasks):
    """Every task's state in one table.

    One table rather than the SDK's sectioned rendering, because comparing
    them is the point: which is stuck, which is behind, which is paused. The
    trigger kind is a column instead of a heading for the same reason.
    """
    if not tasks:
        return "No tasks are registered."
    rows = []
    for task in tasks:
        counts = _counts(task)
        rows.append((
            task["name"],
            "Paused" if task.get("paused") else "Active",
            counts["PENDING"], counts["PROCESSING"],
            counts["DONE"], counts["FAILED"],
        ))
    return "Tasks:\n\n" + sdk.md.table(
        ["Task", "Status", "Pending", "Running", "Done", "Failed"],
        rows, leading_blank=False)


def _path_tasks(tasks):
    """The tasks a reset can reach. Event runs have no reset at all."""
    return [task for task in tasks
            if task.get("trigger", "path") != "event"]


def _bulk_totals(tasks):
    """(rows a reset would queue, rows a retry would queue)."""
    path = _path_tasks(tasks)
    counts = [_counts(task) for task in path]
    every = sum(sum(count.values()) for count in counts)
    failed = sum(count["FAILED"] for count in counts)
    return every, failed, len(path)


def _bulk_labels(tasks):
    """The cost, on the option itself."""
    every, failed, _ = _bulk_totals(tasks)
    return [f"Retry all failed tasks ({failed:,} rows)",
            f"Reset all tasks ({every:,} rows)"]


def _bulk_prompt(sdk, tasks):
    every, failed, count = _bulk_totals(tasks)
    return (
        f"{_show(sdk, tasks)}\n\n"
        f"Across {count} path-driven task(s): {failed:,} failed row(s), "
        f"{every:,} row(s) in total. Resetting queues all of them again."
    )


def _run_bulk(sdk, tasks, action):
    """Retry or reset every path-driven task."""
    path = _path_tasks(tasks)
    every, failed, _ = _bulk_totals(tasks)
    if action == "retry_all":
        if not failed:
            return "Nothing has failed."
        for task in path:
            sdk.tasks.reset(task["name"], failed_only=True)
        return f"Retried {failed:,} failed row(s) across {len(path)} task(s)."
    if action == "reset_all":
        if not every:
            return "There is nothing to reset."
        for task in path:
            sdk.tasks.reset(task["name"])
        return f"Reset {every:,} row(s) across {len(path)} task(s)."
    return f"Unknown action: {action}"


def _schedule(sdk, task, args):
    """Create a Timekeeper job that fires this task's trigger channel.

    The same act ``/schedule`` performs, reached from the task rather than
    from the channel: a job whose channel is one of the task's
    ``trigger_channels`` and whose payload matches its declared schema. The
    kernel side is already closed — the timekeeper emits, ``EventTrigger``
    turns the emit into a run.
    """
    if task.get("trigger", "path") != "event":
        return "Only event-driven tasks can be scheduled."
    channels = [c for c in (task.get("trigger_channels") or []) if c]
    if not channels:
        return f"{task['name']} declares no trigger channel to fire."
    job = (args.get("job_name") or "").strip()
    cron = (args.get("cron") or "").strip()
    if not job or not cron:
        return "A scheduled job needs a name and a cron expression."
    properties = (task.get("event_payload_schema") or {}).get(
        "properties") or {}
    payload = {key: args[key] for key in properties if key in args}
    try:
        sdk.cron.create(job, {"cron": cron, "channel": channels[0],
                              "payload": payload, "enabled": True})
    except sdk.Failed as exc:
        return f"Could not schedule {task['name']}: {exc.error}"
    return f"Scheduled {task['name']}: {job}, {_when(sdk, cron)}."


def _when(sdk, cron):
    """A cron in English, falling back to the expression itself.

    Through the timekeeper because a command is guest code and cannot import
    the kernel's copy; guarded because the service may not be loaded.
    """
    try:
        return str(sdk.services.call("timekeeper", "cron_to_text", cron)).lower()
    except sdk.Failed:
        return f"`{cron}`"


def _describe(sdk, task):
    counts = _counts(task)
    pairs = [
        # First, because it is the one thing on this card that changes what
        # the task *does* — and the card did not show it at all, so the only
        # way to learn a task was paused was to notice nothing happening.
        ("Status", "Paused" if task.get("paused") else "Active"),
        ("Trigger", task.get("trigger", "path")),
        ("Pending", counts["PENDING"]),
        ("Running", counts["PROCESSING"]),
        ("Done", counts["DONE"]),
        ("Failed", counts["FAILED"]),
    ]
    pairs += [
        (setting["title"], sdk.text.value(setting.get("current")))
        for setting in task.get("config_settings") or []
    ]
    card = sdk.md.card(task["name"], pairs)
    description = (task.get("description") or "").strip()
    if description:
        card += f"\n\n{sdk.md.quote(description)}"
    if (
        task.get("trigger", "path") == "event"
        and task.get("schedule_count")
    ):
        card += (
            f"\n\nScheduled jobs: {task['schedule_count']}. "
            "Use /schedule to manage them."
        )
    return card
