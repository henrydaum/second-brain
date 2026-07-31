"""Slash command plugin for `/tasks`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


PIPELINE = "Show pipeline"

# What each trigger kind can do, minus the pause/unpause pair, which depends on
# the task's live state and is added by ``_actions_for``.
PATH_ACTIONS = [("reset", "Reset it"), ("retry", "Retry failures")]
EVENT_ACTIONS = [("trigger", "Trigger it")]


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


def _task_label(task):
    """A name with its state in front of it, as ``/services`` does."""
    return f"{'⏸' if task.get('paused') else '●'} {task['name']}"


class TasksCommand(BaseCommand):
    """Inspect and manage registered pipeline tasks."""

    name = "tasks"
    description = "Pick a task — pause, unpause, reset, retry, or trigger"
    category = "Automation"
    requests = [
        "task.list", "task.graph", "task.pause", "task.reset",
        "task.trigger", "config.write",
    ]

    def form(self, sdk, args):
        """Build task, action, payload, and setting-value steps."""
        tasks = sdk.tasks.list(details=True)
        steps = [FormStep(
            "task_name",
            "Select a task to manage, or view the pipeline.",
            True,
            enum=[*[task["name"] for task in tasks], PIPELINE],
            enum_labels=[*[_task_label(task) for task in tasks], PIPELINE],
            columns=2,
        )]
        if args.get("task_name") == PIPELINE:
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
        if task and action == "trigger":
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
            return sdk.md.tasks(tasks)
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


def _describe(sdk, task):
    counts = {
        "PENDING": 0,
        "PROCESSING": 0,
        "DONE": 0,
        "FAILED": 0,
        **(task.get("counts") or {}),
    }
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
    hint = ""
    if (
        task.get("trigger", "path") == "event"
        and task.get("schedule_count")
    ):
        hint = (
            f"Scheduled jobs: {task['schedule_count']}. "
            "Use /schedule to manage them."
        )
    return card + (f"\n\n{hint}" if hint else "")
