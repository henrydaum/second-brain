"""Slash command plugin for `/schedule`.

Kernel rather than store. It manages *any* Timekeeper job, the Timekeeper is a
kernel service, and subagent spawning is now a kernel capability — so a store
command was the only way to reach two things the kernel already owns.

Two kinds of thing can be scheduled, and the difference matters when reading
this file. A **background agent** is the kernel's own spawn channel: pick it
and you are asked for a prompt. An **event-driven task** is whatever the
pipeline has registered, scheduled by firing its trigger channel with a
payload shaped by its declared schema.
"""

from guest.bases import BaseCommand
from guest.forms import FormStep


ADD = "add"
NONE = "(none)"
SUBAGENT = "background agent"
# The kernel's spawn channel. Named here as a literal because a command cannot
# import kernel modules; it is pinned against events/event_channels.py by a
# test rather than by an import.
SUBAGENT_CHANNEL = "subagent.spawn"

ACTIONS = ["edit", "delete", "enable", "disable"]
ACTION_LABELS = ["Edit it", "Delete it", "Enable it", "Disable it"]

# What a background-agent job carries. The kernel's spawn subscriber reads
# these keys; the shape mirrors what an event task declares for itself.
SUBAGENT_SCHEMA = {
    "properties": {
        "title": {"type": "string",
                  "description": "Short title for the agent's conversation."},
        "prompt": {"type": "string",
                   "description": "Complete, self-contained instructions. "
                                  "Nobody will answer a follow-up question."},
    },
}


class ScheduleCommand(BaseCommand):
    """Slash-command handler for `/schedule`."""

    name = "schedule"
    description = "Manage scheduled jobs: background agents and pipeline tasks"
    category = "Tasks"
    # Creating, editing and deleting a schedule all commit the system to
    # unattended future work, which is what ALWAYS_UNSAFE singles out. The
    # up-front gate states the whole scope before the body runs; without it
    # the cron Requests fall through mid-form to the execution-time approver,
    # which is the path a command cannot be asked from. Enabling and disabling
    # are not listed: cron.enable is safe either way, and disabling narrows.
    #
    # A literal tuple on purpose — this is read by AST, so `tuple(ACTIONS)`
    # would read as nothing at all.
    approval_actions = ("add", "edit", "delete")
    approval_actor_id = "user"
    requests = ["cron.list", "cron.get", "cron.create", "cron.update",
                "cron.remove", "cron.enable", "task.list", "service.call"]

    def form(self, sdk, args):
        """Build dependent steps from the answers collected so far."""
        jobs = sdk.cron.list() or {}
        names = sorted(jobs)
        steps = [FormStep(
            "job_name", _list_prompt(sdk, jobs), True,
            enum=[*names, ADD],
            enum_labels=[*_job_labels(jobs, names), "Schedule new job"],
            columns=1)]

        if args.get("job_name") == ADD:
            targets = _targets(sdk)
            steps += [
                FormStep("target", "What should run on this schedule?", True,
                         enum=sorted(targets) or [NONE], columns=1),
                FormStep("new_job_name",
                         "Enter a unique name for this schedule.", True),
                FormStep("cron",
                         "Enter the cron expression, for example 0 9 * * *.",
                         True),
            ]
            schema = (targets.get(args.get("target")) or {}).get("schema")
            if schema:
                steps += _schema_steps(schema, {})
            return steps

        job = jobs.get(args.get("job_name"))
        if job:
            steps.append(FormStep(
                "action",
                "What do you want to do with this scheduled job?\n\n"
                + _describe(sdk, args["job_name"], job),
                True, enum=ACTIONS, enum_labels=ACTION_LABELS, columns=2))
        if job and args.get("action") == "edit":
            one_time = bool(job.get("one_time"))
            steps.append(FormStep(
                "run_at" if one_time else "cron",
                "Enter the new run time." if one_time
                else "Enter the new cron expression.",
                True,
                default=job.get("run_at") if one_time else job.get("cron")))
            steps += _schema_steps(_schema_for(sdk, job),
                                   job.get("payload") or {})
        return steps

    def run(self, sdk, args):
        """Execute `/schedule` for the active session."""
        name = args.get("job_name")
        if not name:
            return _format_jobs(sdk, sdk.cron.list() or {})
        if name == ADD:
            return _create(sdk, args)

        job = sdk.cron.get(name)
        if job is None:
            return f"No such job: {name}"
        action = args.get("action")
        if action == "delete":
            return (f"Deleted job: {name}" if sdk.cron.remove(name)
                    else f"No such job: {name}")
        if action in ("enable", "disable"):
            sdk.cron.enable(name, action == "enable")
            return f"{action.title()}d job: {name}"
        if action == "edit":
            return _edit(sdk, name, job, args)
        return f"Unknown action: {action}"


# ──────────────────────────────────────────────────────────────────────
# What can be scheduled.
# ──────────────────────────────────────────────────────────────────────

def _targets(sdk) -> dict:
    """Everything schedulable, by display name.

    The background agent is always offered — it is the kernel's own, and it is
    what most schedules are for. Event-driven tasks join it when the pipeline
    has any registered, which on a bare kernel it does not.
    """
    targets = {SUBAGENT: {"channel": SUBAGENT_CHANNEL,
                          "schema": SUBAGENT_SCHEMA}}
    for task in sdk.tasks.list(details=True) or []:
        channels = [c for c in (task.get("trigger_channels") or []) if c]
        if task.get("trigger") != "event" or not channels:
            continue
        targets[task.get("name") or "?"] = {
            "channel": channels[0],
            "schema": task.get("event_payload_schema") or {},
        }
    return targets


def _schema_for(sdk, job) -> dict:
    """The payload schema of whatever this job fires."""
    channel = (job or {}).get("channel")
    if channel == SUBAGENT_CHANNEL:
        return SUBAGENT_SCHEMA
    for target in _targets(sdk).values():
        if target["channel"] == channel:
            return target["schema"]
    return {}


def _target_name(sdk, job) -> str:
    """What this job runs, for display."""
    channel = (job or {}).get("channel")
    for name, target in _targets(sdk).items():
        if target["channel"] == channel:
            return name
    return "-"


# ──────────────────────────────────────────────────────────────────────
# Forms.
# ──────────────────────────────────────────────────────────────────────

def _schema_steps(schema: dict, payload: dict):
    """One optional step per declared payload key.

    A target that declares nothing gets a single JSON step — better than
    refusing to schedule it, and the same fallback the store version used.
    """
    props = (schema or {}).get("properties") or {}
    if not props:
        return [FormStep("payload", "Enter the payload as a JSON object.",
                         False, "object", default=payload,
                         prompt_when_missing=True)]
    return [
        FormStep(key, _prompt_for(key, info), False,
                 info.get("type", "string"), info.get("enum"),
                 default=payload.get(key, info.get("default")),
                 prompt_when_missing=True)
        for key, info in props.items()
    ]


def _prompt_for(key: str, info: dict) -> str:
    """A form prompt from one schema property."""
    described = (info.get("description") or "").strip()
    return described or f"Enter {key.replace('_', ' ')}."


def _payload_from(schema: dict, args: dict, current: dict | None = None) -> dict:
    """Collect the declared keys the form actually answered."""
    if "payload" in args:
        return args["payload"] or {}
    out = dict(current or {})
    for key in ((schema or {}).get("properties") or {}):
        if key in args:
            out[key] = args[key]
    return out


# ──────────────────────────────────────────────────────────────────────
# Doing it.
# ──────────────────────────────────────────────────────────────────────

def _create(sdk, args) -> str:
    """Create one schedule from the answered form."""
    target = _targets(sdk).get(args.get("target"))
    if target is None:
        return "Choose something to schedule."
    name = (args.get("new_job_name") or "").strip()
    cron = (args.get("cron") or "").strip()
    if not name or not cron:
        return "Enter both a schedule name and a cron expression."
    payload = _payload_from(target["schema"], args)
    try:
        sdk.cron.create(name, {"cron": cron, "channel": target["channel"],
                               "payload": payload, "enabled": True})
    except Exception as exc:
        return f"Failed to create job: {exc}"
    job = sdk.cron.get(name) or {"cron": cron}
    return (f"Created schedule '{name}' for {args.get('target')}: "
            f"{_schedule_text(sdk, job)}.")


def _edit(sdk, name: str, job: dict, args) -> str:
    """Apply the edit form to one existing job."""
    one_time = bool(job.get("one_time"))
    when = (args.get("run_at") if one_time else args.get("cron")) or (
        job.get("run_at") if one_time else job.get("cron"))
    patch = {
        "run_at" if one_time else "cron": when,
        "payload": _payload_from(_schema_for(sdk, job), args,
                                 job.get("payload") or {}),
    }
    try:
        sdk.cron.update(name, patch)
    except Exception as exc:
        return f"Failed to update job: {exc}"
    return f"Updated job: {name}"


# ──────────────────────────────────────────────────────────────────────
# Rendering. Markdown on the wire; each frontend renders by policy.
# ──────────────────────────────────────────────────────────────────────

def _list_prompt(sdk, jobs: dict) -> str:
    """The first form step's prompt: the whole schedule, then the question."""
    return (f"{_format_jobs(sdk, jobs)}\n\n"
            "Select a scheduled job, or add a new one.")


def _format_jobs(sdk, jobs: dict) -> str:
    """Every schedule as a table."""
    if not jobs:
        return "No scheduled jobs."
    rows = []
    for name, job in sorted(jobs.items()):
        payload = job.get("payload") or {}
        rows.append((name,
                     "enabled" if job.get("enabled", True) else "disabled",
                     _schedule_text(sdk, job),
                     (payload.get("title") or _target_name(sdk, job)).strip()))
    return "Scheduled jobs:\n\n" + _md_table(
        ["Job", "Status", "Schedule", "Runs"], rows)


def _job_labels(jobs: dict, names: list) -> list:
    """Enum labels that say which jobs are switched off."""
    return [name if (jobs.get(name) or {}).get("enabled", True)
            else f"{name} (disabled)" for name in names]


def _describe(sdk, name: str, job: dict) -> str:
    """One job as a detail card, with its payload quoted underneath."""
    payload = dict(job.get("payload") or {})
    rows = [
        ("Status", "Enabled" if job.get("enabled", True) else "Disabled"),
        ("Runs", _target_name(sdk, job)),
        ("Channel", job.get("channel") or "-"),
        ("Schedule", _schedule_text(sdk, job)),
        ("Next", _next_fire(sdk, name) or "disabled"),
    ]
    if payload.get("title"):
        rows.append(("Title", str(payload.pop("title"))))
    # The prompt reads as prose, so it is quoted; whatever else the payload
    # carries follows it as one compact line.
    prompt = _truncate(str(payload.pop("prompt", "") or ""), 500)
    rest = ", ".join(f"{k}={v}" for k, v in sorted(payload.items()))
    quoted = "\n".join(f"> {line}" for line in
                       [p for p in (prompt, rest) if p])
    card = _md_table([name, ""], rows)
    return card + (f"\n\n**Payload**\n{quoted}" if quoted else "")


def _schedule_text(sdk, job: dict) -> str:
    """When this job runs, in words where the Timekeeper can manage it."""
    if job.get("one_time"):
        return f"once at {job.get('run_at') or '?'}"
    cron = job.get("cron") or ""
    try:
        return sdk.services.call("timekeeper", "cron_to_text", cron).lower()
    except Exception:
        return cron or "?"


def _next_fire(sdk, name: str):
    """The next fire time, or None when the job is switched off."""
    try:
        return sdk.services.call("timekeeper", "get_next_fire_at", name)
    except Exception:
        return None


def _md_table(headers, rows) -> str:
    """A GitHub-flavored table. Frontends render it by policy, not by sender."""
    line = "| " + " | ".join(str(h) for h in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def _truncate(text: str, limit: int) -> str:
    """Shorten prose without hiding that it was shortened."""
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."
