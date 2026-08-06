"""
TASK TEMPLATE
=============
A task is background pipeline work: it runs over files as they appear, or when
an event fires. Reference for authoring one; not imported by the running
system.

Read docs/SDK.md for the Request surface and sandbox/guest/bases.py for every
attribute a task can declare. This file covers what is specific to tasks.

Before writing: read docs/SDK.md, then this entire template. For details not
defined here, inspect sandbox/guest/bases.py (BaseTask declarations),
pipeline/orchestrator.py (dependency and run behavior), pipeline/event_trigger.py
(event tasks), and pipeline/database.py (run and output storage). Validate the
finished file before registering it.

  Where it goes:  DATA_DIR/workspace/tasks/task_<name>.py
  Filename:       must start with "task_"
  Entry point:    run(self, sdk, paths)

WATCH THE ARGUMENT ORDER: run(self, sdk, paths), sdk first. Getting this
backwards binds your paths to the sdk and fails in a confusing way.


TRIGGER KINDS
-------------
Every task picks exactly one:

  trigger = "path"    (default) keyed by file path. Root tasks (reads = [])
                      fire when a file is discovered; downstream tasks fire
                      when their upstream finishes.

  trigger = "event"   keyed by run_id. Fires when a declared bus channel
                      emits. Use for cron-like work, tool-triggered work, or
                      anything not per-file. Also set
                      trigger_channels = ["channel.name"].

The trigger decides your entry point, and getting it wrong fails silently:
"path" tasks implement run(self, sdk, paths) and "event" tasks implement
run_event(self, sdk, payload). A task with the wrong one registers, subscribes,
and is never called.

trigger_channels must be a literal list. trigger_channels = [SOME_CONSTANT]
reads as [] — it validates, registers, and subscribes to nothing.


THE DEPENDENCY GRAPH (the part you cannot guess)
------------------------------------------------
Tasks never reference each other by name. The orchestrator derives the graph
from `reads` and `writes` — but ONLY between tasks of the same trigger kind:

  TaskA (path) writes "text_chunks", TaskB (path) reads it
    -> TaskB is downstream of TaskA and fires when it completes.

  TaskA (event) writes "daily_summary", TaskB (event) reads it
    -> TaskB auto-fires after TaskA, with parent_run_id set.

Cross-kind reads are AMBIENT SQL JOINS, not graph edges. An event task that
reads a path-keyed table just SELECTs it at run time; changes to path data do
NOT invalidate event runs. If a path task needs to kick off an event run, it
emits on the channel explicitly with sdk.events.emit(...).

This is why adding a `reads` entry can silently change when your task runs.


ROWS ARE THE INTERFACE
----------------------
What you return is written to your output table and becomes the input of every
downstream task. Prefer explicit, stable column names — downstream tasks, SQL
inspection, and debugging all read them. Renaming a column is a breaking
change to a contract you cannot see from inside your own file.

Two fields exist for the pipeline rather than for you:

  also_contains     modalities discovered inside the file ("image" in a PDF),
                    which routes it to further parsers.
  discovered_paths  new files to register — how an archive extractor feeds
                    its contents back into the pipeline.


SHIPPING A SCHEDULE
-------------------
Create the timekeeper job from `on_install`, and remove it from
`on_uninstall`. Both run inside `/packages`, once, for the package the user
just asked for.

    job = {"channel": "schedule.tick.nightly", "cron": "0 3 * * *",
           "payload": {"scope": "all"}}

    def on_install(self, sdk):
        if sdk.services.call("timekeeper", "get_job", self.name) is None:
            sdk.services.call("timekeeper", "create_job", self.name, self.job)

    def on_uninstall(self, sdk):
        sdk.services.call("timekeeper", "remove_job", self.name)

Read-then-skip, so a job whose cron the user has edited is left alone. This
was a `default_jobs` declaration seeded at every registration, which meant a
job the user deleted came back at the next boot; the declaration is now
ignored and the validator says so. Declare `service.call`.


The two examples below are separate tasks, shown together for contrast. A real
file declares exactly ONE plugin class.
"""

from guest.bases import BaseTask


class WordStats(BaseTask):
    """A root path task: fires on file discovery, writes rows others can read."""

    name = "word_stats"
    description = "Count words and lines in every text file discovered."
    # Guidance added to the agent's system prompt while this task is
    # registered. A method (``def agent_prompt(self, sdk)``) works too.
    agent_prompt = "## Word stats\nCounts land in the word_stats table."

    trigger = "path"
    modalities = ["text"]
    reads = []                  # no upstream, so this is a root task
    writes = ["word_stats"]     # downstream tasks read this table by name
    batch_size = 8

    def run(self, sdk, paths):
        """Process a batch of paths, returning one row per file."""
        rows = []
        for path in paths:
            # Parsing goes through the registry, so an installed parser
            # package lights up here without this task changing.
            text = sdk.parse.file(path, modality="text")
            rows.append({
                "path": path,
                "words": len(text.split()),
                "lines": len(text.splitlines()),
            })
        return rows


class DailyDigest(BaseTask):
    """An event task on a schedule: no paths, driven by the clock."""

    name = "daily_digest"
    description = "Summarize yesterday's activity once a night."

    trigger = "event"
    trigger_channels = ["schedule.tick.nightly"]
    reads = []
    writes = ["daily_digest"]

    job = {"channel": "schedule.tick.nightly", "cron": "0 3 * * *",
           "payload": {}}

    def on_install(self, sdk):
        """Create the schedule, once, when this package is installed."""
        if sdk.services.call("timekeeper", "get_job", self.name) is None:
            sdk.services.call("timekeeper", "create_job", self.name, self.job)

    def on_uninstall(self, sdk):
        """A job whose task is gone fires into nothing forever."""
        sdk.services.call("timekeeper", "remove_job", self.name)

    def run_event(self, sdk, payload):
        """Summarize the day.

        ``run_event``, not ``run`` — the two entry points are different jobs
        rather than one job with an optional argument, and a ``trigger =
        "event"`` task that implements ``run`` is never called at all. Nothing
        reports that: the task registers, subscribes, and silently does
        nothing every time it fires.

        ``payload`` is whatever was emitted on the channel, verbatim. A
        scheduled job supplies its declared ``payload``; a plugin emitting the
        channel itself supplies its own.
        """
        # Cross-kind read: word_stats is a PATH table, so this is an ambient
        # SQL join rather than a graph edge. Nothing re-runs this when
        # word_stats changes; the schedule is what drives it.
        rows = sdk.db.query(
            "SELECT path, words FROM word_stats ORDER BY words DESC LIMIT 5")
        if not rows:
            return []

        listing = "\n".join(f"- {r['path']}: {r['words']} words" for r in rows)
        summary = sdk.agent.complete(
            f"Write two sentences summarizing today's largest documents:\n{listing}")
        return [{"summary": summary, "documents": len(rows)}]
