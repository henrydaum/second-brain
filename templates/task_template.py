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
Declare `default_jobs` and the orchestrator seeds the timekeeper job when the
task registers, and removes it when the task unregisters. A reinstall picks up
an updated declaration. To silence one durably, disable it — do not delete it,
or the next registration seeds it again.

    default_jobs = {
        "nightly_summary": {
            "channel": "schedule.tick.nightly",
            "cron": "0 3 * * *",
            "payload": {"scope": "all"},
        },
    }


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

    default_jobs = {
        "nightly_digest": {
            "channel": "schedule.tick.nightly",
            "cron": "0 3 * * *",
            "payload": {},
        },
    }

    def run(self, sdk, paths):
        """Summarize the day. Event tasks get no paths — ignore the argument."""
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
