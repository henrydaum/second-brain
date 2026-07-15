---
name: pipeline-database-debugging
description: Diagnose and repair the Second Brain file/task pipeline through its SQLite database — find failure clusters, root-cause them, re-queue recoverable work, and spot misrouted files. Use for "why did files fail to parse/index", "the pipeline is stuck", "recover failed tasks", "which documents didn't get processed", or any question about task_queue / extracted_text / tabular_text state.
dependencies_files: [tools/tool_use_skill.py, tools/tool_sql_query.py, tools/tool_run_command.py]
---

# Pipeline & database debugging

The file pipeline is a DAG of path-driven tasks. A file is discovered into
`files`, and each task that matches its `modality` gets a row in `task_queue`.
Outputs land in per-task tables (`extracted_text`, `tabular_text`, `ocr_text`,
`audio_transcripts`, `text_chunks`, embeddings, `lexical_content`). Almost
every "why didn't X get processed / searchable" question is answered by SQL
over these tables. Diagnose from evidence, never from assumption.

Related skill: `sb-troubleshooting` covers boot/config/plugin-load/log
problems. This skill is the database specialist — load it when the symptom is
about *files not processing* or *pipeline output missing*, not app startup.

## The one query to start with

```sql
SELECT task_name, status, COUNT(*)
FROM task_queue GROUP BY task_name, status ORDER BY status, COUNT(*) DESC;
```

`status` is `DONE` or `FAILED` (uppercase; `PENDING` appears transiently
while work is queued). A pile of FAILED rows names the broken task. That is
always move one.

## Schema you will actually use

- `task_queue(path, task_name, status, priority, created_at, started_at,
  completed_at, error)` — the work list. `error` is the gold: it holds the
  failure string. One row per (file, task).
- `task_runs(run_id, task_name, status, triggered_by, parent_run_id,
  payload_json, created_at, started_at, finished_at, error)` — event/manual
  run history (dream_memory, subagents, retries). Path-driven task failures
  live in `task_queue.error`, not here.
- `registered_tasks(task_name, writes, reads, modalities, trigger,
  trigger_channels)` — the DAG definition. `writes`/`reads` are the output/
  input tables; `modalities` is which file types the task claims; `trigger`
  is `path` or `event`. Use it to see what a task *should* do.
- `files(path, file_name, extension, modality, mtime, discovered_at,
  updated_at, source)` — every indexed file. `modality` is the routing
  decision (text/image/audio/tabular/container) — the source of misroutes.
- Output tables: `extracted_text(path, ...)`, `tabular_text`, `ocr_text`,
  `audio_transcripts`, `text_chunks`, `lexical_content`. Presence of a row =
  that stage succeeded for that path.

## Root-causing a FAILED cluster

Never treat a failure count as one problem. Group by the error signature —
one task usually fails for several *different* reasons:

```sql
SELECT substr(error, 1, 70) AS error_head, COUNT(*) AS n
FROM task_queue
WHERE task_name = '<task>' AND status = 'FAILED'
GROUP BY error_head ORDER BY n DESC;
```

Then classify each cluster into one of three kinds:

1. **Missing dependency** — errors like ``Import <pkg> failed`` /
   ``<pkg> not installed`` / ``Install <pkg>``. Recoverable: install the
   package and re-queue. Confirm it actually resolves in the runtime
   interpreter before re-queuing:
   `run_command: python3 -c "import <pkg>; print('ok')"`.
2. **Genuinely unprocessable** — ``No extractable tables/images found``,
   ``Invalid data``, ``Not a valid TAR``, ``Failed to download``. These are
   correct failures. Do NOT re-queue; they will just fail again. Report and
   move on.
3. **Misroute** — ``No parser for (.docx, tabular)`` and similar: the file's
   `modality` sent it to a task whose parser has no handler for that
   extension. This is a routing/capability gap, not a dependency. Fix is a
   parser or classifier change, not a re-queue. (See pitfalls.)

## Recovering: re-queue only what will succeed

After fixing the cause (installed a dep, added a parser), flip ONLY the rows
matching that specific signature back to pending. This is a mutating
statement — it pauses for user approval, so write a clear justification and
one statement per call:

```sql
UPDATE task_queue SET status = 'PENDING', error = NULL
WHERE task_name = '<task>' AND status = 'FAILED'
  AND error LIKE '%<signature>%';
```

Scope it tightly. Re-queuing an entire task's failures drags the
unprocessable and misrouted rows back in to fail again and muddy the count.

## Is the output actually missing, or just the task row?

"Which files never got searchable text" is a LEFT JOIN, not a guess. A file
can fail one stage yet already have usable output from another (e.g. a .docx
that fails `textualize_tabular` but succeeded `extract_text`):

```sql
SELECT f.path
FROM files f
LEFT JOIN extracted_text et ON et.path = f.path
WHERE f.modality = 'text' AND et.path IS NULL;
```

Always check whether the data you care about exists via a *different* stage
before declaring it lost. This routinely turns a scary failure count into
"the prose is already indexed, only the tables are missing."

## Proactive health sweep (run it before the user notices)

The same queries make a good periodic check, not just a reactive one:

- Failure clusters by signature (query above) — surfaces new dependency gaps
  the moment they appear. A cluster of identical import errors is a
  one-package fix waiting to happen.
- Coverage gaps: files whose modality implies a stage that produced no row.
- Stuck work: rows `started_at IS NOT NULL AND completed_at IS NULL` far in
  the past suggest a crashed/timed-out task.

When you find a recoverable cluster, surface ONE actionable prompt
("install X to recover N files?") rather than a raw dump. Anything that
installs packages or writes to the DB stays behind user approval — review is
the safety boundary.

## After a parser/dependency fix: the reload catch

Parsers are `parse_*.py` helpers discovered when the **parser service loads**.
Editing/adding one (sandbox or installed) does NOT retroactively fix already-
FAILED rows, and an `EXTENSION`-lifecycle service (like `parser`) is not
user-reloadable via `/services`. The registry rebuilds on app restart or a
fresh `/packages install`. So the correct order is: fix parser → ensure it is
loaded (restart / reinstall) → *then* re-queue. Re-queuing before the parser
is live just reproduces the failure.

## Pitfalls (learned the hard way)

- **A failure count is never one bug.** 130 tabular failures were 5 distinct
  causes: 58 missing-openpyxl, 1 missing-xlrd, 51 docx-misroute, 20 tableless
  PDFs. Only 59 were dependency-recoverable. Always group by signature first.
- **"has_tables" detectors over-flag.** PyMuPDF's `find_tables()` marks ruled
  layouts and columns as tables, queueing essays and card art for tabular
  extraction that correctly finds nothing. Text-heavy files failing
  `textualize_tabular` with "No extractable tables" are usually false
  positives, not lost data — their text is already in `extracted_text`.
- **Don't re-queue the unrecoverable.** Flipping "No extractable tables" or
  corrupt-media rows to PENDING wastes a pipeline pass and re-inflates the
  failure count.
- **Confirm imports in `python3`, not just `pip show`.** A package can be
  installed but unimportable in the runtime's interpreter; verify before
  re-queuing.
- **Never read `action_ledger` or `task_runs` linearly** — high-volume
  tables. Always filter (task_name, status, a time window) and
  `ORDER BY ... DESC LIMIT`.
- **Default to read-only.** SELECT/PRAGMA run freely; only issue UPDATE/DELETE
  when the user asked to change state, one approved statement at a time.
