"""
TASK TEMPLATE
=============
This file is a self-contained reference for creating new tasks.
It is NOT imported by the running system — it exists for LLM consumption only.

Write tasks in a practical, explicit style. A task should make its trigger,
inputs, outputs, and failure modes easy to inspect from code and database rows.

Task authoring flow:
  1. Read this template, then read one similar built-in task for style.
  2. Create sandbox_tasks/task_<your_name>.py with edit_file.
  3. The code MUST inherit from BaseTask and include:
       from plugins.BaseTask import BaseTask, TaskResult
  4. Fill in the class attributes and implement run() or run_event().
  5. Call test_plugin(plugin_path="sandbox_tasks/task_<your_name>.py").
  6. If testing fails, read the error, edit the same file, and retry.
  7. Valid plugins are discovered on startup; plugin_watcher live-loads adds/edits when enabled.
  8. To update: edit the file; plugin_watcher reloads it when enabled.
  9. To remove live and durably: delete the sandbox file; plugin_watcher unloads it when enabled.
 10. If the task needs extra packages, install them first with
     run_command(command="pip install <pkg>", justification="...", timeout=300).

test_plugin validates:
  - Correct import (from plugins.BaseTask import BaseTask, TaskResult)
  - Class inheriting BaseTask with a `name` attribute
  - No name collisions with baked-in tasks
  - File naming conventions (must start with "task_")
  - The pytest suite summary


AUTO-DISCOVERY RULES
--------------------
- File must be in plugins/tasks/ (baked-in) or the sandbox tasks dir
- File name must start with "task_"
- Class must inherit from BaseTask
- One task class per file (recommended)


TRIGGER KINDS
-------------
Every task picks ONE trigger kind:

  trigger = "path"   (default) — keyed by file path. Root tasks (reads=[])
                     fire on file discovery; downstream tasks fire when
                     upstream completes. Implements run(paths, context)
                     returning a list[TaskResult] (one per path).

  trigger = "event"  — keyed by run_id. Fires whenever a declared bus
                     channel emits. Use this for cron-like jobs,
                     tool-triggered work, or anything that isn't per-file.
                     Must also set trigger_channels = ["chan1", ...].
                     Implements run_event(run_id, payload, context) and
                     returns a single TaskResult. Rows in result.data
                     must include the run_id.


DEPENDENCY GRAPH
----------------
Tasks never reference each other by name. Dependencies are derived
automatically from reads/writes — but ONLY between tasks of the same
trigger kind:

  If TaskA (path) writes "text_chunks" and TaskB (path) reads it,
  TaskB is downstream of TaskA in the path graph.

  If TaskA (event) writes "daily_summary" and TaskB (event) reads it,
  TaskB auto-fires after TaskA completes, with parent_run_id set.

Cross-kind reads are AMBIENT SQL JOINS, not graph edges. An event task
that reads a path-keyed table just SELECTs it at run time. Path-data
changes do NOT auto-invalidate event runs. If a path task wants to
kick off an event run, it can bus.emit(...) explicitly.


CONTEXT OBJECT
--------------
Every task receives a `context` object with:

  context.db        Database instance (SQLite). Key methods:
                      .get_task_output(table, path) -> list[dict]
                      .conn  (raw connection, use with context.db.lock)
                      .lock  (threading.Lock for direct SQL)

  context.config    Global settings dict (from config.json).

  context.services  Dict of {name: service_instance}. Access shared models:
                      embedder = context.services.get("text_embedder")
                      if embedder and embedder.loaded:
                          embedder.encode(texts)

  context.parse     Parse a file using the Stage 1 parser system:
                      result = context.services.get("parser").parse(path)           # default modality
                      result = context.services.get("parser").parse(path, "image")  # specific modality
                    Returns ParseResult with .success, .output, .also_contains


TASK RESULT
-----------
Return one TaskResult per input path:

  TaskResult(
      success=True,
      data=[{"col": "value", ...}],  # rows written to your output table
      also_contains=["image"],        # modalities discovered (from parser)
      discovered_paths=["/new/file"], # new files to register (e.g. extracted from archive)
  )

  TaskResult.failed("error message")  # shorthand for failures

Keep TaskResult.data explicit and inspection-friendly. Prefer rows with clear,
stable column names so downstream tasks, SQL inspection, and debugging stay easy.


CONFIG SETTINGS
---------------
Tasks can declare config settings that appear in the Settings UI and are
stored in plugin_config.json. Values are accessible via context.config.get().

  config_settings = [
      ("Chunk Overlap", "embed_chunk_overlap",
       "Overlapping tokens between chunks.",
       50,
       {"type": "slider", "range": (0, 200, 40), "is_float": False}),
  ]

Each entry is a tuple: (title, variable_name, description, default, type_info)
See tool_template.py for full type_info options.

Multiple plugins can declare the same variable_name — the value is shared.
In run(), access via: context.config.get("embed_chunk_overlap", 50)
"""

# =====================================================================
# BASE CLASS (copied from plugins/BaseTask.py for self-containment)
# =====================================================================

import logging
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    success: bool = True
    error: str = ""
    data: list[dict] = field(default_factory=list)
    also_contains: list[str] = field(default_factory=list)
    discovered_paths: list[str] = field(default_factory=list)

    @staticmethod
    def failed(error: str) -> "TaskResult":
        return TaskResult(success=False, error=error)


class BaseTask:
    # --- Identity ---
    name: str = ""

    # --- Trigger kind ---
    trigger: str = "path"               # "path" | "event"
    trigger_channels: list[str] = []    # bus channels (event tasks only)

    # --- Routing ---
    modalities: list[str] = []          # file types (root path tasks only). e.g. ["text", "image"]

    # --- Data flow ---
    reads: list[str] = []               # input tables (dependencies derived automatically)
    writes: list[str] = []              # output tables
    require_all_inputs: bool = True     # True=AND (all inputs needed), False=OR (any suffices)

    # --- Service requirements ---
    requires_services: list[str] = []   # services that must be loaded before dispatch

    # --- Schema ---
    output_schema: str = ""             # raw SQL to create output table(s)

    # --- Execution ---
    batch_size: int = 1                 # files per run() call
    max_workers: int = 0                # 0 = use global setting
    timeout: int = 300                  # seconds before considered stuck

    # --- Config settings ---
    config_settings: list = []          # settings shown in the Settings UI

    def setup(self, config: dict):
        """Called once at registration. Optional."""
        pass

    def teardown(self):
        """Called on shutdown. Optional."""
        pass

    def run(self, paths: list[str], context) -> list[TaskResult]:
        """Path-keyed entry point. Override for trigger='path' (default)."""
        return [TaskResult.failed("Not implemented") for _ in paths]

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        """Event-keyed entry point. Override for trigger='event'.
        Rows in the returned TaskResult.data must include run_id."""
        return TaskResult.failed("Not implemented")


# =====================================================================
# EXAMPLE: A simple task that extracts text from files
# =====================================================================

# import time
# from pathlib import Path
# from plugins.BaseTask import BaseTask, TaskResult
#
# logger = logging.getLogger("ExtractText")
#
#
# class ExtractText(BaseTask):
#     name = "extract_text"
#     modalities = ["text"]
#     reads = []                          # root task — no upstream dependencies
#     writes = ["extracted_text"]
#     requires_services = []              # no models needed
#     output_schema = """
#         CREATE TABLE IF NOT EXISTS extracted_text (
#             path TEXT PRIMARY KEY,
#             content TEXT,
#             char_count INTEGER,
#             extracted_at REAL
#         );
#     """
#     batch_size = 8
#     timeout = 120
#
#     def run(self, paths, context):
#         results = []
#         for path in paths:
#             try:
#                 parse_result = context.services.get("parser").parse(path, "text")
#                 if not parse_result.success:
#                     results.append(TaskResult.failed(f"Parse failed: {parse_result.error}"))
#                     continue
#
#                 content = parse_result.output or ""
#                 results.append(TaskResult(
#                     success=True,
#                     data=[{
#                         "path": path,
#                         "content": content,
#                         "char_count": len(content),
#                         "extracted_at": time.time(),
#                     }],
#                     also_contains=parse_result.also_contains,
#                 ))
#             except Exception as e:
#                 results.append(TaskResult.failed(str(e)))
#         return results


# =====================================================================
# EXAMPLE: A downstream task that depends on extracted_text
# =====================================================================

# from plugins.BaseTask import BaseTask, TaskResult
#
#
# class EmbedText(BaseTask):
#     name = "embed_text"
#     modalities = ["text"]
#     reads = ["text_chunks"]             # downstream — triggered after chunk_text completes
#     writes = ["text_embeddings"]
#     requires_services = ["text_embedder"]  # must be loaded before dispatch
#     output_schema = """
#         CREATE TABLE IF NOT EXISTS text_embeddings (
#             path TEXT,
#             chunk_index INTEGER,
#             embedding BLOB,
#             model_name TEXT,
#             embedded_at REAL,
#             PRIMARY KEY (path, chunk_index)
#         );
#     """
#     batch_size = 4
#
#     def run(self, paths, context):
#         embedder = context.services.get("text_embedder")
#         if not embedder or not embedder.loaded:
#             return [TaskResult.failed("text_embedder not loaded") for _ in paths]
#
#         results = []
#         for path in paths:
#             rows = context.db.get_task_output("text_chunks", path)
#             texts = [r["content"] for r in rows]
#             embeddings = embedder.encode(texts)
#             data = [
#                 {"path": path, "chunk_index": i, "embedding": emb.tobytes(),
#                  "model_name": embedder.model_name, "embedded_at": time.time()}
#                 for i, emb in enumerate(embeddings)
#             ]
#             results.append(TaskResult(success=True, data=data))
#         return results


# =====================================================================
# EXAMPLE: An event-triggered task (cron-like / tool-triggered)
# =====================================================================

# import time
# from plugins.BaseTask import BaseTask, TaskResult
#
#
# class ClusterEmbeddings(BaseTask):
#     name = "cluster_embeddings"
#     trigger = "event"
#     trigger_channels = ["schedule.tick.daily", "trigger.cluster_now"]
#     reads = ["text_embeddings"]         # ambient read (path-keyed) — SQL join at run time
#     writes = ["embedding_clusters"]
#     requires_services = []
#     output_schema = """
#         CREATE TABLE IF NOT EXISTS embedding_clusters (
#             run_id TEXT,
#             cluster_id INTEGER,
#             path TEXT,
#             chunk_index INTEGER,
#             created_at REAL,
#             PRIMARY KEY (run_id, path, chunk_index)
#         );
#     """
#
#     def run_event(self, run_id, payload, context):
#         # Cross-kind read: pull all embeddings as an ambient SQL join.
#         with context.db.lock:
#             rows = context.db.conn.execute(
#                 "SELECT path, chunk_index, embedding FROM text_embeddings"
#             ).fetchall()
#         # ... run your clustering here ...
#         data = [
#             {"run_id": run_id, "cluster_id": 0, "path": r[0],
#              "chunk_index": r[1], "created_at": time.time()}
#             for r in rows
#         ]
#         return TaskResult(success=True, data=data)
