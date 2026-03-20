"""
TASK TEMPLATE
=============
This file is a self-contained reference for creating new tasks.
It is NOT imported by the running system — it exists for LLM consumption only.

To create a new task:
  1. Copy this file to Stage_2/tasks/task_<your_name>.py
  2. Rename the class and fill in the class attributes
  3. Implement run()
  4. The system auto-discovers it on startup (or via /reload)


AUTO-DISCOVERY RULES
--------------------
- File must be in Stage_2/tasks/
- File name must start with "task_"
- Class must inherit from BaseTask
- One task class per file (recommended)


DEPENDENCY GRAPH
----------------
Tasks never reference each other by name. Dependencies are derived
automatically from reads/writes:

  If TaskA writes to "text_chunks" and TaskB reads from "text_chunks",
  then TaskB depends on TaskA. The orchestrator figures this out.

Root tasks (reads=[]) are triggered directly by file discovery.
Downstream tasks are triggered when their upstream dependencies complete.


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
                      result = context.parse(path)           # default modality
                      result = context.parse(path, "image")  # specific modality
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
"""

# =====================================================================
# BASE CLASS (copied from Stage_2/BaseTask.py for self-containment)
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

    # --- Routing ---
    modalities: list[str] = []          # file types (root tasks only). e.g. ["text", "image"]

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

    def setup(self, config: dict):
        """Called once at registration. Optional."""
        pass

    def teardown(self):
        """Called on shutdown. Optional."""
        pass

    def run(self, paths: list[str], context) -> list[TaskResult]:
        """Process files. Return one TaskResult per input path."""
        return [TaskResult.failed("Not implemented") for _ in paths]


# =====================================================================
# EXAMPLE: A simple task that extracts text from files
# =====================================================================

# import time
# from pathlib import Path
# from Stage_2.BaseTask import BaseTask, TaskResult
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
#                 parse_result = context.parse(path, "text")
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

# from Stage_2.BaseTask import BaseTask, TaskResult
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
