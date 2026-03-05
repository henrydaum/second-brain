import logging
from dataclasses import dataclass, field
from typing import Any

# Use the unified context
from context import DataRefineryContext

logger = logging.getLogger("BaseTask")

"""
Task interface.

Every task in the system inherits from BaseTask and implements run().
The orchestrator dispatches tasks without knowing what they do —
it only knows the interface defined here.

A task declares:
	- What files it works on (modalities)
	- What must finish first (depends_on)
	- What shared services must be loaded (requires_services)
	- What tables it writes to (output_tables, output_schema)
	- How to execute (run)

Example concrete task:

	class EmbedText(BaseTask):
		name = "embed_text"
		version = 1
		modalities = ["text"]
		depends_on = ["extract_text"]
		requires_services = ["embedder"]
		output_tables = ["embeddings"]
		output_schema = \"""
			CREATE TABLE IF NOT EXISTS embeddings (
				path TEXT,
				chunk_index INTEGER,
				text_content TEXT,
				embedding BLOB,
				PRIMARY KEY (path, chunk_index)
			)
		\"""
		batch_size = 16

		def run(self, paths, context):
			embedder = context.services.get("embedder")
			...
"""


@dataclass
class TaskResult:
	"""
	What a task hands back after processing one file.

	The orchestrator reads success to decide DONE vs FAILED.
	The data dict is passed to db.write_outputs() if non-empty.
	also_contains is forwarded from the parser for multi-modal discovery.
	discovered_paths is for tasks that produce new files (e.g. container
	extraction). The orchestrator registers these as first-class files
	and queues tasks for them, just like the watcher would.
	"""
	success: bool = True
	error: str = ""
	data: list[dict] = field(default_factory=list)  # rows to write to output table(s)
	also_contains: list[str] = field(default_factory=list)  # from ParseResult
	discovered_paths: list[str] = field(default_factory=list)  # new files to register

	@staticmethod
	def failed(error: str) -> "TaskResult":
		return TaskResult(success=False, error=error)


class BaseTask:
	"""
	The contract every task implements.

	Class attributes (override these):
		name              Unique identifier. "embed_text", "ocr", etc.
		version           Integer. Bump when the model or logic changes.
		modalities        List of modalities this task works on.
		depends_on        List of task names that must complete first.
		requires_services List of service names that must be loaded before dispatch.
		                  In manual mode, task sits in queue until user loads the service.
		                  In auto mode, system loads the service before dispatch.
		                  Empty list = no service requirements (e.g. extract_text).
		output_tables     List of table names this task writes to.
		output_schema     Raw SQL to create the output table(s).
		batch_size        How many files to process per run() call.
		max_workers       0 = use global max. >0 = limit concurrent workers for this task.

	Methods (override these):
		setup(config)         Called once at registration.
		teardown()            Called on shutdown.
		run(paths, context)   Process files. Return list[TaskResult].
	"""

	# --- Identity ---
	name: str = ""
	version: int = 1

	# --- Routing ---
	modalities: list[str] = []
	depends_on: list[str] = []

	# --- Service requirements ---
	requires_services: list[str] = []

	# --- Schema ---
	output_tables: list[str] = []
	output_schema: str = ""

	# --- Execution ---
	batch_size: int = 1
	max_workers: int = 0  # 0 = use all available workers

	def setup(self, config: dict):
		"""Called once when the task is registered. Load models, warm caches."""
		pass

	def teardown(self):
		"""Called on shutdown. Release GPU memory, close connections."""
		pass

	def run(self, paths: list[str], context: DataRefineryContext) -> list[TaskResult]:
		"""
		Process multiple files. Return a list of TaskResult objects, one per input path.
		"""
		return [TaskResult.failed("Not implemented") for path in paths]