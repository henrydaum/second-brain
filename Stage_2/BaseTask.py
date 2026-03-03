import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

"""
Task interface.

Every task in the system inherits from BaseTask and implements run().
The orchestrator dispatches tasks without knowing what they do —
it only knows the interface defined here.

A task declares:
	- What files it works on (modalities)
	- What must finish first (depends_on)
	- What tables it writes to (output_tables, output_schema)
	- Whether it batches (batch_size)
	- How to execute (run / run_batch)

Example concrete task:

	class EmbedText(BaseTask):
		name = "embed_text"
		version = 1
		modalities = ["text"]
		depends_on = []
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

		def setup(self, config):
			self.model = load_model(config["embed_model"])

		def is_ready(self):
			return self.model is not None

		def run_batch(self, paths, context):
			# parse, chunk, embed, return results
			...
"""


@dataclass
class TaskResult:
	"""
	What a task hands back after processing one file.

	The orchestrator reads success to decide DONE vs FAILED.
	The data dict is passed to db.write_outputs() if non-empty.
	also_contains is forwarded from the parser for multi-modal discovery.
	"""
	success: bool = True
	error: str = ""
	data: list[dict] = field(default_factory=list)  # rows to write to output table(s)
	also_contains: list[str] = field(default_factory=list)  # from ParseResult

	@staticmethod
	def failed(error: str) -> "TaskResult":
		return TaskResult(success=False, error=error)


@dataclass
class TaskContext:
	"""
	What the orchestrator passes to a task when it runs.

	config:  global settings (model paths, batch sizes, etc.)
	db:      read-only database access for checking prior results
	"""
	config: dict = field(default_factory=dict)
	db: Any = None  # Database instance


class BaseTask:
	"""
	The contract every task implements.

	Class attributes (override these):
		name            Unique identifier. "embed_text", "ocr", etc.
		version         Integer. Bump when the model or logic changes.
						Triggers re-processing of all files for this task.
		modalities      List of modalities this task works on.
						["text"], ["image"], ["text", "image"], etc.
		depends_on      List of task names that must complete first.
						[] means no dependencies — runs as soon as file is cataloged.
		output_tables   List of table names this task writes to.
						Used for cleanup when files are deleted.
		output_schema   Raw SQL to create the output table(s).
						Only CREATE TABLE and CREATE INDEX allowed.
		batch_size      0 = run() called once per file.
						>0 = run_batch() called with up to N files.

	Methods (override these):
		setup(config)           Called once at registration. Load models here.
		teardown()              Called on shutdown. Release resources.
		is_ready() -> bool      Can this task execute right now?
		run(path, context)      Process one file. Return TaskResult.
		run_batch(paths, context)   Process multiple files. Return list[TaskResult].
	"""

	# --- Identity ---
	name: str = ""
	version: int = 1

	# --- Routing ---
	modalities: list[str] = []
	depends_on: list[str] = []

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

	def is_ready(self) -> bool:
		"""Can this task execute right now? Check model loaded, API reachable, etc."""
		return True

	def run(self, path: str, context: TaskContext) -> TaskResult:
		"""
		Process one file. Called when batch_size is 1.

		The task calls parse(path, modality) itself to get the data it needs.
		Returns TaskResult with success/failure and any data to store.
		"""
		raise NotImplementedError(f"Task '{self.name}' must implement run()")

	def run_batch(self, paths: list[str], context: TaskContext) -> list[TaskResult]:
		"""
		Process multiple files. Can be used with batch_size=1 for single tasks.

		Default implementation just calls run() in a loop.
		Override for real batching (e.g. batch embedding).
		"""
		return [self.run(path, context) for path in paths]