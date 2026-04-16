import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("BaseTask")

"""
Task interface.

Every task in the system inherits from BaseTask. The orchestrator
dispatches tasks without knowing what they do — it only knows the
interface defined here.

There are two trigger kinds:
	- trigger="path"  (default): keyed by file path, dispatched from the
	  watcher and path-dependency graph. Implements run(paths, context).
	- trigger="event": keyed by run_id, dispatched from EventTrigger
	  whenever a declared bus channel fires. Implements
	  run_event(run_id, payload, context).

A task declares:
	- Trigger kind (trigger) and, for event tasks, trigger_channels
	- What files it works on (modalities) — path tasks only
	- What tables it reads from (reads) — dependencies are derived
	  automatically by matching reads to writes across tasks of the
	  SAME trigger kind. Cross-kind reads are ambient SQL joins, not
	  graph edges: path-data changes do NOT auto-invalidate event runs.
	- What tables it writes to (writes)
	- Whether it needs ALL inputs or ANY input (require_all_inputs)
	- What shared services must be loaded (requires_services)

The dependency graph is built automatically from reads/writes — one
graph per trigger kind. Tasks never reference each other by name.

Example concrete task:

	class EmbedText(BaseTask):
		name = "embed_text"
		modalities = ["text"]
		reads = ["text_chunks"]
		writes = ["text_embeddings"]
		requires_services = ["embedder"]
		output_schema = \"""
			CREATE TABLE IF NOT EXISTS text_embeddings (
				path TEXT,
				chunk_index INTEGER,
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
		trigger           "path" (default) or "event". Picks which run
		                  signature the orchestrator calls and which
		                  dependency graph this task participates in.
		trigger_channels  (event tasks) bus channels that enqueue runs.
		modalities        File types this task processes. Required for root
		                  path tasks (no reads). Downstream path tasks leave
		                  this empty — they're triggered by upstream completion.
		reads             Input tables. Dependencies are derived automatically
		                  by matching reads to other tasks' writes.
		writes            Output tables this task writes to.
		require_all_inputs True (default) = AND: all input tables must have
		                  upstream data. False = OR: at least one suffices.
		requires_services List of service names that must be loaded before dispatch.
		                  In manual mode, task sits in queue until user loads the service.
		                  In auto mode, system loads the service before dispatch.
		                  Empty list = no service requirements (e.g. extract_text).
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

	# --- Trigger kind ---
	# "path"  = keyed by file path, dispatched from watcher/path-graph (default)
	# "event" = keyed by run_id, dispatched from EventTrigger on declared bus channels
	trigger: str = "path"
	trigger_channels: list[str] = []   # bus channels this event task subscribes to

	# --- Routing ---
	modalities: list[str] = []

	# --- Data flow ---
	reads: list[str] = []              # input tables (dependencies derived automatically)
	writes: list[str] = []             # output tables
	require_all_inputs: bool = True    # True=AND (all inputs needed), False=OR (any input suffices)

	# --- Service requirements ---
	requires_services: list[str] = []

	# --- Schema ---
	output_schema: str = ""

	# --- Execution ---
	batch_size: int = 1
	max_workers: int = 0  # 0 = use all available workers
	timeout: int = 300  # Seconds before a PROCESSING entry is considered stuck. Make sure to keep in mind the batch size.

	# --- Config settings this plugin needs ---
	# List of tuples: (title, variable_name, description, default, type_info)
	# Same format as SETTINGS_DATA in config_data.py.
	config_settings: list = []

	def __init_subclass__(cls, **kwargs):
		super().__init_subclass__(**kwargs)
		# Prevent subclasses from sharing mutable class attributes.
		# Without .copy(), every subclass would mutate the same list object.
		for attr in ("modalities", "reads", "writes", "requires_services", "config_settings", "trigger_channels"):
			value = getattr(cls, attr)
			if isinstance(value, (dict, list)):
				setattr(cls, attr, value.copy())

	def setup(self, config: dict):
		"""Called once when the task is registered. Load models, warm caches."""
		pass

	def teardown(self):
		"""Called on shutdown. Release GPU memory, close connections."""
		pass

	def run(self, paths: list[str], context) -> list[TaskResult]:
		"""
		Path-keyed task entry point.
		Process multiple files [batching]. Return a list of TaskResult objects, one per input path.
		Override this for trigger="path" tasks (the default).
		"""
		return [TaskResult.failed("Not implemented") for path in paths]

	def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
		"""
		Event-keyed task entry point.
		Called once per triggered run. Return a single TaskResult.
		Rows in result.data must include run_id.
		Override this for trigger="event" tasks.
		"""
		return TaskResult.failed("Not implemented")