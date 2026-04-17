"""
Task interface.

Tasks are the background processing layer of Second Brain. The
orchestrator does not need to know what a task means semantically; it
only relies on the interface defined here.

There are two trigger kinds:
- trigger="path":
  keyed by file path, dispatched from file discovery and the path
  dependency graph. Implement run(paths, context).
- trigger="event":
  keyed by run_id, dispatched from EventTrigger whenever a subscribed
  bus channel fires. Implement run_event(run_id, payload, context).

A task declares:
- its trigger kind and, for event tasks, trigger_channels
- the file modalities it roots on, for root path tasks
- the tables it reads and writes
- whether readiness is AND or OR across declared inputs
- which shared services must be loaded first

Dependencies are derived automatically from reads and writes within the
same trigger kind. Cross-kind reads are ambient database reads at run
time, not graph edges. Tasks never reference each other by name.

Example concrete task:

	class EmbedText(BaseTask):
		name = "embed_text"
		modalities = ["text"]
		reads = ["text_chunks"]
		writes = ["text_embeddings"]
		requires_services = ["text_embedder"]
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
			embedder = context.services.get("text_embedder")
			...
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("BaseTask")


@dataclass
class TaskResult:
	"""
	The standardized result returned by a task.

	success:
	    Whether processing succeeded for this path or run.
	error:
	    Human-readable failure reason when success is False.
	data:
	    Rows to write into the task's declared output tables.
	also_contains:
	    Additional modalities discovered during parsing, used for
	    multi-modal fan-out.
	discovered_paths:
	    Newly created or extracted files that should be registered as
	    first-class files and fed back into the pipeline.
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
		name:
		    Stable identifier used in the pipeline and database.
		trigger:
		    "path" (default) or "event". This selects the execution entry
		    point and the dependency graph the task participates in.
		trigger_channels:
		    For event tasks, bus channels that enqueue runs.
		modalities:
		    File types this task roots on. Required for root path tasks.
		    Downstream path tasks usually leave this empty.
		reads:
		    Input tables. Dependencies are derived automatically by
		    matching reads to other tasks' writes.
		writes:
		    Output tables this task writes to.
		require_all_inputs:
		    True means all declared inputs must be ready. False means any
		    declared input is enough to run.
		requires_services:
		    Service names that must be loaded before dispatch.
		output_schema:
		    Raw SQL to create the output table or tables.
		batch_size:
		    How many files to process per run() call.
		max_workers:
		    0 uses the global worker limit. Values greater than 0 cap
		    concurrency for this task.

	Methods (override these):
		setup(config)         Called once at registration.
		teardown()            Called on shutdown.
		run(paths, context)   Process files. Return list[TaskResult].
	"""

	# --- Identity ---
	name: str = ""

	# --- Trigger kind ---
	# "path"  = keyed by file path, dispatched from file discovery and
	#           the path dependency graph (default)
	# "event" = keyed by run_id, dispatched from EventTrigger on declared
	#           bus channels
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
	max_workers: int = 0  # 0 = use the global worker limit
	timeout: int = 300  # Seconds before a PROCESSING entry is considered stuck; choose a value that matches expected batch size and runtime.

	# --- Config settings this plugin needs ---
	# Each entry is a tuple:
	# (title, variable_name, description, default, type_info)
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
		"""Called once at registration time. Override for setup work."""
		pass

	def teardown(self):
		"""Called on shutdown. Override to release resources."""
		pass

	def run(self, paths: list[str], context) -> list[TaskResult]:
		"""
		Path-keyed task entry point.

		Process a batch of file paths and return one TaskResult per input
		path. Override this for trigger="path" tasks.
		"""
		return [TaskResult.failed("Not implemented") for path in paths]

	def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
		"""
		Event-keyed task entry point.

		Called once per triggered run. Return a single TaskResult. Rows in
		result.data must include run_id. Override this for trigger="event"
		tasks.
		"""
		return TaskResult.failed("Not implemented")
