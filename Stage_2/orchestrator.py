import logging
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from context import ForgeContext
from Stage_1.registry import parse
from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger(__name__)

"""
Orchestrator.

The generic dispatcher. Zero knowledge of what any task does.
It only knows the BaseTask interface: modalities, depends_on,
requires_services, batch_size, run().

Flow:
	Watcher detects file -> on_file_discovered()
		-> checks which tasks accept this modality
		-> if no deps or all deps met, enqueue

	Worker completes task -> on_task_completed()
		-> checks which tasks depend on completed one
		-> if ALL deps now met, enqueue

	Dispatch loop:
		-> for each task, check services are loaded
		-> skip if services not ready (task stays PENDING)
		-> claim work from DB, route to thread pool
"""


class Orchestrator:
	def __init__(self, db, config: dict, service_manager=None):
		self.db = db
		self.config = config
		self.service_manager = service_manager

		# Task registry: name -> BaseTask instance
		self.tasks: dict[str, BaseTask] = {}

		# Thread pool
		self.max_workers = config.get("max_workers", 4)
		self.executor = ThreadPoolExecutor(
			max_workers=self.max_workers, thread_name_prefix="Worker"
		)
		self.task_semaphores: dict[str, threading.Semaphore] = {}

		# Dispatch control
		self.running = False
		self.dispatch_thread = None
		self.poll_interval = config.get("poll_interval", 1.0)

		# Track which tasks were skipped last cycle (avoid log spam)
		self._skip_logged: set[str] = set()

	# =================================================================
	# TASK REGISTRATION
	# =================================================================

	def register_task(self, task: BaseTask):
		"""
		Register a task with the system.

		1. Call setup() to initialize
		2. Create output tables from output_schema
		3. Check for version changes
		4. Save registration to DB
		5. Backfill existing files
		"""
		task.setup(self.config)

		if task.output_schema:
			self.db.ensure_output_table(task.name, task.output_schema)

		version_changed = self.db.has_version_changed(task.name, task.version)
		if version_changed:
			logger.info(f"Task '{task.name}' version changed -- will re-process")
			self._reset_task_for_reprocessing(task)

		self.db.register_task(
			name=task.name,
			version=task.version,
			output_table=",".join(task.output_tables),
			modalities=task.modalities,
			depends_on=task.depends_on,
		)

		self.tasks[task.name] = task

		svc_info = f", requires={task.requires_services}" if task.requires_services else ""
		logger.info(
			f"Registered: {task.name} v{task.version} "
			f"(modalities={task.modalities}, depends_on={task.depends_on}, "
			f"batch={task.batch_size}{svc_info})"
		)

		max_w = task.max_workers if task.max_workers > 0 else self.max_workers
		self.task_semaphores[task.name] = threading.Semaphore(max_w)

		self._backfill_task(task)

	def _reset_task_for_reprocessing(self, task: BaseTask):
		"""Version changed -- reset all DONE entries so they get re-queued."""
		self.db.reset_task(task.name)

	def _backfill_task(self, task: BaseTask):
		"""Queue this task for existing files that match and have deps met."""
		for modality in task.modalities:
			paths = self.db.get_files_by_modality(modality)
			for path in paths:
				if self._deps_met(path, task):
					self.db.enqueue_task(path, task.name, task.version)

	def _deps_met(self, path: str, task: BaseTask) -> bool:
		"""Are all dependencies satisfied for this file?"""
		for dep in task.depends_on:
			if not self.db.is_task_done(path, dep):
				return False
		return True

	# =================================================================
	# SERVICE CHECK
	# =================================================================

	def _services_ready(self, task: BaseTask) -> bool:
		"""
		Check if all required services for a task are loaded.

		In auto mode:   ServiceManager loads them on demand.
		In manual mode: ServiceManager just checks.

		Either way, if this returns False, the task is skipped this cycle.
		Tasks stay PENDING in the queue — they'll run once services are loaded.
		"""
		if not task.requires_services:
			return True

		if self.service_manager is None:
			# No service manager — skip tasks that need services
			if task.name not in self._skip_logged:
				logger.warning(
					f"Task '{task.name}' requires services {task.requires_services} "
					f"but no ServiceManager is configured"
				)
				self._skip_logged.add(task.name)
			return False

		ready = self.service_manager.ensure_loaded(task.requires_services)

		if not ready:
			# Log once per task to avoid spam
			if task.name not in self._skip_logged:
				missing = [
					s for s in task.requires_services
					if not self.service_manager.is_loaded(s)
				]
				logger.info(
					f"Task '{task.name}' waiting for services: {missing}"
				)
				self._skip_logged.add(task.name)
		else:
			# Services came online — clear the skip flag so we log again if they go down
			self._skip_logged.discard(task.name)

		return ready

	# =================================================================
	# FILE EVENTS (called by crawler / watcher)
	# =================================================================

	def on_file_discovered(self, path: str, extension: str, modality: str):
		"""
		New or modified file found.
		Queue every task that accepts this modality and has deps met.
		"""
		for task in self.tasks.values():
			if modality in task.modalities:
				if self._deps_met(path, task):
					self.db.re_enqueue_task(path, task.name, task.version)

	def on_file_deleted(self, path: str):
		all_tables = []
		for task in self.tasks.values():
			all_tables.extend(task.output_tables)
		self.db.clean_output_tables(path, all_tables)
		self.db.remove_file(path)

	# =================================================================
	# TASK COMPLETION
	# =================================================================

	def on_task_completed(self, path: str, task_name: str):
		"""
		Task finished. Check downstream tasks:
		if all their deps are now met, queue them.
		"""
		for task in self.tasks.values():
			if task_name in task.depends_on:
				if self._deps_met(path, task):
					self.db.enqueue_task(path, task.name, task.version)

	def on_also_contains(self, path: str, modalities: list[str]):
		"""
		Parser discovered extra modalities (e.g. images in a PDF).
		Queue tasks that accept these modalities.
		"""
		for modality in modalities:
			for task in self.tasks.values():
				if modality in task.modalities:
					if self._deps_met(path, task):
						self.db.enqueue_task(path, task.name, task.version)

	# =================================================================
	# DISPATCH LOOP
	# =================================================================

	def start(self):
		self.running = True

		timeout = self.config.get("task_timeout", 300)
		self.db.reset_stuck_tasks(timeout)

		self.dispatch_thread = threading.Thread(
			target=self._dispatch_loop, daemon=True
		)
		self.dispatch_thread.start()
		logger.info(f"Orchestrator started ({self.max_workers} workers)")

	def stop(self):
		self.running = False
		self.executor.shutdown(wait=True)
		for task in self.tasks.values():
			task.teardown()
		logger.info("Orchestrator stopped")

	def _dispatch_loop(self):
		"""
		Main loop. For each registered task:
			- Check services are loaded
			- Skip if not ready (tasks stay PENDING)
			- Claim up to batch_size paths from DB
			- Submit to thread pool
		"""
		while self.running:
			dispatched_any = False

			for task in self.tasks.values():
				# Gate 1: services loaded?
				if not self._services_ready(task):
					continue

				# Gate 2: semaphore available?
				sem = self.task_semaphores[task.name]
				if not sem.acquire(blocking=False):
					continue

				# Gate 3: any work to do?
				paths = self.db.claim_tasks(task.name, task.batch_size)
				if not paths:
					sem.release()
					continue

				dispatched_any = True
				self.executor.submit(self._execute_wrapper, task, paths, sem)

			if not dispatched_any:
				time.sleep(self.poll_interval)

	def _execute_wrapper(self, task, paths, sem):
		try:
			self._execute(task, paths)
		finally:
			sem.release()

	def _execute(self, task: BaseTask, paths: list[str]):
		"""
		Run a task on a batch of paths. Called in a worker thread.

		1. Build ForgeContext
		2. Call run()
		3. Per result: write outputs, mark done/failed, trigger downstream
		"""
		context = ForgeContext(
			db=self.db,
			config=self.config,
			services=self.service_manager,
			parse=parse,
		)

		try:
			results = task.run(paths, context)
		except Exception as e:
			logger.error(f"Task '{task.name}' batch failed: {e}")
			for path in paths:
				self.db.fail_task(path, task.name, str(e))
			return

		for path, result in zip(paths, results):
			if result.success:
				# Write outputs
				if result.data and task.output_tables:
					try:
						self.db.write_outputs(task.output_tables[0], result.data)
					except Exception as e:
						logger.error(
							f"Write failed for '{task.name}' on {Path(path).name}: {e}"
						)
						self.db.fail_task(path, task.name, f"Write failed: {e}")
						continue

				self.db.complete_task(path, task.name)
				self.on_task_completed(path, task.name)

				if result.also_contains:
					self.on_also_contains(path, result.also_contains)
			else:
				self.db.fail_task(path, task.name, result.error)
				logger.warning(
					f"Task '{task.name}' failed on {Path(path).name}: {result.error}"
				)