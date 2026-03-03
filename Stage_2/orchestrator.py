import logging
from pathlib import Path
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from Stage_2.BaseTask import BaseTask, TaskContext, TaskResult

logger = logging.getLogger(__name__)

"""
Orchestrator.

The generic dispatcher. Zero knowledge of what any task does.
It only knows the BaseTask interface: modalities, depends_on,
batch_size, is_ready(), run_batch().

Flow:
	Watcher detects file -> on_file_discovered()
		-> checks which tasks accept this modality
		-> if no deps or all deps met, enqueue

	Worker completes task -> on_task_completed()
		-> checks which tasks depend on completed one
		-> if ALL deps now met, enqueue

	Dispatch loop polls DB, claims work, routes to thread pool.
"""


class Orchestrator:
	def __init__(self, db, config: dict):
		self.db = db
		self.config = config

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

	# =================================================================
	# TASK REGISTRATION
	# =================================================================

	def register_task(self, task: BaseTask):
		"""
		Register a task with the system.

		1. Call setup() to load models
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
		logger.info(
			f"Registered: {task.name} v{task.version} "
			f"(modalities={task.modalities}, depends_on={task.depends_on}, "
			f"batch={task.batch_size})"
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
					self.db.enqueue_task(path, task.name, task.version)

	def on_file_deleted(self, path: str):
		"""
		File removed from disk.
		Clean up output tables, task queue, and files table.
		"""
		for task in self.tasks.values():
			for table in task.output_tables:
				try:
					self.db.conn.execute(
						f"DELETE FROM {table} WHERE path = ?", (path,)
					)
				except Exception:
					pass
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
		logger.info(f"Orchestrator started ({self.config.get('max_workers', 4)} workers)")

	def stop(self):
		self.running = False
		self.executor.shutdown(wait=True)
		for task in self.tasks.values():
			task.teardown()
		logger.info("Orchestrator stopped")

	def _dispatch_loop(self):
		"""
		Main loop. For each registered task:
			- Skip if not ready
			- Claim up to batch_size paths from DB
			- Submit to thread pool
		"""
		while self.running:
			dispatched_any = False

			for task in self.tasks.values():
				if not task.is_ready():
					continue

				sem = self.task_semaphores[task.name]
				if not sem.acquire(blocking=False):
					continue  # all slots for this task are busy

				paths = self.db.claim_tasks(task.name, task.batch_size)
				if not paths:
					sem.release()  # nothing to do, give the slot back
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

		1. Build context
		2. Call run_batch()
		3. Per result: write outputs, mark done/failed, trigger downstream
		"""
		context = TaskContext(config=self.config, db=self.db)

		try:
			results = task.run_batch(paths, context)
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