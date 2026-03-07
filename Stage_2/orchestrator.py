import logging
import os
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from context import build_context
from Stage_1.registry import get_modality
from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("Orchestrator")

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
		-> for each task, check paused, services, semaphore
		-> skip if not ready (task stays PENDING)
		-> claim work from DB, route to thread pool
"""


class Orchestrator:
	def __init__(self, db, config: dict, services: dict = {}):
		self.db = db
		self.config = config
		self.services = services

		# Task registry: name -> BaseTask instance
		self.tasks: dict[str, BaseTask] = {}

		# Paused tasks — skipped during dispatch, work stays PENDING
		self.paused: set[str] = set()

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
		logger.info(f"Registered: {task.name} v{task.version}")

		max_w = task.max_workers if task.max_workers > 0 else self.max_workers
		self.task_semaphores[task.name] = threading.Semaphore(max_w)

		self._backfill_task(task)

	def _reset_task_for_reprocessing(self, task: BaseTask):
		self.db.reset_task(task.name)

	def _backfill_task(self, task: BaseTask):
		for modality in task.modalities:
			paths = self.db.get_files_by_modality(modality)
			for path in paths:
				if self._deps_met(path, task):
					self.db.enqueue_task(path, task.name, task.version)

	def _deps_met(self, path: str, task: BaseTask) -> bool:
		for dep in task.depends_on:
			if not self.db.is_task_done(path, dep):
				return False
		return True

	# =================================================================
	# SERVICE CHECK
	# =================================================================

	def _services_ready(self, task: BaseTask) -> bool:
		if not task.requires_services:
			return True

		not_registered = []
		not_loaded = []
		for name in task.requires_services:
			svc = self.services.get(name)
			if svc is None:
				not_registered.append(name)
			elif not svc.loaded:
				not_loaded.append(name)

		if not_registered or not_loaded:
			if task.name not in self._skip_logged:
				if not_registered:
					logger.warning(f"Task '{task.name}' requires unregistered services: {not_registered}")
				if not_loaded:
					logger.info(f"Task '{task.name}' waiting for services to load: {not_loaded}")
				self._skip_logged.add(task.name)
			return False

		self._skip_logged.discard(task.name)
		return True

	# =================================================================
	# FILE EVENTS
	# =================================================================

	def on_file_discovered(self, path: str, extension: str, modality: str):
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
		for task in self.tasks.values():
			if task_name in task.depends_on:
				if self._deps_met(path, task):
					self.db.enqueue_task(path, task.name, task.version)

	def on_also_contains(self, path: str, modalities: list[str]):
		for modality in modalities:
			for task in self.tasks.values():
				if modality in task.modalities:
					if self._deps_met(path, task):
						self.db.enqueue_task(path, task.name, task.version)

	def on_paths_discovered(self, child_paths: list[str]):
		from Stage_1.registry import get_modality, get_supported_extensions
		supported = get_supported_extensions()

		for child_path in child_paths:
			if not os.path.exists(child_path):
				logger.warning(f"Discovered path does not exist, skipping: {child_path}")
				continue

			p = Path(child_path)
			ext = p.suffix.lower()

			if ext not in supported:
				logger.debug(f"Skipping unsupported extension from container: {p.name}")
				continue

			modality = get_modality(ext)
			mtime = os.path.getmtime(child_path)

			self.db.upsert_file(
				path=child_path,
				file_name=p.name,
				extension=ext,
				modality=modality,
				mtime=mtime,
			)

			logger.info(f"[Container] Registered: {p.name} ({modality})")
			self.on_file_discovered(child_path, ext, modality)

	# =================================================================
	# DISPATCH LOOP
	# =================================================================

	def start(self):
		self.running = True

		for task in self.tasks.values():
			count = self.db.reset_stuck_tasks_for(task.name, task.timeout)
			if count > 0:
				logger.info(f"Startup: reset {count} stuck '{task.name}' entries")

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
		# Periodic stuck-task recovery
		stuck_check_interval = 60  # How often (seconds) to sweep for stuck tasks
		last_stuck_check = 0.0     # Force an immediate first check

		while self.running:
			dispatched_any = False

			now = time.time()
			if now - last_stuck_check >= stuck_check_interval:
				for task in self.tasks.values():
					count = self.db.reset_stuck_tasks_for(task.name, task.timeout)
					if count > 0:
						logger.warning(
							f"Reset {count} stuck '{task.name}' entries "
							f"(PROCESSING > {task.timeout}s)"
						)
				last_stuck_check = now

			for task in self.tasks.values():
				# Gate 1: paused?
				if task.name in self.paused:
					continue

				# Gate 2: services loaded?
				if not self._services_ready(task):
					continue

				# Gate 3: semaphore available?
				sem = self.task_semaphores[task.name]
				if not sem.acquire(blocking=False):
					continue

				# Gate 4: any work to do?
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
		context = build_context(self.db, self.config, self.services)

		try:
			results = task.run(paths, context)
		except Exception as e:
			logger.error(f"Task '{task.name}' batch failed: {e}")
			for path in paths:
				self.db.fail_task(path, task.name, str(e))
			return

		for path, result in zip(paths, results):
			if result.success:
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

				if result.discovered_paths:
					self.on_paths_discovered(result.discovered_paths)
			else:
				self.db.fail_task(path, task.name, result.error)
				logger.warning(
					f"Task '{task.name}' failed on {Path(path).name}: {result.error}"
				)