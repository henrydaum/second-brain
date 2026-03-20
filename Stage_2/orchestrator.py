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
It only knows the BaseTask interface: modalities, reads, writes,
requires_services, batch_size, run().

Dependencies are derived automatically from reads/writes declarations.
If task A writes to table X, and task B reads from table X, then B
depends on A. No explicit wiring needed.

Flow:
	Watcher detects file -> on_file_discovered()
		-> checks which tasks accept this modality
		-> if no deps or all deps met, enqueue

	Worker completes task -> on_task_completed()
		-> walks the graph: which tasks read from this task's output tables?
		-> if deps now met (AND/OR), enqueue

	Dispatch loop:
		-> for each task, check paused, services, semaphore
		-> skip if not ready (task stays PENDING)
		-> claim work from DB, route to thread pool
"""


class Orchestrator:
	def __init__(self, db, config: dict, services: dict = None):
		self.db = db
		self.config = config
		self.services = services or {}

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
		self.skip_cache: set[str] = set()

		# Dependency graph — built by _build_graph() after all tasks are registered
		self.table_producers: dict[str, str] = {}    # table -> task name that writes it
		self.upstream: dict[str, list[str]] = {}      # task name -> upstream task names

	# =================================================================
	# TASK REGISTRATION
	# =================================================================

	def register_task(self, task: BaseTask):
		task.setup(self.config)

		if task.output_schema:
			self.db.ensure_output_table(task.name, task.output_schema)

		self.db.register_task(
			name=task.name,
			writes=task.writes,
			reads=task.reads,
			modalities=task.modalities,
		)

		self.tasks[task.name] = task
		logger.info(f"Registered task: {task.name}")

		max_w = task.max_workers if task.max_workers > 0 else self.max_workers
		self.task_semaphores[task.name] = threading.Semaphore(max_w)

	def _build_graph(self):
		"""Derive the dependency graph from reads/writes declarations.
		Called once after all tasks are registered, before start()."""

		# Phase 1: Map each output table to the task that produces it.
		# If multiple tasks write the same table (e.g. lexical_content), we keep
		# the first writer here but downstream logic uses reads-matching anyway.
		self.table_producers = {}
		for task in self.tasks.values():
			for table in task.writes:
				if table in self.table_producers:
					logger.debug(
						f"Shared table '{table}': written by both "
						f"'{self.table_producers[table]}' and '{task.name}'"
					)
				else:
					self.table_producers[table] = task.name

		# Phase 2: For each task, find which other tasks must complete first.
		# If task B reads from table X, and task A writes to table X, then B depends on A.
		self.upstream = {}
		for task in self.tasks.values():
			deps = set()
			for table in task.reads:
				producer = self.table_producers.get(table)
				if producer:
					deps.add(producer)
			self.upstream[task.name] = list(deps)
			if deps:
				logger.debug(f"Dependencies for '{task.name}': {list(deps)}")
			else:
				logger.debug(f"Root task (no dependencies): '{task.name}'")

	def _backfill_tasks(self):
		"""Enqueue all existing files for tasks whose deps are already met."""
		total_enqueued = 0
		for task in self.tasks.values():
			enqueued = 0
			if task.modalities:
				# Root task — find files by modality
				for modality in task.modalities:
					paths = self.db.get_files_by_modality(modality)
					for path in paths:
						if self._deps_met(path, task):
							self.db.enqueue_task(path, task.name)
							enqueued += 1
			elif task.reads:
				# Downstream task (no modalities) — find paths with upstream done
				paths = self._get_backfill_paths(task)
				for path in paths:
					if self._deps_met(path, task):
						self.db.enqueue_task(path, task.name)
						enqueued += 1
			if enqueued:
				logger.debug(f"Backfill: enqueued {enqueued} entries for '{task.name}'")
			total_enqueued += enqueued
		if total_enqueued:
			logger.info(f"Backfill: {total_enqueued} total entries enqueued across all tasks")

	def _get_backfill_paths(self, task):
		"""Get paths where at least one upstream task is DONE."""
		upstream_tasks = self.upstream.get(task.name, [])
		return self.db.get_paths_with_any_task_done(upstream_tasks)

	def _deps_met(self, path: str, task: BaseTask) -> bool:
		"""Check if upstream dependencies are satisfied for a path.
		AND mode: all upstream tasks must be DONE.
		OR mode: at least one upstream task must be DONE."""
		upstream_tasks = self.upstream.get(task.name, [])
		if not upstream_tasks:
			return True
		done = [dep for dep in upstream_tasks if self.db.is_task_done(path, dep)]
		if task.require_all_inputs:
			return len(done) == len(upstream_tasks)
		else:
			return len(done) > 0

	def get_all_downstream(self, task_name: str) -> list[str]:
		"""Return all task names that transitively depend on task_name."""
		downstream = []
		frontier = [task_name]
		while frontier:
			current = frontier.pop()
			current_task = self.tasks.get(current)
			if current_task is None:
				continue
			for table in current_task.writes:
				for task in self.tasks.values():
					if table in task.reads and task.name not in downstream:
						downstream.append(task.name)
						frontier.append(task.name)
		return downstream

	def _invalidate_downstream(self, task_name: str, paths: list[str]):
		"""Reset downstream tasks to PENDING for specific paths."""
		downstream = self.get_all_downstream(task_name)
		if downstream:
			logger.debug(
				f"Invalidating downstream of '{task_name}': "
				f"{downstream} for {len(paths)} path(s)"
			)
			self.db.invalidate_tasks_for_paths(downstream, paths)

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
			if task.name not in self.skip_cache:
				if not_registered:
					logger.warning(f"Task '{task.name}' requires unregistered services: {not_registered}")
				if not_loaded:
					logger.info(f"Task '{task.name}' waiting for services to load: {not_loaded}")
				self.skip_cache.add(task.name)
			return False

		self.skip_cache.discard(task.name)
		return True

	# =================================================================
	# FILE EVENTS
	# =================================================================

	def on_file_discovered(self, path: str, extension: str, modality: str):
		for task in self.tasks.values():
			if modality in task.modalities:
				if self._deps_met(path, task):
					self.db.re_enqueue_task(path, task.name)
					# Task is being re-queued — invalidate downstream tasks
					self._invalidate_downstream(task.name, [path])

	def on_file_deleted(self, path: str):
		all_tables = []
		for task in self.tasks.values():
			all_tables.extend(task.writes)
		self.db.clean_output_tables(path, all_tables)

		# Cascade: if the deleted file was an archive, also remove all files
		# that were extracted from it (and clean up the temp directory on disk).
		# This recurses — if an extracted child was itself an archive, its
		# children get deleted too.
		try:
			with self.db.lock:
				cur = self.db.conn.execute(
					"SELECT extract_dir FROM extracted_containers WHERE path = ?", (path,)
				)
				row = cur.fetchone()
			if row and row["extract_dir"]:
				child_paths = self.db.get_container_children(row["extract_dir"])
				logger.debug(
					f"Cascading delete: {len(child_paths)} children from container {Path(path).name}"
				)
				for child in child_paths:
					self.on_file_deleted(child)
				import shutil
				extract_dir = row["extract_dir"]
				if os.path.isdir(extract_dir):
					shutil.rmtree(extract_dir, ignore_errors=True)
					logger.info(f"Cleaned up extracted directory: {extract_dir}")
		except Exception:
			pass  # extracted_containers table may not exist yet

		self.db.remove_file(path)

	# =================================================================
	# TASK COMPLETION
	# =================================================================

	def on_task_completed(self, path: str, task_name: str):
		completed_task = self.tasks.get(task_name)
		if not completed_task:
			return
		triggered = []
		for table in completed_task.writes:
			for task in self.tasks.values():
				if table in task.reads:
					if self._deps_met(path, task):
						self.db.re_enqueue_task(path, task.name)
						triggered.append(task.name)
		if triggered:
			logger.debug(
				f"'{task_name}' completed on {Path(path).name} → triggered: {triggered}"
			)

	def on_also_contains(self, path: str, modalities: list[str]):
		for modality in modalities:
			for task in self.tasks.values():
				if modality in task.modalities:
					if self._deps_met(path, task):
						self.db.re_enqueue_task(path, task.name)
						logger.debug(
							f"Multi-modal: {Path(path).name} also contains "
							f"'{modality}' → enqueued '{task.name}'"
						)

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
				source="container",
			)

			logger.info(f"[Container] Registered: {p.name} ({modality})")
			self.on_file_discovered(child_path, ext, modality)

	# =================================================================
	# DISPATCH LOOP
	# =================================================================

	def start(self):
		self.running = True

		t0 = time.time()
		self._build_graph()
		logger.debug(f"Dependency graph built in {time.time() - t0:.3f}s")
		print(self.dependency_pipeline_graph())
		self._create_cascade_triggers()

		t0 = time.time()
		self._backfill_tasks()
		logger.debug(f"Backfill scan completed in {time.time() - t0:.3f}s")

		for task in self.tasks.values():
			count = self.db.reset_stuck_tasks_for(task.name, task.timeout)
			if count > 0:
				logger.info(f"Startup: reset {count} stuck '{task.name}' entries")

		self.dispatch_thread = threading.Thread(
			target=self._dispatch_loop, daemon=True
		)
		self.dispatch_thread.start()
		logger.info(f"Orchestrator started ({self.max_workers} workers)")

	def _create_cascade_triggers(self):
		"""
		Auto-create SQL DELETE cascade triggers from the reads/writes graph.

		For each task that reads from a table, create a trigger so that
		deleting (or replacing) rows in the upstream table automatically
		cleans the downstream task's output table. INSERT OR REPLACE fires
		DELETE triggers in SQLite, so re-running an upstream task cascades
		cleanup through the entire chain automatically.

		Shared output tables (written by multiple tasks) are skipped —
		cascade could delete rows from other writers. For those, INSERT OR
		REPLACE with composite keys handles correctness.
		"""
		# Detect shared tables (written by multiple tasks)
		table_writers: dict[str, list[str]] = {}
		for task in self.tasks.values():
			for table in task.writes:
				table_writers.setdefault(table, []).append(task.name)
		shared_tables = {t for t, writers in table_writers.items() if len(writers) > 1}

		for task in self.tasks.values():
			if not task.reads:
				continue
			for input_table in task.reads:
				for output_table in task.writes:
					if output_table in shared_tables:
						logger.info(f"Skipping cascade {input_table} -> {output_table} (shared table)")
						continue
					self.db.create_cascade_trigger(input_table, output_table)
					logger.info(f"Cascade trigger: {input_table} -> {output_table}")

	def dependency_pipeline_graph(self):
		"""Log a visual representation of the pipeline at startup."""
		if not self.tasks:
			return

		lines = ["Pipeline:"]

		# Pre-compute children (downstream tasks) for each task
		children: dict[str, list[str]] = {name: [] for name in self.tasks}
		for task in self.tasks.values():
			for table in task.writes:
				for downstream in self.tasks.values():
					if table in downstream.reads and downstream.name not in children[task.name]:
						children[task.name].append(downstream.name)

		# Find root tasks (no upstream dependencies)
		roots = [t.name for t in self.tasks.values() if not self.upstream.get(t.name)]

		# BFS to find orphans (unreachable from roots)
		reachable = set()
		queue = list(roots)
		while queue:
			name = queue.pop(0)
			if name in reachable:
				continue
			reachable.add(name)
			queue.extend(children[name])

		orphans = [name for name in self.tasks if name not in reachable]
		top_level = roots + orphans

		visited: set[str] = set()

		def walk(task_name: str, prefix: str, is_last: bool):
			task = self.tasks[task_name]
			connector = "└── " if is_last else "├── "

			if task_name in visited:
				# Cross-reference: show the edge but mark as already printed
				mode = " (OR)" if (task.reads and not task.require_all_inputs) else ""
				lines.append(f"{prefix}{connector}{task_name}{mode} *")
				return

			visited.add(task_name)
			mode = " (OR)" if (task.reads and not task.require_all_inputs) else ""
			tables_str = ", ".join(f"[{t}]" for t in task.writes)
			lines.append(f"{prefix}{connector}{task_name}{mode} -> {tables_str}")

			child_prefix = prefix + ("    " if is_last else "│   ")
			kids = children[task_name]
			for i, child_name in enumerate(kids):
				walk(child_name, child_prefix, is_last=(i == len(kids) - 1))

		for i, task_name in enumerate(top_level):
			walk(task_name, prefix="", is_last=(i == len(top_level) - 1))

		return "\n".join(lines)

	def stop(self):
		self.running = False
		# Pause all tasks so queued futures flush back to PENDING (see _execute check)
		self.paused.update(self.tasks.keys())
		self.executor.shutdown(wait=False, cancel_futures=True)
		for task in self.tasks.values():
			task.teardown()
		logger.info("Orchestrator stopped")

	def _dispatch_loop(self):
		# Periodic stuck-task recovery — tasks stuck in PROCESSING longer than
		# their timeout are reset to PENDING so they get retried.
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
				# Gate 1: Skip paused tasks — work stays PENDING until unpaused
				if task.name in self.paused:
					continue

				# Gate 2: Skip if required services aren't loaded yet
				if not self._services_ready(task):
					continue

				# Gate 3: Skip if this task already has max concurrent workers running.
				# Non-blocking acquire: if all worker slots are taken, move on.
				sem = self.task_semaphores[task.name]
				if not sem.acquire(blocking=False):
					logger.debug(f"Skipping '{task.name}': all worker slots busy")
					continue

				# Gate 4: Atomically claim PENDING work from DB.
				# If nothing is PENDING, release the semaphore and move on.
				paths = self.db.claim_tasks(task.name, task.batch_size)
				if not paths:
					sem.release()
					continue

				# Gate 5: Double-check dependencies right before dispatch.
				# Protects against a race where an upstream task was invalidated
				# between the DB claim and now.
				if task.reads:
					ready = [p for p in paths if self._deps_met(p, task)]
					not_ready = [p for p in paths if p not in ready]
					if not_ready:
						self.db.unclaim_tasks(task.name, not_ready)
						logger.debug(
							f"Unclaimed {len(not_ready)} '{task.name}' entries: deps not met"
						)
					if not ready:
						sem.release()
						continue
					paths = ready

				dispatched_any = True
				logger.info(f"Dispatching {task.name}: {len(paths)} file(s)")
				self.executor.submit(self._execute_wrapper, task, paths, sem)

			if not dispatched_any:
				time.sleep(self.poll_interval)

	def _execute_wrapper(self, task, paths, sem):
		try:
			self._execute(task, paths)
		finally:
			sem.release()

	def _execute(self, task: BaseTask, paths: list[str]):
		# If task was paused after dispatch, return paths to PENDING
		if task.name in self.paused:
			self.db.unclaim_tasks(task.name, paths)
			logger.info(f"Task '{task.name}' is paused — returned {len(paths)} path(s) to PENDING")
			return

		context = build_context(self.db, self.config, self.services)

		t0 = time.time()
		try:
			results = task.run(paths, context)
		except Exception as e:
			logger.error(f"Task '{task.name}' batch failed after {time.time() - t0:.2f}s: {e}")
			for path in paths:
				self.db.fail_task(path, task.name, str(e))
			self._invalidate_downstream(task.name, paths)
			return

		elapsed = time.time() - t0
		logger.debug(f"Task '{task.name}' ran {len(paths)} file(s) in {elapsed:.2f}s")

		failed_paths = []
		for path, result in zip(paths, results):
			self._process_result(task, path, result, failed_paths)
		if failed_paths:
			self._invalidate_downstream(task.name, failed_paths)

	def _process_result(self, task, path, result, failed_paths):
		if result.success:
			if not self._handle_success(task, path, result):
				failed_paths.append(path)
		else:
			self.db.fail_task(path, task.name, result.error)
			failed_paths.append(path)
			logger.warning(
				f"Task '{task.name}' failed on {Path(path).name}: {result.error}"
			)

	def _handle_success(self, task, path, result):
		"""Write outputs, verify deps, complete task, and cascade. Returns False on write failure."""
		# NOTE: writes same data to all tables — assumes one write table per task
		if result.data and task.writes:
			try:
				for table in task.writes:
					self.db.write_outputs(table, result.data)
			except Exception as e:
				logger.error(f"Write failed for '{task.name}' on {Path(path).name}: {e}")
				self.db.fail_task(path, task.name, f"Write failed: {e}")
				return False

		# Safety: verify deps are still met before completing.
		# Handles the race where upstream was invalidated mid-execution.
		if task.reads and not self._deps_met(path, task):
			self.db.re_enqueue_task(path, task.name)
			logger.warning(
				f"Deps no longer met for '{task.name}' on {Path(path).name}, re-enqueued"
			)
			return True  # not a failure, just re-queued

		self.db.complete_task(path, task.name)
		self.on_task_completed(path, task.name)

		if result.also_contains:
			self.on_also_contains(path, result.also_contains)
		if result.discovered_paths:
			self.on_paths_discovered(result.discovered_paths)
		return True