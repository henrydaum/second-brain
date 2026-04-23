"""
Orchestrator.

Central dispatcher for background tasks. The orchestrator does not need
task-specific logic; it only relies on the BaseTask contract and the
reads/writes graph declared by each task.

It is responsible for:
- registering tasks and building same-kind dependency graphs
- backfilling eligible work for already-known files
- gating dispatch on pause state, service readiness, and worker limits
- claiming work from the database and routing it to workers
- re-enqueueing downstream work and recovering stuck entries
"""

import logging
import os
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from context import build_context
from Stage_1.parser_registry import get_modality
from Stage_2.BaseTask import BaseTask, TaskResult
from event_bus import bus
from event_channels import TASK_COMPLETED, TASK_FAILED, SERVICE_LOADED

logger = logging.getLogger("Orchestrator")


class Orchestrator:
	def __init__(self, db, config: dict, services: dict = None):
		self.db = db
		self.config = config
		self.services = services or {}

		# Task registry: name -> BaseTask instance
		self.tasks: dict[str, BaseTask] = {}
		self._task_lock = threading.Lock()

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

		# Dependency graph — built by _build_graph() after all tasks are registered.
		# Two graphs: path-keyed and event-keyed. Cross-kind reads are ambient
		# SQL joins inside run(), NOT edges in either graph.
		self.table_producers: dict[str, str] = {}      # table -> task name that writes it
		self.upstream: dict[str, list[str]] = {}       # path-keyed: task -> upstream path tasks
		self.event_upstream: dict[str, list[str]] = {} # event-keyed: task -> upstream event tasks
		self.event_trigger = None
		self.tool_registry = None

		# Re-check service-blocked tasks whenever a service finishes loading.
		bus.subscribe(SERVICE_LOADED, lambda payload: self.clear_skip_cache())

	def clear_skip_cache(self, name: str = None):
		"""Clear service-skip tracking so tasks are checked again.

		If name is given, clear only that task; otherwise clear all cached
		skip state.
		"""
		if name:
			self.skip_cache.discard(name)
		else:
			self.skip_cache.clear()

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

		with self._task_lock:
			self.tasks[task.name] = task
			max_w = task.max_workers if task.max_workers > 0 else self.max_workers
			self.task_semaphores[task.name] = threading.Semaphore(max_w)
		self._build_graph()
		self.refresh_event_subscriptions()
		logger.info(f"Registered task: {task.name}")

	def unregister_task(self, name: str):
		"""Remove a task from the orchestrator (used by build_plugin on delete)."""
		with self._task_lock:
			removed = self.tasks.pop(name, None)
			self.task_semaphores.pop(name, None)
		if removed:
			self._build_graph()
			self.refresh_event_subscriptions()
			logger.info(f"Unregistered task: {name}")

	def refresh_event_subscriptions(self):
		"""Refresh event-task bus subscriptions when task definitions change."""
		if self.event_trigger is not None:
			self.event_trigger.refresh()

	def _build_graph(self):
		"""Build same-kind dependency graphs from task reads and writes."""

		# Phase 1: map each output table to one producing task. If multiple
		# tasks write the same table, we log it and keep the first writer
		# here; downstream logic still relies on reads matching at runtime.
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

		# Phase 2: for each task, find same-kind upstream producers. Split
		# into path and event graphs; cross-kind reads are ignored here
		# because they behave as ambient SQL reads inside task code.
		self.upstream = {}
		self.event_upstream = {}
		for task in self.tasks.values():
			kind = getattr(task, "trigger", "path")
			same_kind_deps = set()
			for table in task.reads:
				producer_name = self.table_producers.get(table)
				if not producer_name:
					continue
				producer = self.tasks.get(producer_name)
				if producer is None:
					continue
				if getattr(producer, "trigger", "path") == kind:
					same_kind_deps.add(producer_name)
			if kind == "event":
				self.event_upstream[task.name] = list(same_kind_deps)
			else:
				self.upstream[task.name] = list(same_kind_deps)
			if same_kind_deps:
				logger.debug(f"Dependencies for '{task.name}' ({kind}): {list(same_kind_deps)}")
			else:
				logger.debug(f"Root task (no same-kind dependencies): '{task.name}' ({kind})")

	def _backfill_tasks(self):
		"""Enqueue existing files for eligible path tasks.

		Event tasks are never backfilled; they only run when triggered.
		"""
		total_enqueued = 0
		for task in self.tasks.values():
			if getattr(task, "trigger", "path") != "path":
				continue
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
		"""Return paths where at least one upstream task is already DONE."""
		upstream_tasks = self.upstream.get(task.name, [])
		return self.db.get_paths_with_any_task_done(upstream_tasks)

	def _deps_met(self, path: str, task: BaseTask) -> bool:
		"""Return whether upstream dependencies are satisfied for a path."""
		upstream_tasks = self.upstream.get(task.name, [])
		if not upstream_tasks:
			return True
		done = [dep for dep in upstream_tasks if self.db.is_task_done(path, dep)]
		if task.require_all_inputs:
			return len(done) == len(upstream_tasks)
		else:
			return len(done) > 0

	def get_all_downstream(self, task_name: str) -> list[str]:
		"""Return all path-keyed tasks that transitively depend on task_name."""
		root = self.tasks.get(task_name)
		root_kind = getattr(root, "trigger", "path") if root else "path"
		if root_kind != "path":
			return []
		downstream = []
		frontier = [task_name]
		while frontier:
			current = frontier.pop()
			current_task = self.tasks.get(current)
			if current_task is None:
				continue
			for table in current_task.writes:
				for task in self.tasks.values():
					if getattr(task, "trigger", "path") != "path":
						continue
					if table in task.reads and task.name not in downstream:
						downstream.append(task.name)
						frontier.append(task.name)
		return downstream

	def _invalidate_downstream(self, task_name: str, paths: list[str]):
		"""Reset downstream path tasks to PENDING for the given paths."""
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
			if getattr(task, "trigger", "path") != "path":
				continue
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
				if getattr(task, "trigger", "path") != "path":
					continue
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
				if getattr(task, "trigger", "path") != "path":
					continue
				if modality in task.modalities:
					if self._deps_met(path, task):
						self.db.re_enqueue_task(path, task.name)
						logger.debug(
							f"Multi-modal: {Path(path).name} also contains "
							f"'{modality}' → enqueued '{task.name}'"
						)

	# =================================================================
	# EVENT RUNS
	# =================================================================

	def on_run_enqueued(self, run_id: str, task_name: str):
		"""Handle a newly enqueued event-task run.

		Clears the skip cache so the next dispatch tick re-checks the task.
		"""
		self.clear_skip_cache(task_name)
		logger.debug(f"Run enqueued: {run_id}")

	def on_paths_discovered(self, child_paths: list[str]):
		from Stage_1.parser_registry import get_modality, get_supported_extensions
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
		Create SQL DELETE cascade triggers from the path-task graph.

		When upstream rows are deleted or replaced, SQLite triggers remove
		downstream rows automatically. This keeps derived path-task outputs
		in sync when earlier pipeline stages are rerun.

		Shared output tables are skipped because a cascade could delete rows
		from another writer. Those tables must rely on their own keys and
		replacement behavior for correctness.
		"""
		# Detect shared tables written by multiple path tasks. Cascade
		# triggers assume a `path` column and a single logical writer.
		table_writers: dict[str, list[str]] = {}
		for task in self.tasks.values():
			if getattr(task, "trigger", "path") != "path":
				continue
			for table in task.writes:
				table_writers.setdefault(table, []).append(task.name)
		shared_tables = {t for t, writers in table_writers.items() if len(writers) > 1}

		for task in self.tasks.values():
			if getattr(task, "trigger", "path") != "path":
				continue
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
		"""Return a human-readable view of the current path-task pipeline."""
		if not self.tasks:
			return

		lines = ["Path Pipeline:"]

		# Pre-compute downstream children for each task.
		children: dict[str, list[str]] = {name: [] for name in self.tasks}
		for task in self.tasks.values():
			for table in task.writes:
				for downstream in self.tasks.values():
					if table in downstream.reads and downstream.name not in children[task.name]:
						children[task.name].append(downstream.name)

		# Root tasks have no upstream path dependencies.
		roots = [t.name for t in self.tasks.values() if not self.upstream.get(t.name)]

		# Walk from roots to detect orphaned tasks that are not reachable.
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
				# Cross-reference: show the edge but do not expand twice.
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
		# Periodically sweep for stuck PROCESSING entries and reset them to
		# PENDING so the pipeline can make forward progress again.
		stuck_check_interval = 60  # seconds between recovery sweeps
		last_stuck_check = 0.0     # force an immediate first check

		while self.running:
			dispatched_any = False

			now = time.time()
			if now - last_stuck_check >= stuck_check_interval:
				for task in self.tasks.values():
					if getattr(task, "trigger", "path") == "event":
						count = self.db.reset_stuck_runs_for(task.name, task.timeout)
					else:
						count = self.db.reset_stuck_tasks_for(task.name, task.timeout)
					if count > 0:
						logger.warning(
							f"Reset {count} stuck '{task.name}' entries "
							f"(PROCESSING > {task.timeout}s)"
						)
				last_stuck_check = now

			for task in self.tasks.values():
				# Event tasks are dispatched in the second pass below.
				if getattr(task, "trigger", "path") == "event":
					continue

				# Gate 1: paused tasks keep their work in PENDING.
				if task.name in self.paused:
					continue

				# Gate 2: required services must be ready.
				if not self._services_ready(task):
					continue

				# Gate 3: respect this task's concurrency limit.
				sem = self.task_semaphores[task.name]
				if not sem.acquire(blocking=False):
					logger.debug(f"Skipping '{task.name}': all worker slots busy")
					continue

				# Gate 4: atomically claim PENDING work from the database.
				paths = self.db.claim_tasks(task.name, task.batch_size)
				if not paths:
					sem.release()
					continue

				# Gate 5: re-check dependencies right before dispatch in case an
				# upstream task was invalidated after the DB claim.
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

			# Second pass: event-triggered tasks
			if self._dispatch_event_runs():
				dispatched_any = True

			if not dispatched_any:
				time.sleep(self.poll_interval)

	def _dispatch_event_runs(self) -> bool:
		"""Dispatch event-task runs using the same gating logic as path tasks."""
		dispatched = False
		for task in self.tasks.values():
			if getattr(task, "trigger", "path") != "event":
				continue
			if task.name in self.paused:
				continue
			if not self._services_ready(task):
				continue

			sem = self.task_semaphores[task.name]
			if not sem.acquire(blocking=False):
				logger.debug(f"Skipping '{task.name}': all worker slots busy")
				continue

			runs = self.db.claim_runs(task.name, batch_size=1)
			if not runs:
				sem.release()
				continue

			run_id, payload_json = runs[0]
			dispatched = True
			logger.info(f"Dispatching event run {run_id}")
			self.executor.submit(self._execute_event_run_wrapper,
								 task, run_id, payload_json, sem)
		return dispatched

	def _execute_event_run_wrapper(self, task, run_id, payload_json, sem):
		try:
			self._execute_event_run(task, run_id, payload_json)
		finally:
			sem.release()

	def _execute_event_run(self, task: BaseTask, run_id: str, payload_json: str):
		"""Execute one claimed event-task run."""
		import json as _json
		if task.name in self.paused:
			self.db.unclaim_run(run_id)
			logger.info(f"Task '{task.name}' paused — returned run {run_id} to PENDING")
			return

		try:
			payload = _json.loads(payload_json) if payload_json else {}
		except Exception:
			payload = {}

		context = build_context(
			self.db, self.config, self.services,
			tool_registry=self.tool_registry,
			orchestrator=self,
		)

		t0 = time.time()
		try:
			result = task.run_event(run_id, payload, context)
		except Exception as e:
			elapsed = time.time() - t0
			logger.error(f"Event task '{task.name}' (run {run_id}) failed after {elapsed:.2f}s: {e}")
			self.db.fail_run(run_id, str(e))
			bus.emit(TASK_FAILED, {"task_name": task.name, "run_id": run_id, "error": str(e)})
			return

		elapsed = time.time() - t0

		if not result.success:
			self.db.fail_run(run_id, result.error)
			logger.warning(f"Event task '{task.name}' (run {run_id}) failed: {result.error}")
			bus.emit(TASK_FAILED, {"task_name": task.name, "run_id": run_id, "error": result.error})
			return

		if result.data and task.writes:
			try:
				for table in task.writes:
					self.db.write_outputs(table, result.data)
			except Exception as e:
				logger.error(f"Write failed for '{task.name}' (run {run_id}): {e}")
				self.db.fail_run(run_id, f"Write failed: {e}")
				bus.emit(TASK_FAILED, {"task_name": task.name, "run_id": run_id, "error": f"Write failed: {e}"})
				return

		self.db.complete_run(run_id)
		logger.debug(f"Event task '{task.name}' (run {run_id}) done in {elapsed:.2f}s")

		bus.emit(TASK_COMPLETED, {
			"task_name": task.name,
			"run_id": run_id,
			"rows_written": len(result.data) if result.data else 0,
			"duration_s": elapsed,
		})

		# Event-graph fan-out: enqueue downstream event tasks with parent_run_id.
		self._enqueue_event_downstream(task, run_id, payload)

	def _enqueue_event_downstream(self, parent_task: BaseTask, parent_run_id: str, payload: dict):
		"""Enqueue downstream event tasks that depend on this task's outputs."""
		import json as _json
		from uuid import uuid4
		for table in parent_task.writes:
			for task in self.tasks.values():
				if getattr(task, "trigger", "path") != "event":
					continue
				if task.name == parent_task.name:
					continue
				if table in task.reads:
					run_id = f"{task.name}:{uuid4().hex[:12]}"
					self.db.create_run(
						run_id, task.name,
						triggered_by=f"upstream:{parent_task.name}",
						payload_json=_json.dumps(payload or {}),
						parent_run_id=parent_run_id,
					)
					self.on_run_enqueued(run_id, task.name)
					logger.debug(
						f"Event chain: '{parent_task.name}' → enqueued '{task.name}' "
						f"(run {run_id}, parent {parent_run_id})"
					)

	def _execute_wrapper(self, task, paths, sem):
		try:
			self._execute(task, paths)
		finally:
			sem.release()

	def _execute(self, task: BaseTask, paths: list[str]):
		# If the task was paused after dispatch, return the claim to PENDING.
		if task.name in self.paused:
			self.db.unclaim_tasks(task.name, paths)
			logger.info(f"Task '{task.name}' is paused — returned {len(paths)} path(s) to PENDING")
			return

		context = build_context(
			self.db, self.config, self.services,
			tool_registry=self.tool_registry,
			orchestrator=self,
		)

		t0 = time.time()
		try:
			results = task.run(paths, context)
		except Exception as e:
			logger.error(f"Task '{task.name}' batch failed after {time.time() - t0:.2f}s: {e}")
			for path in paths:
				self.db.fail_task(path, task.name, str(e))
				bus.emit(TASK_FAILED, {"task_name": task.name, "path": path, "error": str(e)})
			self._invalidate_downstream(task.name, paths)
			return

		elapsed = time.time() - t0
		per_path = elapsed / len(paths) if paths else 0.0
		logger.debug(f"Task '{task.name}' ran {len(paths)} file(s) in {elapsed:.2f}s")

		failed_paths = []
		for path, result in zip(paths, results):
			self._process_result(task, path, result, failed_paths, per_path)
		if failed_paths:
			self._invalidate_downstream(task.name, failed_paths)

	def _process_result(self, task, path, result, failed_paths, duration_s=0.0):
		if result.success:
			if not self._handle_success(task, path, result, duration_s):
				failed_paths.append(path)
		else:
			self.db.fail_task(path, task.name, result.error)
			failed_paths.append(path)
			logger.warning(
				f"Task '{task.name}' failed on {Path(path).name}: {result.error}"
			)
			bus.emit(TASK_FAILED, {"task_name": task.name, "path": path, "error": result.error})

	def _handle_success(self, task, path, result, duration_s=0.0):
		"""Persist a successful result and trigger downstream effects."""
		# Most tasks write to a single table, but the interface allows more
		# than one declared output table.
		if result.data and task.writes:
			try:
				for table in task.writes:
					self.db.write_outputs(table, result.data)
			except Exception as e:
				logger.error(f"Write failed for '{task.name}' on {Path(path).name}: {e}")
				self.db.fail_task(path, task.name, f"Write failed: {e}")
				bus.emit(TASK_FAILED, {"task_name": task.name, "path": path, "error": f"Write failed: {e}"})
				return False

		# Re-check dependencies before marking complete in case upstream work
		# was invalidated while this task was running.
		if task.reads and not self._deps_met(path, task):
			self.db.re_enqueue_task(path, task.name)
			logger.warning(
				f"Deps no longer met for '{task.name}' on {Path(path).name}, re-enqueued"
			)
			return True  # not a failure; the task was safely re-queued

		self.db.complete_task(path, task.name)
		self.on_task_completed(path, task.name)

		bus.emit(TASK_COMPLETED, {
			"task_name": task.name,
			"path": path,
			"rows_written": len(result.data) if result.data else 0,
			"duration_s": duration_s,
		})

		if result.also_contains:
			self.on_also_contains(path, result.also_contains)
		if result.discovered_paths:
			self.on_paths_discovered(result.discovered_paths)
		return True
