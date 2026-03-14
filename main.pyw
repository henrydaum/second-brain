import importlib
import inspect
import logging
import signal
import sys
import time
import threading
import json
from pathlib import Path

# Silence noisy libraries
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("fitz").setLevel(logging.WARNING)
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
	datefmt="%I:%M%p",
)

logger = logging.getLogger("Main")

import config_manager
from Stage_2.database import Database
from Stage_2.orchestrator import Orchestrator
from Stage_2.watcher import Watcher
from controller import Controller
from Stage_3.agent import Agent
from Stage_3.BaseTool import ToolRegistry
from Stage_3.system_prompt import build_system_prompt


_ROOT = Path(__file__).parent


def _auto_discover_services(config: dict) -> dict:
	"""
	Scan Stage_0/services/ for modules that expose a build_services(config)
	function and collect all returned service instances.

	To add a new service, drop a file into Stage_0/services/ and add a
	module-level build_services(config) -> dict function.
	"""
	services = {}
	services_dir = _ROOT / "Stage_0" / "services"
	for py_file in sorted(services_dir.glob("*.py")):
		if py_file.stem.startswith("_"):
			continue
		module_name = f"Stage_0.services.{py_file.stem}"
		try:
			module = importlib.import_module(module_name)
		except Exception as e:
			logger.warning(f"Could not import {module_name}: {e}")
			continue
		build_fn = getattr(module, "build_services", None)
		if build_fn is None:
			continue
		try:
			built = build_fn(config)
			if built:
				services.update(built)
		except Exception as e:
			logger.warning(f"build_services() in {module_name} failed: {e}")
	return services


def _auto_discover_tasks(orchestrator, config):
	"""
	Scan Stage_2/tasks/task_*.py for BaseTask subclasses and register them.

	To add a new task, drop a task_<name>.py file into Stage_2/tasks/.
	"""
	from Stage_2.BaseTask import BaseTask
	tasks_dir = _ROOT / "Stage_2" / "tasks"
	for py_file in sorted(tasks_dir.glob("task_*.py")):
		module_name = f"Stage_2.tasks.{py_file.stem}"
		try:
			module = importlib.import_module(module_name)
		except Exception as e:
			logger.warning(f"Could not import {module_name}: {e}")
			continue
		for _, cls in inspect.getmembers(module, inspect.isclass):
			if issubclass(cls, BaseTask) and cls is not BaseTask and cls.__module__ == module_name:
				try:
					orchestrator.register_task(cls())
				except Exception as e:
					logger.warning(f"Could not register task {cls.__name__}: {e}")


def _auto_discover_tools(tool_registry, config):
	"""
	Scan Stage_3/tools/tool_*.py for BaseTool subclasses and register them.

	To add a new tool, drop a tool_<name>.py file into Stage_3/tools/.
	"""
	from Stage_3.BaseTool import BaseTool
	tools_dir = _ROOT / "Stage_3" / "tools"
	for py_file in sorted(tools_dir.glob("tool_*.py")):
		module_name = f"Stage_3.tools.{py_file.stem}"
		try:
			module = importlib.import_module(module_name)
		except Exception as e:
			logger.warning(f"Could not import {module_name}: {e}")
			continue
		for _, cls in inspect.getmembers(module, inspect.isclass):
			if issubclass(cls, BaseTool) and cls is not BaseTool and cls.__module__ == module_name:
				try:
					tool_registry.register(cls())
				except Exception as e:
					logger.warning(f"Could not register tool {cls.__name__}: {e}")


# Global shutdown event
_shutdown = threading.Event()


def main():
	# --- 1. Load config ---
	config = config_manager.load()

	if not config["sync_directories"]:
		logger.error("No sync_directories set in config.json. Add at least one folder path.")
		sys.exit(1)

	# --- 2. Initialize database ---
	database = Database(config["db_path"])
	logger.info(f"Database: {config['db_path']}")

	# --- 3. Initialize services ---
	services = _auto_discover_services(config)

	# --- 4. Initialize orchestrator ---
	orchestrator = Orchestrator(database, config, services)

	# --- 5. Register tasks ---
	_auto_discover_tasks(orchestrator, config)

	# --- 5b. Initialize tool registry ---
	tool_registry = ToolRegistry(database, config, services)
	_auto_discover_tools(tool_registry, config)

	# --- 6. Initialize controller ---
	ctrl = Controller(orchestrator, database, services, config, tool_registry)

	# --- 7. Start orchestrator ---
	orchestrator.start()

	# --- 8. Start watcher ---
	watcher = Watcher(orchestrator, database, config)
	watcher.start()
	logger.info("-----------------------------")
	logger.info("DataRefinery is running. Type 'help' for commands, 'quit' to exit.")

	# --- 9. Shutdown handler ---
	def shutdown(sig=None, frame=None):
		if _shutdown.is_set():
			return  # Already shutting down
		_shutdown.set()
		logger.info("-----------------------------")
		logger.info("Shutting down...")
		watcher.stop()
		orchestrator.stop()
		for svc in services.values():
			if getattr(svc, 'loaded', False):
				try:
					logger.info(f"Unloading service: {svc.model_name}")
					svc.unload()
				except Exception as e:
					logger.debug(f"Service unload error: {e}")
		logger.info("Saving config...")
		config_manager.save(config)
		logger.info("Done.")
		os._exit(0)

	import os
	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# --- 11. Start REPL on its own thread ---
	repl_thread = threading.Thread(
		target=_repl,
		args=(ctrl, shutdown, tool_registry, services, config),
		daemon=True,
	)
	repl_thread.start()

	# --- 12. Main thread just keeps the process alive ---
	while not _shutdown.is_set():
		_shutdown.wait(timeout=1.0)


def _repl(ctrl, shutdown_fn, tool_registry, services, config):
	"""
	Simple command loop. Maps user input to controller methods.
	Runs on its own daemon thread so it never blocks the dispatch loop.
	"""
	agent = None

	# --- Command handlers (one per REPL command) ---

	def cmd_help(arg):
		print(ctrl.help())
	
	def cmd_pipeline(arg):
		print(ctrl.orchestrator.dependency_pipeline_graph())

	def cmd_services(arg):
		print(ctrl.list_services())

	def cmd_load(arg):
		print(ctrl.load_service(arg) if arg else "Usage: load <service_name>")

	def cmd_unload(arg):
		print(ctrl.unload_service(arg) if arg else "Usage: unload <service_name>")

	def cmd_tasks(arg):
		print(ctrl.list_tasks())

	def cmd_pause(arg):
		print(ctrl.pause_task(arg) if arg else "Usage: pause <task_name>")

	def cmd_unpause(arg):
		print(ctrl.unpause_task(arg) if arg else "Usage: unpause <task_name>")

	def cmd_reset(arg):
		print(ctrl.reset_task(arg) if arg else "Usage: reset <task_name>")

	def cmd_retry(arg):
		if arg.lower() == "all":
			print(ctrl.retry_all())
		elif arg:
			print(ctrl.retry_task(arg))
		else:
			print("Usage: retry <task_name> | retry all")

	def cmd_stats(arg):
		print(ctrl.stats())

	def cmd_tools(arg):
		print(ctrl.list_tools())

	def cmd_call(arg):
		if not arg:
			print("Usage: call <tool_name> {\"arg\": \"value\"}")
			print("Example: call sql_query {\"sql\": \"SELECT * FROM files LIMIT 5\"}")
			return

		call_parts = arg.split(maxsplit=1)
		tool_name = call_parts[0]
		raw_args = call_parts[1] if len(call_parts) > 1 else "{}"

		try:
			kwargs = json.loads(raw_args)
		except json.JSONDecodeError as e:
			print(f"Invalid JSON arguments: {e}")
			print("Expected format: call <tool_name> {\"key\": \"value\"}")
			return

		print(ctrl.call_tool(tool_name, kwargs))

	def cmd_chat(arg):
		nonlocal agent
		llm = services.get("llm")
		if llm is None or not llm.loaded:
			print("LLM service not loaded. Run 'load llm' first.")
			return

		prompt = build_system_prompt(ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services)
		agent = Agent(llm, tool_registry, config, system_prompt=prompt)
		logger.info("Agent initialized.")

		print("Entering chat mode. Type 'exit' to return to REPL.")
		print("---")

		while not _shutdown.is_set():
			try:
				user_input = input("you> ").strip()
			except (KeyboardInterrupt, EOFError):
				break

			if not user_input:
				continue
			if user_input.lower() in ("exit", "quit", "back"):
				break
			if user_input.lower() == "reset":
				agent.reset()
				print("(conversation history cleared)")
				continue

			try:
				response = agent.chat(user_input)
				print(f"\nassistant> {response}\n")
			except Exception as e:
				logger.error(f"Agent error: {e}")
				print(f"Error: {e}")

		print("---")
		print("Exited chat mode.")

	# --- Command dispatch table ---

	commands = {
		"help": cmd_help, "services": cmd_services,
		"load": cmd_load, "unload": cmd_unload,
		"tasks": cmd_tasks, "pause": cmd_pause,
		"unpause": cmd_unpause, "reset": cmd_reset,
		"retry": cmd_retry, "stats": cmd_stats,
		"tools": cmd_tools, "call": cmd_call,
		"chat": cmd_chat, "pipeline": cmd_pipeline,
	}

	# --- Main loop ---

	while not _shutdown.is_set():
		try:
			raw = input("\n> ").strip()
			if not raw:
				continue

			parts = raw.split(maxsplit=1)
			cmd = parts[0].lower()
			arg = parts[1].strip() if len(parts) > 1 else ""

			if cmd in ("quit", "exit"):
				shutdown_fn()
				return

			handler = commands.get(cmd)
			if handler:
				handler(arg)
			else:
				print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")

		except (KeyboardInterrupt, EOFError):
			shutdown_fn()
			return


if __name__ == "__main__":
	main()
