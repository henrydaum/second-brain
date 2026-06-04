import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from paths import DATA_DIR

# Silence noisy libraries
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("fitz").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

_LOG_FORMAT = "%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s"
_LOG_DATEFMT = "%I:%M%p"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "memory.md").touch(exist_ok=True)
LOG_FILE = DATA_DIR / "app.log"
_file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("Main")

from dataclasses import dataclass, field
from typing import Any

from config import config_manager
from pipeline.database import Database
from pipeline.orchestrator import Orchestrator
from pipeline.watcher import Watcher
from pipeline.event_trigger import EventTrigger
from agent.tool_registry import ToolRegistry
from runtime.bootstrap import start_frontends
from plugins.BaseService import should_autoload_service
from plugins.plugin_discovery import discover_services, discover_tasks, discover_tools, get_plugin_settings


@dataclass
class Scaffold:
	"""Lightweight bag of runtime references for bootstrap and frontends."""
	orchestrator: Any = None
	db: Any = None
	services: dict = field(default_factory=dict)
	config: dict = field(default_factory=dict)
	tool_registry: Any = None
	watcher: Any = None
	event_trigger: Any = None
	frontend_runtime: Any = None
	restart: Any = None


_ROOT = Path(__file__).parent


# Global shutdown event
_shutdown = threading.Event()


def main():
	t_start = time.time()

	# --- 1. Load config ---
	config = config_manager.load()

	if not config["sync_directories"]:
		logger.error("No sync_directories set in config.json. Add at least one folder path.")
		sys.exit(1)

	# --- 1b. Ensure mutable plugin directories exist ---
	from plugins.helpers.plugin_paths import iter_plugin_dirs
	for _plugin_type, d in iter_plugin_dirs():
		if d.is_relative_to(_ROOT / "plugins"):
			continue
		d.mkdir(parents=True, exist_ok=True)

	# --- 1c. Load existing plugin config into runtime config ---
	config_manager.load_plugin_config_early(config)

	# --- 2. Initialize database ---
	t0 = time.time()
	database = Database(config["db_path"])
	logger.info(f"Database ready: {config['db_path']} ({time.time() - t0:.2f}s)")

	# --- 3. Initialize services ---
	t0 = time.time()
	services = discover_services(_ROOT, config)
	logger.info(f"Services discovered: {list(services.keys())} ({time.time() - t0:.2f}s)")

	# --- 3b. Auto-load managed services from config plus installed extensions ---
	for svc_name, svc in services.items():
		if not should_autoload_service(svc_name, svc, config):
			continue
		try:
			svc.load()
			logger.info(f"Auto-loaded service: {svc_name}")
		except Exception as e:
			logger.error(f"Auto-load failed for '{svc_name}': {e}")
	for svc_name in config.get("autoload_services", []):
		if svc_name not in services:
			logger.warning(f"Auto-load: unknown service '{svc_name}', skipping.")

	# --- 4. Initialize orchestrator ---
	orchestrator = Orchestrator(database, config, services)

	# --- 5. Register tasks ---
	t0 = time.time()
	discover_tasks(_ROOT, orchestrator, config)
	logger.info(f"Tasks registered: {list(orchestrator.tasks.keys())} ({time.time() - t0:.2f}s)")

	# --- 5b. Initialize tool registry ---
	t0 = time.time()
	tool_registry = ToolRegistry(database, config, services)
	tool_registry.orchestrator = orchestrator
	orchestrator.tool_registry = tool_registry
	discover_tools(_ROOT, tool_registry, config)
	logger.info(f"Tools registered: {list(tool_registry.tools.keys())} ({time.time() - t0:.2f}s)")

	# --- 5c. Reconcile plugin config defaults ---
	config_manager.reconcile_plugin_config(config, get_plugin_settings())

	# --- 6. Initialize app context ---
	scaffold = Scaffold(orchestrator, database, services, config, tool_registry)

	# --- 6b. Determine which frontends to start ---
	frontends = set(config.get("enabled_frontends", ["repl", "telegram"]))
	logger.info(f"Enabled frontends: {sorted(frontends)}")

	# --- 7. Start orchestrator ---
	orchestrator.start()

	# --- 8. Start watcher ---
	config["_root"] = str(_ROOT)

	watcher = Watcher(orchestrator, database, config)
	watcher.start()
	scaffold.watcher = watcher

	# --- 8b. Start event trigger (bus-driven run enqueue for event tasks) ---
	event_trigger = EventTrigger(orchestrator, database, config)
	event_trigger.start()
	scaffold.event_trigger = event_trigger
	logger.info("-----------------------------")
	logger.info(f"SecondBrain started in {time.time() - t_start:.2f}s. Type /commands for commands, /quit to exit.")

	# --- 9. Shutdown handler ---
	def shutdown(sig=None, frame=None):
		if _shutdown.is_set():
			return  # Already shutting down
		_shutdown.set()
		logger.info("-----------------------------")
		logger.info("Shutting down...")
		event_trigger.stop()
		watcher.stop()
		orchestrator.stop()
		for svc in services.values():
			if getattr(svc, 'loaded', False):
				try:
					t0 = time.time()
					logger.info(f"Unloading model: {svc.model_name}")
					svc.unload()
					logger.debug(f"Unloaded {svc.model_name} in {time.time() - t0:.2f}s")
				except Exception as e:
					logger.debug(f"Model unload error: {e}")
		logger.info("Saving config...")
		config_manager.save(config)
		# Save plugin config separately
		plugin_keys = {entry[1] for entry in get_plugin_settings()}
		plugin_vals = {k: v for k, v in config.items() if k in plugin_keys}
		if plugin_vals:
			config_manager.save_plugin_config(plugin_vals)
		logger.info("Done.")
		os._exit(0)

	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# --- 9b. Restart — hard fallback that re-execs the process ---
	_restart_lock = threading.Lock()

	def restart():
		def _exec_self():
			if not _restart_lock.acquire(blocking=False):
				return
			logger.info("Re-execing process now.")
			args = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
			if sys.platform == "win32":
				# Windows os.execv doesn't truly overlay — the MSVC runtime
				# spawns a child and exits the parent, orphaning the child
				# from the parent's console. Result: terminal returns to
				# prompt and the "new" process is gone. Spawn with a fresh
				# console so the new instance survives the parent exit.
				import subprocess
				subprocess.Popen(
					args,
					cwd=str(_ROOT),
					close_fds=True,
					creationflags=subprocess.CREATE_NEW_CONSOLE,
				)
				os._exit(0)
			# On Unix execv overlays the process in place, keeping stdin
			# attached so the REPL frontend's blocking input() keeps reading
			# from the user's terminal.
			os.execv(sys.executable, args)

		def graceful_then_exec():
			try:
				logger.info("Restart: graceful shutdown starting...")
				event_trigger.stop()
				watcher.stop()
				orchestrator.stop()
				for svc in services.values():
					if getattr(svc, "loaded", False):
						try:
							svc.unload()
						except Exception as e:
							logger.debug(f"Restart: unload '{svc.model_name}' failed: {e}")
				config_manager.save(config)
				plugin_keys = {entry[1] for entry in get_plugin_settings()}
				plugin_vals = {k: v for k, v in config.items() if k in plugin_keys}
				if plugin_vals:
					config_manager.save_plugin_config(plugin_vals)
			except Exception as e:
				logger.error(f"Restart: graceful shutdown error (forcing exec anyway): {e}")
			_exec_self()

		def watchdog_force_exec():
			time.sleep(5.0)
			logger.warning("Restart: graceful shutdown exceeded 5s — forcing re-exec")
			_exec_self()

		threading.Thread(target=watchdog_force_exec, daemon=True, name="restart-watchdog").start()
		threading.Thread(target=graceful_then_exec, daemon=True, name="restart-graceful").start()

	scaffold.restart = restart

	# --- 10. Start frontends via the shared runtime/bootstrap path ---
	scaffold.frontend_runtime, _adapters, _frontend_threads = start_frontends(
		frontends, scaffold, shutdown, _shutdown, tool_registry, services, config, _ROOT
	)
	_bind_runtime_services(services, tool_registry, orchestrator, scaffold.frontend_runtime)

	# --- 10b. DEBUG: dump the full base system prompt at startup ---
	# Quick-and-dirty: prints the no-session base prompt (frontend statements and
	# per-frontend command filtering are added per session at turn time, so they
	# aren't shown here). Delete this block when you're done eyeballing it.
	try:
		_rt = scaffold.frontend_runtime
		_sp = getattr(_rt, "system_prompt", None)
		_sections = _sp() if callable(_sp) else _sp
		if not isinstance(_sections, list):
			_sections = [{"role": "system", "content": str(_sections)}]
		print("\n" + "=" * 80 + "\nFULL SYSTEM PROMPT (startup dump)\n" + "=" * 80)
		for _m in _sections:
			print(f"\n----- role: {_m.get('role')} -----\n{_m.get('content', '')}")
		print("=" * 80 + "\n")
	except Exception:
		logger.exception("System prompt debug dump failed")

	# --- 11. Main thread idles until shutdown ---
	while not _shutdown.is_set():
		_shutdown.wait(timeout=1.0)

def _bind_runtime_services(services, tool_registry, orchestrator, runtime):
	for svc in services.values():
		if hasattr(svc, "bind_runtime"):
			svc.bind_runtime(
				tool_registry=tool_registry,
				orchestrator=orchestrator,
				runtime=runtime,
				command_registry=getattr(runtime, "command_registry", None),
				frontend_manager=getattr(runtime, "frontend_manager", None),
			)


if __name__ == "__main__":
	main()
