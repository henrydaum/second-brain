import logging
import os
import signal
import sys
import threading
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
from Stage_3.BaseTool import ToolRegistry
from Stage_0.auto_discover_services import discover as discover_services
from Stage_2.auto_discover_tasks import discover as discover_tasks
from Stage_3.auto_discover_tools import discover as discover_tools
from repl import run_repl


_ROOT = Path(__file__).parent


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
	services = discover_services(_ROOT, config)

	# --- 4. Initialize orchestrator ---
	orchestrator = Orchestrator(database, config, services)

	# --- 5. Register tasks ---
	discover_tasks(_ROOT, orchestrator, config)

	# --- 5b. Initialize tool registry ---
	tool_registry = ToolRegistry(database, config, services)
	discover_tools(_ROOT, tool_registry, config)

	# --- 6. Initialize controller ---
	ctrl = Controller(orchestrator, database, services, config, tool_registry)

	# --- 7. Start orchestrator ---
	orchestrator.start()

	# --- 8. Start watcher ---
	config["_root"] = str(_ROOT)

	def reload_plugins():
		logger.info("Hot-reloading plugins...")
		discover_tasks(_ROOT, orchestrator, config, reload=True)
		discover_tools(_ROOT, tool_registry, config, reload=True)
		logger.info("Plugins reloaded.")

	watcher = Watcher(orchestrator, database, config, on_plugin_changed=reload_plugins)
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

	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# --- 11. Start REPL on its own thread ---
	repl_thread = threading.Thread(
		target=run_repl,
		args=(ctrl, shutdown, _shutdown, tool_registry, services, config, _ROOT),
		daemon=True,
	)
	repl_thread.start()

	# --- 12. Main thread just keeps the process alive ---
	while not _shutdown.is_set():
		_shutdown.wait(timeout=1.0)


if __name__ == "__main__":
	main()
