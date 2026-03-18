import argparse
import logging
import os
import signal
import sys
import threading
import time
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
	t_start = time.time()

	# --- 1. Load config ---
	config = config_manager.load()

	if not config["sync_directories"]:
		logger.error("No sync_directories set in config.json. Add at least one folder path.")
		sys.exit(1)

	# --- 2. Initialize database ---
	t0 = time.time()
	database = Database(config["db_path"])
	logger.info(f"Database ready: {config['db_path']} ({time.time() - t0:.2f}s)")

	# --- 3. Initialize services ---
	t0 = time.time()
	services = discover_services(_ROOT, config)
	logger.info(f"Services discovered: {list(services.keys())} ({time.time() - t0:.2f}s)")

	# --- 4. Initialize orchestrator ---
	orchestrator = Orchestrator(database, config, services)

	# --- 5. Register tasks ---
	t0 = time.time()
	discover_tasks(_ROOT, orchestrator, config)
	logger.info(f"Tasks registered: {list(orchestrator.tasks.keys())} ({time.time() - t0:.2f}s)")

	# --- 5b. Initialize tool registry ---
	t0 = time.time()
	tool_registry = ToolRegistry(database, config, services)
	discover_tools(_ROOT, tool_registry, config)
	logger.info(f"Tools registered: {list(tool_registry.tools.keys())} ({time.time() - t0:.2f}s)")

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
	logger.info(f"DataRefinery started in {time.time() - t_start:.2f}s. Type 'help' for commands, 'quit' to exit.")

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
					t0 = time.time()
					logger.info(f"Unloading model: {svc.model_name}")
					svc.unload()
					logger.debug(f"Unloaded {svc.model_name} in {time.time() - t0:.2f}s")
				except Exception as e:
					logger.debug(f"Model unload error: {e}")
		logger.info("Saving config...")
		config_manager.save(config)
		logger.info("Done.")
		os._exit(0)

	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# --- 10. Parse CLI args ---
	parser = argparse.ArgumentParser(description="The Data Refinery")
	parser.add_argument("--no-gui", action="store_true", help="Run without GUI (REPL only)")
	args = parser.parse_args()

	# --- 11. Start REPL on its own thread (always, for debugging) ---
	repl_thread = threading.Thread(
		target=run_repl,
		args=(ctrl, shutdown, _shutdown, tool_registry, services, config, _ROOT),
		daemon=True,
	)
	repl_thread.start()

	if args.no_gui:
		# --- 12a. No GUI: main thread waits ---
		while not _shutdown.is_set():
			_shutdown.wait(timeout=1.0)
	else:
		# --- 12b. GUI mode: pystray + Flet ---
		from gui.app import run_gui

		# Holds references for tray interaction
		_page_ref = {"page": None, "close_app": None}

		def on_page_ready(page, close_app):
			_page_ref["page"] = page
			_page_ref["close_app"] = close_app

		# System tray icon via pystray (run_detached)
		tray_icon = None
		try:
			import pystray
			from PIL import Image as PILImage

			# Load app icon
			icon_img = PILImage.open(str(_ROOT / "icon.ico"))

			def show_window(icon, item):
				"""Bring Flet window back to front."""
				p = _page_ref.get("page")
				if p:
					p.window.visible = True
					p.update()

			def quit_app(icon, item):
				icon.stop()
				close_fn = _page_ref.get("close_app")
				if close_fn:
					close_fn()
				else:
					shutdown()

			tray_icon = pystray.Icon(
				"DataRefinery",
				icon_img,
				"The Data Refinery",
				menu=pystray.Menu(
					pystray.MenuItem("Show Window", show_window, default=True),
					pystray.MenuItem("Quit", quit_app),
				),
			)
			tray_icon.run_detached()
			logger.info("System tray icon active.")
		except ImportError:
			logger.warning("pystray not installed — no system tray icon.")
		except Exception as e:
			logger.warning(f"Failed to start tray icon: {e}")

		# Flet on main thread (blocks until window closes)
		try:
			run_gui(ctrl, shutdown, _shutdown, tool_registry, services, config, _ROOT,
			        on_page_ready=on_page_ready, watcher=watcher)
		finally:
			if tray_icon:
				try:
					tray_icon.stop()
				except Exception:
					pass
			if not _shutdown.is_set():
				shutdown()


if __name__ == "__main__":
	main()
