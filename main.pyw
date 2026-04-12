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
logging.getLogger("httpx").setLevel(logging.WARNING)
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
from Stage_3.tool_registry import ToolRegistry
from plugin_discovery import discover_services, discover_tasks, discover_tools, get_plugin_settings


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

	# --- 1b. Ensure sandbox directories exist ---
	from paths import SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES
	for d in (SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES):
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

	# --- 3b. Auto-load services ---
	for svc_name in config.get("autoload_services", []):
		svc = services.get(svc_name)
		if svc is None:
			logger.warning(f"Auto-load: unknown service '{svc_name}', skipping.")
			continue
		try:
			svc.load()
			logger.info(f"Auto-loaded service: {svc_name}")
		except Exception as e:
			logger.error(f"Auto-load failed for '{svc_name}': {e}")

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
	discover_tools(_ROOT, tool_registry, config)
	logger.info(f"Tools registered: {list(tool_registry.tools.keys())} ({time.time() - t0:.2f}s)")

	# --- 5c. Reconcile plugin config (defaults + migration) ---
	config_manager.reconcile_plugin_config(config, get_plugin_settings())

	# --- Debug: print the full system prompt ---
	# from Stage_3.system_prompt import build_system_prompt
	# prompt = build_system_prompt(database, orchestrator, tool_registry, services)
	# print("\n" + "=" * 80)
	# print("SYSTEM PROMPT")
	# print("=" * 80)
	# print(prompt)
	# print("=" * 80 + "\n")

	# --- 6. Initialize controller ---
	ctrl = Controller(orchestrator, database, services, config, tool_registry)

	# --- 6b. Determine which frontends to start ---
	frontends = set(config.get("enabled_frontends", ["gui", "repl"]))
	if sys.platform != "win32" and "gui" in frontends:
		logger.info("GUI not supported on this platform — skipping.")
		frontends.discard("gui")
	logger.info(f"Enabled frontends: {sorted(frontends)}")

	# --- 7. Start orchestrator ---
	orchestrator.start()

	# --- 8. Start watcher ---
	config["_root"] = str(_ROOT)

	watcher = Watcher(orchestrator, database, config)
	watcher.start()
	ctrl.watcher = watcher
	logger.info("-----------------------------")
	logger.info(f"SecondBrain started in {time.time() - t_start:.2f}s. Type 'help' for commands, 'quit' to exit.")

	# Holds GUI references for shutdown/tray (populated by on_page_ready)
	_page_ref = {"page": None, "close_app": None}

	# --- 9. Shutdown handler ---
	def shutdown(sig=None, frame=None):
		if _shutdown.is_set():
			return  # Already shutting down
		_shutdown.set()
		logger.info("-----------------------------")
		logger.info("Shutting down...")
		# Kill the Flet window FIRST — page.window.destroy() bypasses
		# prevent_close and works reliably from any thread, ensuring the
		# renderer subprocess is torn down before we start cleanup.
		gui_page = _page_ref.get("page")
		if gui_page:
			try:
				gui_page.window.prevent_close = False
				gui_page.window.destroy()
			except Exception:
				pass
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

	# --- 10. Start REPL ---
	if "repl" in frontends:
		from frontend.repl.repl import run_repl
		repl_thread = threading.Thread(
			target=run_repl,
			args=(ctrl, shutdown, _shutdown, tool_registry, services, config, _ROOT),
			daemon=True,
		)
		repl_thread.start()

	# --- 10b. Start Telegram bot ---
	if "telegram" in frontends:
		from frontend.telegram.bot import run_telegram_bot
		telegram_thread = threading.Thread(
			target=run_telegram_bot,
			args=(ctrl, shutdown, _shutdown, tool_registry, services, config, _ROOT),
			daemon=True,
		)
		telegram_thread.start()

	# --- 11. Start GUI or wait ---
	if "gui" in frontends:
		from frontend.gui.app import run_gui

		def on_page_ready(page, close_app):
			_page_ref["page"] = page
			_page_ref["close_app"] = close_app

		# System tray icon via pystray (run_detached)
		tray_icon = None
		try:
			import pystray
			from PIL import Image as PILImage

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
				"SecondBrain",
				icon_img,
				"Second Brain",
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
	else:
		# No GUI — main thread idles until shutdown
		while not _shutdown.is_set():
			_shutdown.wait(timeout=1.0)


if __name__ == "__main__":
	main()
