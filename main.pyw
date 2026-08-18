import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import migrations
from paths import DATA_DIR

# Silence noisy libraries
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("fitz").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

_LOG_FORMAT = "%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s"
_LOG_DATEFMT = "%I:%M%p"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

# The terminal is a user-facing frontend, not the diagnostic log. Keep routine
# INFO chatter (including sandbox/backend timing) in app.log while still
# surfacing warnings and errors that need the operator's attention.
for _handler in logging.getLogger().handlers:
	if isinstance(_handler, logging.StreamHandler):
		_handler.setLevel(logging.WARNING)

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = DATA_DIR / "app.log"

logger = logging.getLogger("Main")

# Before anything reads the trees. Idempotent and near-free once done, so it
# runs every boot rather than behind a marker file that could itself go stale.
for _line in migrations.migrate():
	logger.info("DATA_DIR migration: %s", _line)

# And then make the layout real. A declared root that only appears once
# something lands in it is not a claim about where things go — which is why
# the three trees showed three different folder lists, and why ``scripts/``
# (unwatched, so the watcher never made it) existed nowhere at all.
import trees as _trees

for _made in _trees.materialize():
	logger.info("created tree root: %s", _made)


# Preserve the previous run's log before truncating: whatever ended the last
# generation explains itself there, and an unclean exit is exactly when that
# traceback is worth having.
try:
	if LOG_FILE.exists():
		os.replace(LOG_FILE, LOG_FILE.parent / (LOG_FILE.name + ".1"))
except OSError:
	pass

_file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
logging.getLogger().addHandler(_file_handler)

from dataclasses import dataclass, field
from typing import Any

import llm
import parsing
from config import config_manager
from runtime import notifications
from pipeline.database import Database
from pipeline.orchestrator import Orchestrator
from pipeline.watcher import Watcher
from pipeline.event_trigger import EventTrigger
from agent.tool_registry import ToolRegistry
from runtime.bootstrap import start_frontends
from plugins.native.service import forget_stale_autoloads, should_autoload_service
from plugins.plugin_discovery import discover_services, discover_tasks, discover_tools, get_plugin_settings
from plugins.plugin_watcher import PluginWatcher


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

	# --- 1b. Ensure the mutable trees exist ---
	# Every root, not just the five families: a workspace missing scripts/ or
	# parsers/ makes the agent's first write to one fail on a directory it was
	# told to use. The bundled tree ships in the repo and is skipped.
	import trees as _trees
	for _tree, _root, d in _trees.iter_root_dirs():
		if _tree.builtin:
			continue
		d.mkdir(parents=True, exist_ok=True)

	# --- 1c. Load existing plugin config into runtime config ---
	config_manager.load_plugin_config_early(config)

	# --- 2. Initialize database ---
	t0 = time.time()
	database = Database(config["db_path"])
	logger.info(f"Database ready: {config['db_path']} ({time.time() - t0:.2f}s)")

	# Notifications persist so a panel can be filled on a fresh load rather than
	# only from the moment a client connected. A reference, not a lifecycle —
	# the same arrangement parsing.bind_services makes below. Wired here because
	# the plugin watcher raises notifications and holds no runtime to reach.
	notifications.bind_db(database)

	# --- 3. Initialize services ---
	# The sandbox's host context is wired *first*. Handlers answer Requests
	# from a SecondBrainContext, and a resident service has no session to
	# build one from — so without this every config/db Request a service makes
	# is answered from nothing, silently. Discovery loads services, so this has
	# to be in place before the next line and not after it.
	from runtime.context import kernel_context, set_kernel_parts
	from runtime.ledger import sandbox_sink
	set_kernel_parts(db=database, config=config, root_dir=_ROOT)
	try:
		from sandbox.bridge import get_sandbox
		get_sandbox().bind_context(kernel_context)
		# The flight recorder. Reads are filtered out at the sink or a
		# polling frontend would write twenty rows a second forever.
		get_sandbox().bind_ledger(sandbox_sink(database))
	except Exception:
		logger.exception("could not wire the sandbox host context")

	t0 = time.time()
	services = discover_services(config)
	set_kernel_parts(services=services)
	logger.info(f"Services discovered: {list(services.keys())} ({time.time() - t0:.2f}s)")

	# --- 3a. Discover parsers. Kernel routing, not a service: nothing loads it,
	#         and it has to be answerable before anything asks what a file is. ---
	t0 = time.time()
	parsing.bind_services(services)
	logger.info(f"Parsers discovered: {parsing.discover()} module(s) ({time.time() - t0:.2f}s)")

	# --- 3a-ii. Discover LLM backends and build the brains. Kernel routing for
	#            the same reason parsing is: which models exist is standing
	#            knowledge, while the backends themselves are installable
	#            packages that run in boxes. Only the default profile is
	#            loaded — the rest open a box on first use, so a user with six
	#            profiles written down does not start six processes. ---
	t0 = time.time()
	found = llm.discover()
	llm.refresh(config)
	loaded = llm.load_default(config)
	logger.info(f"LLM backends discovered: {found}; default profile "
	            f"{'loaded' if loaded else 'not loaded'} ({time.time() - t0:.2f}s)")

	# --- 3b. Auto-load managed services from config plus installed extensions ---
	for svc_name, svc in services.items():
		if not should_autoload_service(svc_name, svc, config):
			continue
		try:
			svc.load()
			logger.info(f"Auto-loaded service: {svc_name}")
		except Exception as e:
			logger.error(f"Auto-load failed for '{svc_name}': {e}")
	# A name discovery cannot resolve is a service that is no longer installed
	# (or never was — the package manager used to write the *filename* here).
	# Warning about it on every boot forever is what teaches a person to
	# ignore boot warnings, so it is dropped instead, exactly as bootstrap
	# already does for enabled_frontends.
	forget_stale_autoloads(config, services)

	# --- 4. Initialize orchestrator ---
	orchestrator = Orchestrator(database, config, services)
	set_kernel_parts(orchestrator=orchestrator)

	# --- 5. Register tasks ---
	t0 = time.time()
	discover_tasks(orchestrator)
	logger.info(f"Tasks registered: {list(orchestrator.tasks.keys())} ({time.time() - t0:.2f}s)")

	# --- 5b. Initialize tool registry ---
	t0 = time.time()
	tool_registry = ToolRegistry(database, config, services)
	tool_registry.orchestrator = orchestrator
	orchestrator.tool_registry = tool_registry
	# ``call_tool`` too: tool.call is classified ALWAYS_SAFE, so a service is
	# meant to be able to reach a tool, and the handler refuses for want of
	# the callable when nothing supplies one.
	set_kernel_parts(tool_registry=tool_registry, call_tool=tool_registry.call)
	discover_tools(tool_registry)
	logger.info(f"Tools registered: {list(tool_registry.tools.keys())} ({time.time() - t0:.2f}s)")

	# --- 5c. Reconcile plugin config defaults ---
	config_manager.reconcile_plugin_config(config, get_plugin_settings())

	# --- 6. Initialize app context ---
	scaffold = Scaffold(orchestrator, database, services, config, tool_registry)

	# --- 6b. Determine which frontends to start ---
	# The fallback is the kernel's own frontend and nothing else. Telegram used
	# to be named here, which put a store package in the kernel's defaults.
	frontends = set(config.get("enabled_frontends") or ["repl"])
	logger.info(f"Enabled frontends: {sorted(frontends)}")

	# --- 7. Start orchestrator ---
	orchestrator.start()

	# --- 8. Start watcher ---
	config["_root"] = str(_ROOT)

	watcher = Watcher(orchestrator, database, config)
	watcher.start()
	scaffold.watcher = watcher
	orchestrator.watcher = watcher

	# --- 8b. Start event trigger (bus-driven run enqueue for event tasks) ---
	event_trigger = EventTrigger(orchestrator, database, config)
	event_trigger.start()
	scaffold.event_trigger = event_trigger
	plugin_watcher = None
	logger.info("-----------------------------")
	logger.info(f"SecondBrain started in {time.time() - t_start:.2f}s. Type /commands for commands, /quit to exit.")

	# --- 9. Shutdown handler ---
	def shutdown(sig=None, frame=None):
		if _shutdown.is_set():
			return  # Already shutting down
		_shutdown.set()
		logger.info("-----------------------------")
		logger.info("Shutting down...")
		if plugin_watcher is not None:
			plugin_watcher.stop()
		event_trigger.stop()
		watcher.stop()
		_stop_subagents(scaffold)
		orchestrator.stop()
		# Brains hold live boxes (processes, under isolation), so they are
		# closed explicitly rather than left to the service loop below — which
		# no longer knows about them.
		llm.unload_all()
		for name, svc in services.items():
			if getattr(svc, 'loaded', False):
				try:
					t0 = time.time()
					logger.info(f"Unloading service: {name}")
					svc.unload()
					logger.debug(f"Unloaded {name} in {time.time() - t0:.2f}s")
				except Exception as e:
					logger.debug(f"Service unload error: {e}")
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
				if plugin_watcher is not None:
					plugin_watcher.stop()
				event_trigger.stop()
				watcher.stop()
				_stop_subagents(scaffold)
				orchestrator.stop()
				llm.unload_all()
				for name, svc in services.items():
					if getattr(svc, "loaded", False):
						try:
							svc.unload()
						except Exception as e:
							logger.debug(f"Restart: unload '{name}' failed: {e}")
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
	plugin_watcher = PluginWatcher(
		config,
		services=services,
		tool_registry=tool_registry,
		orchestrator=orchestrator,
		command_registry=getattr(scaffold.frontend_runtime, "command_registry", None),
		frontend_manager=getattr(scaffold.frontend_runtime, "frontend_manager", None),
		runtime=scaffold.frontend_runtime,
	)
	scaffold.frontend_runtime.plugin_watcher = plugin_watcher
	plugin_watcher.start()

	# --- 11. Main thread idles until shutdown ---
	try:
		while not _shutdown.is_set():
			_shutdown.wait(timeout=1.0)
	except KeyboardInterrupt:
		shutdown()

def _stop_subagents(scaffold):
	"""End every background agent before the pieces they run on go away.

	Ahead of the orchestrator and the brains, because a child mid-turn is
	still driving a model call. The pool's threads are not daemons, so a
	child left running holds the interpreter open at exit and /quit looks
	like a hang.
	"""
	registry = getattr(getattr(scaffold, "frontend_runtime", None), "subagents", None)
	if registry is None:
		return
	try:
		registry.stop()
	except Exception as e:
		logger.debug(f"Subagent shutdown failed: {e}")


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
