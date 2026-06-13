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
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

_LOG_FORMAT = "%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s"
_LOG_DATEFMT = "%I:%M%p"

logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)

DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "memory.md").touch(exist_ok=True)
LOG_FILE = DATA_DIR / "app.log"

logger = logging.getLogger("Main")


# ── Crash-restart launcher ───────────────────────────────────────────
#
# When ``restart_on_crash`` is enabled (the default), the process started by
# the user becomes a tiny supervisor: it runs the real app as a child process
# (marked with SB_SUPERVISED=1) and relaunches it whenever it dies with a
# non-zero exit code — including hard native crashes (segfaults, OOM kills)
# that no in-process supervision can survive. When ``stall_timeout`` > 0 it
# also watches the child's heartbeat file (runtime/heartbeat.py): a child
# that is alive but frozen stops beating, gets its process tree killed, and
# is relaunched the same way. The persistence layer restores
# conversations and suspended forms on the way back up, so a crash costs
# seconds, not state. Clean exits (/quit, Ctrl+C) stop everything.
#
# This branch runs before the heavy imports below, so the supervisor process
# stays a few-MB stdlib-only watchdog (runtime/heartbeat.py is stdlib+paths).

_RESTART_EXIT_CODE = 42  # child asks the supervisor for an intentional relaunch
# Exit codes that mean "the user stopped it", never "it crashed":
# STATUS_CONTROL_C_EXIT on Windows (signed/unsigned), SIGINT death on POSIX.
_CLEAN_STOP_CODES = {0, 0xC000013A, -1073741510, -2, 130}
_SUPERVISE_POLL = 5.0          # seconds between child liveness polls
_STALL_GIVE_UP = 5             # consecutive stall-kills before giving up
_STALL_STREAK_RESET = 3600.0   # uptime (s) that forgives the stall streak


def _restart_on_crash_enabled() -> bool:
	"""Read restart_on_crash straight from config.json (default True).

	The supervisor must not import the config package (it would drag in the
	app), so this is a raw JSON peek. Missing file/key means the default.
	"""
	import json
	try:
		with open(DATA_DIR / "config.json", "r") as f:
			return bool(json.load(f).get("restart_on_crash", True))
	except Exception:
		return True


def _stall_timeout() -> float:
	"""Read stall_timeout from config.json (raw JSON peek, like
	_restart_on_crash_enabled). 0 disables stall detection; nonzero values are
	floored so a hand-edited tiny value can never out-race the beat interval."""
	import json
	from runtime.heartbeat import clamp_stall_timeout, DEFAULT_STALL_TIMEOUT
	try:
		with open(DATA_DIR / "config.json", "r") as f:
			return clamp_stall_timeout(json.load(f).get("stall_timeout", DEFAULT_STALL_TIMEOUT))
	except Exception:
		return DEFAULT_STALL_TIMEOUT


def _supervise() -> int:
	"""Run the app as a supervised child; relaunch on crash or stall.
	Returns the launcher's exit code."""
	import subprocess
	from runtime.heartbeat import _StallMonitor, HEARTBEAT_FILE, kill_tree, read_mtime_ns

	launcher_log = logging.getLogger("Launcher")
	stop_requested = threading.Event()
	signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
	signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())

	args = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
	rapid_failures = 0
	consecutive_stalls = 0
	# Why the previous generation ended ("stall" / "crash" / "" for a first
	# start, clean stop, or intentional /restart) — exported to the child so
	# startup messaging can say "back online after a crash" vs plain "online".
	restart_reason = ""

	while True:
		# Re-peeked per generation so config toggles apply on the next restart.
		stall_timeout = _stall_timeout()
		baseline = read_mtime_ns(HEARTBEAT_FILE)  # previous run's file never arms the monitor
		monitor = _StallMonitor(stall_timeout, baseline) if stall_timeout > 0 else None
		started = time.time()
		env = {**os.environ, "SB_SUPERVISED": "1", "SB_RESTART_REASON": restart_reason}
		proc = subprocess.Popen(args, env=env, cwd=str(Path(__file__).parent))
		stalled = False

		while True:
			try:
				code = proc.wait(timeout=_SUPERVISE_POLL)
				break  # a real exit always wins over the stall flag
			except subprocess.TimeoutExpired:
				pass
			if stop_requested.is_set():
				continue  # user is stopping: wait for the child, never stall-kill
			if monitor and monitor.observe(read_mtime_ns(HEARTBEAT_FILE)):
				stalled = True
				launcher_log.error(
					f"No heartbeat for {stall_timeout:.0f}s — app appears stalled. "
					f"Killing process tree. Check {LOG_FILE}.1 after the restart "
					f"for which loop went silent (Heartbeat warnings name the "
					f"stale probe; the restart rotates the log there).")
				kill_tree(proc)
				code = proc.wait()  # kill is prompt; no timeout games
				break

		uptime = time.time() - started

		# Decision ladder — exit code FIRST, stall flag second. Our kills can
		# never produce a clean code (TerminateProcess→1, SIGKILL→-9), so a
		# clean code during a kill race means the child genuinely finished its
		# own exit just before the kill landed — honor it.
		if code == _RESTART_EXIT_CODE:
			launcher_log.info("Restart requested — relaunching.")
			rapid_failures = 0
			consecutive_stalls = 0
			restart_reason = ""  # intentional restart reads as a normal startup
			continue
		if stop_requested.is_set() or code in _CLEAN_STOP_CODES:
			return 0
		if not _restart_on_crash_enabled():
			launcher_log.error(f"App exited with code {code}; restart_on_crash is disabled — not restarting.")
			return code

		restart_reason = "stall" if stalled else "crash"
		if stalled:
			# A stall ran >= stall_timeout, so it can never trip the <60s
			# rapid-failure test — it gets its own give-up counter, forgiven
			# by a long healthy run or any non-stall exit.
			rapid_failures = 0
			consecutive_stalls = 1 if uptime >= _STALL_STREAK_RESET else consecutive_stalls + 1
			if consecutive_stalls >= _STALL_GIVE_UP:
				launcher_log.error(
					f"App stalled {consecutive_stalls} times in a row — "
					f"giving up. Check {LOG_FILE} for the cause.")
				return 1
			delay = min(2 ** consecutive_stalls, 60)
			launcher_log.error(f"App stalled after {uptime:.0f}s — restarting in {delay}s (Ctrl+C to stop).")
		else:
			# Backoff: a crash after a long healthy run restarts almost
			# instantly; a boot-crash loop backs off and eventually gives up
			# instead of spinning forever on a broken install or bad config.
			consecutive_stalls = 0
			rapid_failures = 0 if uptime >= 60 else rapid_failures + 1
			if rapid_failures >= 5:
				launcher_log.error(
					f"App crashed {rapid_failures} times in quick succession (exit {code}) — "
					f"giving up. Check {LOG_FILE} for the cause.")
				return code
			delay = min(2 ** rapid_failures, 60)
			launcher_log.error(f"App exited with code {code} after {uptime:.0f}s — restarting in {delay}s (Ctrl+C to stop).")

		for _ in range(delay):
			if stop_requested.is_set():
				return 0
			time.sleep(1)


if __name__ == "__main__" and os.environ.get("SB_SUPERVISED") != "1" and _restart_on_crash_enabled():
	sys.exit(_supervise())


# ── The real app (supervised child, or direct run when the launcher is off) ──

# Preserve the previous run's log before truncating: a stall-kill restart must
# not destroy the Heartbeat warning that names the wedged loop.
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

from config import config_manager
from pipeline.database import Database
from pipeline.orchestrator import Orchestrator
from pipeline.watcher import Watcher
from pipeline.event_trigger import EventTrigger
from agent.tool_registry import ToolRegistry
from runtime.bootstrap import start_frontends
from runtime.supervisor import supervisor
from runtime.heartbeat import heartbeat
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

	# --- 0. Note an unclean previous generation (set by the launcher) ---
	_restart_reason = os.environ.get("SB_RESTART_REASON", "")
	if _restart_reason:
		logger.warning(
			f"Recovering from a {'stall-kill' if _restart_reason == 'stall' else 'crash'} — "
			f"the previous run's log was rotated to {LOG_FILE}.1.")

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

	# --- 1d. Bind the plugin supervisor to the live config ---
	supervisor.configure(config)

	# --- 1e. Start the stall-watchdog heartbeat ---
	# Always beats when healthy regardless of stall_timeout (the knob gates
	# only launcher enforcement), so a config mismatch between the two
	# processes can never kill a healthy child.
	heartbeat.start()

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

	# --- 3c. Start the memory watchdog (Phase-1 stopgap; no-op without psutil) ---
	supervisor.start_memory_watchdog()

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
	orchestrator.watcher = watcher

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
		# Release all probes FIRST: zero probes => the beat thread keeps the
		# heartbeat fresh while models unload, so a slow clean shutdown is
		# never stall-killed. A genuinely hung shutdown is deliberately not
		# covered (false-positive bias).
		heartbeat.unregister_all()
		logger.info("-----------------------------")
		logger.info("Shutting down...")
		supervisor.stop_memory_watchdog()
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
			if os.environ.get("SB_SUPERVISED") == "1":
				# Running under the crash-restart launcher: exit with the
				# sentinel code and let it relaunch us in the same console.
				logger.info("Restarting via launcher.")
				os._exit(_RESTART_EXIT_CODE)
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

	# --- 11. Main thread idles until shutdown, proving liveness as it goes ---
	probe = heartbeat.register("main-loop")  # never unregistered: this loop only exits at process death
	while not _shutdown.is_set():
		probe.beat()
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
