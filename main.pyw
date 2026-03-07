import logging
import signal
import sys
import time
import threading

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

# Import services
from Stage_0.services.embedService import SentenceTransformerEmbedder
from Stage_0.services.llmService import LMStudioLLM, OpenAILLM
from Stage_0.services.ocrService import WindowsOCR
from Stage_0.services.driveService import GoogleDriveService

# Import tasks
from Stage_2.tasks.task_extract_text import ExtractText
from Stage_2.tasks.task_ocr_images import OCRImages
from Stage_2.tasks.task_extract_container import ExtractContainer
# from Stage_2.tasks.task_embed_text import EmbedText

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
	services = {}

	text_embedder = SentenceTransformerEmbedder(
		model_name=config.get("embed_text_model_name", "BAAI/bge-small-en-v1.5"),
		use_cuda=config.get("embed_use_cuda", False),
		chunk_size=config.get("embed_chunk_size", 512),
	)
	services["text_embedder"] = text_embedder

	image_embedder = SentenceTransformerEmbedder(
		model_name=config.get("embed_image_model_name", "clip-ViT-L-14"),
		use_cuda=config.get("embed_use_cuda", False),
		chunk_size=config.get("embed_chunk_size", 512),
	)
	services["image_embedder"] = image_embedder

	llm = OpenAILLM(
		model_name=config.get("llm_model_name", "gemma-3-4b-it"),
		base_url=config.get("llm_endpoint", "http://localhost:1234/v1"),
	)
	services["llm"] = llm

	drive_service = GoogleDriveService()
	services["google_drive"] = drive_service

	ocr = WindowsOCR()
	services["ocr"] = ocr

	# --- 4. Initialize orchestrator ---
	orchestrator = Orchestrator(database, config, services)

	# --- 5. Register tasks ---
	orchestrator.register_task(ExtractContainer())
	orchestrator.register_task(ExtractText())
	orchestrator.register_task(OCRImages())
	# orchestrator.register_task(EmbedText())
	# orchestrator.register_task(EmbedImages())

	# --- 6. Initialize controller ---
	ctrl = Controller(orchestrator, database, services, config)

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
					svc.unload()
				except Exception:
					pass
		config_manager.save(config)
		logger.info("Done.")
		os._exit(0)

	import os
	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# --- 10. Start REPL on its own thread ---
	repl_thread = threading.Thread(target=_repl, args=(ctrl, shutdown), daemon=True)
	repl_thread.start()

	# --- 11. Main thread just keeps the process alive ---
	while not _shutdown.is_set():
		_shutdown.wait(timeout=1.0)


def _repl(ctrl, shutdown_fn):
	"""
	Simple command loop. Maps user input to controller methods.
	Runs on its own daemon thread so it never blocks the dispatch loop.
	"""
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

			elif cmd == "help":
				print(ctrl.help())

			# --- Services ---
			elif cmd == "services":
				print(ctrl.list_services())

			elif cmd == "load":
				if not arg:
					print("Usage: load <service_name>")
				else:
					print(ctrl.load_service(arg))

			elif cmd == "unload":
				if not arg:
					print("Usage: unload <service_name>")
				else:
					print(ctrl.unload_service(arg))

			# --- Tasks ---
			elif cmd == "tasks":
				print(ctrl.list_tasks())

			elif cmd == "pause":
				if not arg:
					print("Usage: pause <task_name>")
				else:
					print(ctrl.pause_task(arg))

			elif cmd == "unpause":
				if not arg:
					print("Usage: unpause <task_name>")
				else:
					print(ctrl.unpause_task(arg))

			elif cmd == "reset":
				if not arg:
					print("Usage: reset <task_name>")
				else:
					print(ctrl.reset_task(arg))

			elif cmd == "retry":
				if arg.lower() == "all":
					print(ctrl.retry_all())
				elif not arg:
					print("Usage: retry <task_name> | retry all")
				else:
					print(ctrl.retry_task(arg))

			# --- Stats ---
			elif cmd == "stats":
				print(ctrl.stats())

			# --- Direct Query ---
			elif cmd == "sql":
				if not arg:
					print("Usage: sql <SELECT ...>")
					print("  Example: sql SELECT path, status FROM task_queue WHERE status='FAILED'")
				else:
					print(ctrl.query(arg))

			elif cmd == "tables":
				print(ctrl.query(
					"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
				))

			elif cmd == "schema":
				if not arg:
					print("Usage: schema <table_name>")
				else:
					print(ctrl.query(f"PRAGMA table_info({arg})"))

			else:
				print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")

		except (KeyboardInterrupt, EOFError):
			shutdown_fn()
			return


if __name__ == "__main__":
	main()