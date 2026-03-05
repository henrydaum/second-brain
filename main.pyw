import logging
import signal
import sys
import time

# Silence noisy libraries
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("fitz").setLevel(logging.WARNING)
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
	datefmt="%I:%M%p",
)

logger = logging.getLogger("Main")

import config as config_manager
from Stage_2.database import Database
from Stage_2.orchestrator import Orchestrator
from Stage_2.watcher import Watcher

# Import services
from services.embedService import SentenceTransformerEmbedder
from services.llmService import LMStudioLLM, OpenAILLM
from services.ocrService import WindowsOCR

# Import tasks
from Stage_2.tasks.task_extract_text import ExtractText
from Stage_2.tasks.task_ocr_images import OCRImages
from Stage_2.tasks.task_extract_container import ExtractContainer
# from Stage_2.tasks.task_embed_text import EmbedText


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

	# Shared services — one instance, load/unload controlled
	text_embedder = SentenceTransformerEmbedder(model_name=config.get("embed_text_model_name", "BAAI/bge-small-en-v1.5"), use_cuda=config.get("embed_use_cuda", False), chunk_size=config.get("embed_chunk_size", 512))
	services["text_embedder"] = text_embedder

	image_embedder = SentenceTransformerEmbedder(model_name=config.get("embed_image_model_name", "clip-ViT-L-14"), use_cuda=config.get("embed_use_cuda", False), chunk_size=config.get("embed_chunk_size", 512))
	services["image_embedder"] = image_embedder

	llm = LMStudioLLM(model_name=config["llm_model_name"])
	services["llm"] = llm
	
	ocr = WindowsOCR()
	services["ocr"] = ocr

	ocr.load()

	# Per-instance services (factories):
	# services["gdrive"] = GDriveDownloader(config=config)

	# --- 4. Initialize orchestrator ---
	orchestrator = Orchestrator(database, config, services)

	# --- 5. Register tasks ---
	orchestrator.register_task(ExtractContainer())
	orchestrator.register_task(ExtractText())
	orchestrator.register_task(OCRImages())
	# orchestrator.register_task(EmbedText())

	# --- 6. Start orchestrator ---
	orchestrator.start()

	# --- 7. Start watcher ---
	watcher = Watcher(orchestrator, database, config)
	watcher.start()
	logger.info("-----------------------------")

	# --- 8. Run until interrupted ---
	logger.info("DataRefinery is running. Press Ctrl+C to stop.")

	def shutdown(sig, frame):
		logger.info("-----------------------------")
		logger.info("Shutting down...")
		watcher.stop()
		orchestrator.stop()
		for service in services.values():
			service.unload()
		config_manager.save(config)
		logger.info("Done.")
		sys.exit(0)

	signal.signal(signal.SIGINT, shutdown)
	signal.signal(signal.SIGTERM, shutdown)

	# Keep main thread alive
	while True:
		try:
			time.sleep(1)
		except KeyboardInterrupt:
			shutdown(None, None)


if __name__ == "__main__":
	main()