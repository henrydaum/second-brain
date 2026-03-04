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
	datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

import config
from Stage_2.database import Database
from Stage_2.orchestrator import Orchestrator
from Stage_2.watcher import Watcher
from services.manager import ServiceManager

# Import services
from services.embedService import SentenceTransformerEmbedder
from services.llmService import LMStudioLLM, OpenAILLM
from services.ocrService import WindowsOCR

# Import tasks
from Stage_2.tasks.task_extract_text import ExtractText
# from Stage_2.tasks.task_embed_text import EmbedText
from Stage_2.tasks.task_ocr_images import OCRImages


def main():
	# --- 1. Load config ---
	cfg = config.load()

	if not cfg["sync_directories"]:
		logger.error("No sync_directories set in config.json. Add at least one folder path.")
		sys.exit(1)

	# --- 2. Initialize database ---
	db = Database(cfg["db_path"])
	logger.info(f"Database: {cfg['db_path']}")

	# --- 3. Initialize services ---
	svc = ServiceManager(cfg)

	# Shared services — one instance, load/unload controlled
	text_embedder = SentenceTransformerEmbedder(model_name=cfg.get("embed_text_model_name", "BAAI/bge-small-en-v1.5"), use_cuda=cfg.get("embed_use_cuda", False), chunk_size=cfg.get("embed_chunk_size", 512))
	svc.register("text_embedder", text_embedder, shared=True)

	image_embedder = SentenceTransformerEmbedder(model_name=cfg.get("embed_image_model_name", "clip-ViT-L-14"), use_cuda=cfg.get("embed_use_cuda", False), chunk_size=cfg.get("embed_chunk_size", 512))
	svc.register("image_embedder", image_embedder, shared=True)

	llm = LMStudioLLM(model_name=cfg.get("llm_model_name", "gemma-3-4b-it@q4_k_s"))
	svc.register("llm", llm, shared=True)
	
	ocr = WindowsOCR()
	svc.register("ocr", ocr, shared=True)

	# Per-instance services (factories) — uncomment when ready:
	# svc.register("gdrive", GDriveDownloader, shared=False, config=cfg)

	logger.info(f"Service mode: {svc.mode}")

	# --- 4. Initialize orchestrator ---
	orch = Orchestrator(db, cfg, service_manager=svc)

	# --- 5. Register tasks ---
	orch.register_task(ExtractText())
	orch.register_task(OCRImages())
	# orch.register_task(EmbedText())

	# --- 6. Start orchestrator ---
	orch.start()

	# --- 7. Start watcher ---
	watcher = Watcher(orch, db, cfg)
	watcher.start()

	# --- 8. Run until interrupted ---
	logger.info("Forge is running. Press Ctrl+C to stop.")

	def shutdown(sig, frame):
		logger.info("Shutting down...")
		watcher.stop()
		orch.stop()
		svc.shutdown()
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