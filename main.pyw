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

# Import tasks
from Stage_2.tasks.task_extract_text import ExtractText


def main():
    # --- 1. Load config ---
    cfg = config.load()

    if not cfg["sync_directories"]:
        logger.error("No sync_directories set in config.json. Add at least one folder path.")
        sys.exit(1)

    # --- 2. Initialize database ---
    db = Database(cfg["db_path"])
    logger.info(f"Database: {cfg['db_path']}")

    # --- 3. Initialize orchestrator ---
    orch = Orchestrator(db, cfg)

    # --- 4. Register tasks ---
    orch.register_task(ExtractText())
    # Add more tasks here as you build them:
    # orch.register_task(EmbedText())
    # orch.register_task(LLMSummarize())

    # --- 5. Start orchestrator ---
    orch.start()

    # --- 6. Start watcher ---
    watcher = Watcher(orch, db, cfg)
    watcher.start()

    # --- 7. Run until interrupted ---
    logger.info("Forge is running. Press Ctrl+C to stop.")

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        watcher.stop()
        orch.stop()
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