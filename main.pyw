import logging
from pathlib import Path

import Stage_1.registry as registry
import Stage_2.database as database
import Stage_2.orchestrator as orchestrator
import Stage_2.watcher as watcher

logging.getLogger("pdfminer").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

