"""
OCR task.

Scans image files for text using Windows OCR (or whatever OCR service
is registered). Stores extracted text in the ocr_text table.

Requires the "ocr" service to be loaded. In manual mode, this task
sits in the queue until the user loads the OCR engine. In auto mode,
the system loads it before dispatching.
"""

import logging
import time
from pathlib import Path

from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("OCRImages")


class OCRImages(BaseTask):
	name = "ocr_images"
	version = 1
	modalities = ["image"]
	depends_on = []
	requires_services = ["ocr"]
	output_tables = ["ocr_text"]
	output_schema = """
		CREATE TABLE IF NOT EXISTS ocr_text (
			path TEXT PRIMARY KEY,
			content TEXT,
			char_count INTEGER,
			model_name TEXT,
			extracted_at REAL
		);
	"""
	batch_size = 4
	max_workers = 1  # OCR is CPU-heavy, don't saturate
	timeout = 300

	def run(self, paths, context):
		ocr = context.services.get("ocr")
		if ocr is None or not ocr.loaded:
			return [TaskResult.failed("OCR service not available") for _ in paths]

		results = []
		for path in paths:
			try:
				text = ocr.process_image(path)

				if text and text.strip():
					logger.info(f"OCR extracted {len(text)} chars from {Path(path).name}")
				else:
					text = ""
					logger.info(f"OCR found no text in {Path(path).name}")

				results.append(TaskResult(
					success=True,
					data=[{
						"path": path,
						"content": text,
						"char_count": len(text),
						"model_name": getattr(ocr, 'model_name', 'unknown'),
						"extracted_at": time.time(),
					}],
				))
			except Exception as e:
				logger.error(f"OCR failed for {Path(path).name}: {e}")
				results.append(TaskResult.failed(str(e)))

		return results