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
import os
import tempfile
from pathlib import Path

from Stage_2.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("OCRImages")


class OCRImages(BaseTask):
    name = "ocr_images"
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
                # Paso 1: Extract images using the appropriate parser
                parse_result = context.parse(path, "image")
                
                if not parse_result.success:
                    logger.error(f"Image parse failed for {Path(path).name}: {parse_result.error}")
                    results.append(TaskResult.failed(parse_result.error))
                    continue

                images = parse_result.output or []
                all_text = []

                # Paso 2: Process each extracted image
                for img in images:
                    temp_path = None
                    try:
                        # Save PIL object to a temporary file for the OCR engine
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                            img.save(temp_file, format="PNG")
                            temp_path = temp_file.name
                        
                        # Run OCR on the temporary file
                        text_chunk = ocr.process_image(temp_path)
                        if text_chunk and text_chunk.strip():
                            all_text.append(text_chunk.strip())
                            
                    finally:
                        # Clean up the temporary file
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass

                final_text = "\n\n".join(all_text).strip()

                if final_text:
                    logger.info(f"OCR extracted {len(final_text)} chars from {Path(path).name}")
                else:
                    logger.info(f"OCR found no text in {Path(path).name}")

                results.append(TaskResult(
                    success=True,
                    data=[{
                        "path": path,
                        "content": final_text,
                        "char_count": len(final_text),
                        "model_name": getattr(ocr, 'model_name', 'unknown'),
                        "extracted_at": time.time(),
                    }],
                ))
            except Exception as e:
                logger.error(f"OCR failed for {Path(path).name}: {e}")
                results.append(TaskResult.failed(str(e)))

        return results