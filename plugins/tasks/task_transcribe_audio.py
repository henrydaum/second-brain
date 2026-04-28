"""
Audio transcription task.

Transcribes audio files using the Whisper service and stores the result
in the audio_transcripts table. Mirrors the ocr_images pattern.

Requires the "whisper" service to be loaded.
"""

import logging
import time
from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult

logger = logging.getLogger("TranscribeAudio")


class TranscribeAudio(BaseTask):
    name = "transcribe_audio"
    modalities = ["audio"]
    reads = []
    writes = ["audio_transcripts"]
    requires_services = ["whisper"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS audio_transcripts (
            path TEXT PRIMARY KEY,
            content TEXT,
            char_count INTEGER,
            model_name TEXT,
            transcribed_at REAL
        );
    """
    batch_size = 4
    max_workers = 1  # Whisper is CPU/GPU-heavy, don't saturate
    timeout = 600

    def run(self, paths, context):
        whisper = context.services.get("whisper")
        if whisper is None or not whisper.loaded:
            return [TaskResult.failed("Whisper service not available") for _ in paths]

        results = []
        for path in paths:
            try:
                text = (whisper.transcribe(path) or "").strip()

                if text:
                    logger.info(f"Transcribed {len(text)} chars from {Path(path).name}")
                else:
                    logger.info(f"No speech detected in {Path(path).name}")

                results.append(TaskResult(
                    success=True,
                    data=[{
                        "path": path,
                        "content": text,
                        "char_count": len(text),
                        "model_name": getattr(whisper, "model_name", "unknown"),
                        "transcribed_at": time.time(),
                    }],
                ))
            except Exception as e:
                logger.error(f"Transcription failed for {Path(path).name}: {e}")
                results.append(TaskResult.failed(str(e)))

        return results
