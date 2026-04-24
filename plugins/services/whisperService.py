"""
Whisper transcription service.

Uses faster-whisper (CTranslate2-optimized) for local audio transcription.
Supports CUDA GPU acceleration. The model is downloaded on first load
and cached locally.

Requires: pip install faster-whisper
"""

import gc
import logging
import os
from pathlib import Path

from plugins.BaseService import BaseService
from paths import DATA_DIR

logger = logging.getLogger("WhisperService")

WHISPER_DIR = DATA_DIR / "whisper"


class FasterWhisperService(BaseService):
    """Local audio transcription via faster-whisper."""

    shared = True  # transcribe() is stateless

    def __init__(self, model_name="base", device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load(self):
        from faster_whisper import WhisperModel
        import torch

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            logger.info("CUDA not available, falling back to CPU")
            device = "cpu"

        WHISPER_DIR.mkdir(parents=True, exist_ok=True)

        self.model = WhisperModel(
            self.model_name,
            device=device,
            compute_type="auto",
            download_root=str(WHISPER_DIR),
        )
        self.loaded = True
        return True

    def unload(self):
        if self.model:
            del self.model
            self.model = None
        self.loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Whisper model unloaded.")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file. Returns full transcript as a single string."""
        if not self.loaded or not self.model:
            return ""
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return ""

        logger.info(f"Transcribing: {Path(audio_path).name}")
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info(
            f"Transcribed {Path(audio_path).name}: "
            f"{len(text)} chars, language={info.language} ({info.language_probability:.0%})"
        )
        return text


def build_services(config: dict) -> dict:
    return {
        "whisper": FasterWhisperService(
            model_name=config.get("whisper_model_name", "base"),
            device="cuda" if config.get("whisper_use_cuda", True) else "cpu",
        ),
    }
