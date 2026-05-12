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


def _looks_like_hallucination(text: str) -> bool:
    """Heuristic for Whisper's classic non-speech hallucinations (music,
    silence, ambient noise transcribed as repeated YouTube-trained phrases
    like "Thank you." or "Thanks for watching.").

    Two signals:
      1. Unique-word ratio is very low (the same phrase repeated).
      2. Total content is short and matches a known canned phrase.
    """
    import re
    stripped = text.strip()
    if not stripped:
        return False

    words = re.findall(r"\w+", stripped.lower())
    if not words:
        return False

    # Very short outputs that match canned hallucinations.
    canned = {
        "thank you", "thank you.", "thanks for watching", "thanks for watching.",
        "you", "bye", "bye.", ".",
    }
    if stripped.lower() in canned:
        return True

    # Repetition: < 30% unique words across at least 8 words is a strong signal.
    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return True

    return False


class FasterWhisperService(BaseService):
    """Local audio transcription via faster-whisper."""

    shared = True  # transcribe() is stateless

    def __init__(self, model_name="base", device="cuda"):
        """Initialize the faster Whisper service."""
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model = None

    def _load(self):
        """Internal helper to load faster Whisper service."""
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
        """Handle unload."""
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
        """Transcribe an audio file. Returns full transcript as a single string.

        Filters out non-speech segments via Silero VAD and per-segment
        no_speech_prob, then post-filters obvious hallucination patterns
        (e.g. "Thank you. Thank you. Thank you." over music) before returning.
        """
        if not self.loaded or not self.model:
            return ""
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            return ""

        logger.info(f"Transcribing: {Path(audio_path).name}")
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        kept = [
            seg.text.strip()
            for seg in segments
            if seg.no_speech_prob < 0.6 and seg.text.strip()
        ]
        text = " ".join(kept).strip()

        if text and _looks_like_hallucination(text):
            logger.info(
                f"Discarded likely-hallucinated transcript for {Path(audio_path).name}: "
                f"{text[:80]!r}"
            )
            text = ""

        logger.info(
            f"Transcribed {Path(audio_path).name}: "
            f"{len(text)} chars, language={info.language} ({info.language_probability:.0%})"
        )
        return text


def build_services(config: dict) -> dict:
    """Build services."""
    return {
        "whisper": FasterWhisperService(
            model_name=config.get("whisper_model_name", "base"),
            device="cuda" if config.get("whisper_use_cuda", True) else "cpu",
        ),
    }
