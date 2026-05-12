"""Audio parser. Transcribes via the ``whisper`` service when available
(see plugins/services/service_whisper.py). Returns None when the service
is not registered or transcription fails — the LLM-side fallback will
then either route to a native-audio model or emit the pointer string."""

import logging

from attachments.registry import register

logger = logging.getLogger("AttachmentParserAudio")

_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma"]


def parse_audio(path: str, services: dict, config: dict) -> str | None:
    """Parse audio."""
    whisper = (services or {}).get("whisper")
    if whisper is None:
        return None
    if hasattr(whisper, "load") and not getattr(whisper, "loaded", True):
        try:
            whisper.load()
        except Exception as e:
            logger.debug(f"whisper load failed: {e}")
            return None
    try:
        text = whisper.transcribe(path)
    except Exception as e:
        logger.debug(f"whisper transcription failed for {path}: {e}")
        return None
    text = (text or "").strip()
    return text or None


register(_EXTENSIONS, "audio", parse_audio)
