"""Frontend-agnostic helpers for turning a cached attachment into inline
context for the agent. Reused across Telegram, REPL, and any future frontend."""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("attachment_inliner")


async def transcribe_audio_inline(
    services,
    cache_path: str | Path,
    file_name: str,
) -> str:
    """Transcribe an audio file via the whisper service and return a fragment
    suitable for appending to user_text. Runs the model off-thread so the
    event loop stays responsive on CPU-only machines.

    Returns the inline fragment (with leading newlines), or a fallback note
    if transcription fails or the service is unavailable.
    """
    cache_path = str(cache_path)
    transcript = ""
    try:
        whisper = services.get("whisper")
        if whisper:
            if not whisper.loaded:
                whisper._load()
            transcript = (await asyncio.to_thread(whisper.transcribe, cache_path) or "").strip()
    except Exception as e:
        logger.warning(f"Whisper transcription failed for {file_name}: {e}")

    if transcript:
        return (
            f"\n\n[The user sent a voice note: {file_name} "
            f"(cached at {cache_path})]\nTranscript:\n{transcript}"
        )
    return (
        f"\n\n[The user sent a voice note: {file_name} "
        f"(cached at {cache_path}). Transcription failed or empty.]"
    )
