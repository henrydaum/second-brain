from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from attachments.attachment import Attachment

logger = logging.getLogger("AttachmentRegistry")

# Parser signature: func(path: str, services: dict, config: dict) -> str | None
# Returning None (or raising) means "no parsed text available" — the bundle
# will then fall back to the pointer-fallback string at LLM time.
ParserFn = Callable[[str, dict, dict], "str | None"]

_PARSERS: dict[str, ParserFn] = {}
_MODALITIES: dict[str, str] = {}


# Default modality lookup for extensions that don't get an explicit
# parser registration. "image"/"audio"/"video" map to native LLM
# capabilities; "text"/"binary" never have a native cap and always
# fall through to parsed-text or pointer fallback.
_DEFAULT_MODALITY: dict[str, str] = {
    # Image
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".tiff": "image", ".tif": "image",
    # Audio
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".m4a": "audio",
    ".aac": "audio", ".ogg": "audio", ".oga": "audio", ".opus": "audio",
    ".wma": "audio",
    # Video
    ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video",
    ".avi": "video",
    # Text-like
    ".txt": "text", ".md": "text", ".pdf": "text", ".csv": "text",
    ".json": "text", ".html": "text", ".htm": "text", ".log": "text",
    ".rtf": "text", ".xml": "text", ".yml": "text", ".yaml": "text",
}


def _normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


def register(extensions: str | list[str], modality: str, func: ParserFn) -> None:
    """Register a text-blurb parser for one or more extensions."""
    if isinstance(extensions, str):
        extensions = [extensions]
    for ext in extensions:
        ext = _normalize_ext(ext)
        _PARSERS[ext] = func
        _MODALITIES[ext] = modality


def modality_for(extension: str) -> str:
    ext = _normalize_ext(extension)
    return _MODALITIES.get(ext) or _DEFAULT_MODALITY.get(ext) or "binary"


def parse_attachment(
    path: str,
    *,
    file_name: str | None = None,
    services: dict | None = None,
    config: dict | None = None,
) -> Attachment:
    """Build an :class:`Attachment` for a file on disk.

    Looks up the parser registered for the file's extension and runs it
    to produce ``parsed_text``. If no parser is registered or the parser
    fails, ``parsed_text`` is left as ``None`` — the LLM-side fallback
    will turn that into a pointer line.
    """
    p = Path(path)
    ext = p.suffix.lower()
    name = file_name or p.name or "attachment"
    modality = modality_for(ext)

    parsed_text: str | None = None
    metadata: dict = {}
    parser = _PARSERS.get(ext)
    if parser is not None:
        try:
            outcome = parser(str(p), services or {}, config or {})
        except Exception as e:
            logger.warning(f"Attachment parser for {ext} failed on {p.name}: {e}")
            outcome = None
        if isinstance(outcome, tuple):
            text_out, meta_out = outcome
            parsed_text = str(text_out) if text_out else None
            metadata = dict(meta_out or {})
        elif outcome:
            parsed_text = str(outcome)

    return Attachment(
        path=str(p),
        extension=ext,
        file_name=name,
        modality=modality,
        parsed_text=parsed_text,
        metadata=metadata,
    )
