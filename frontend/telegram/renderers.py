"""
Telegram media renderers.

Takes a list of file paths, classifies each by Telegram-compatible media type,
groups them into Media Group actions, and returns structured SendAction objects
that bot.py can execute via the Telegram Bot API.

Design philosophy mirrors frontend/gui/renderers.py — group by modality,
render appropriately for the medium. Telegram Media Groups bundle 2-10 files
into a single collage (Photo/Video, Audio, or Document).
"""

import html
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("TelegramRenderers")


# ===================================================================
# TELEGRAM-NATIVE EXTENSION SETS
# ===================================================================

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac"}

# These image formats can't be sent as Telegram photos — fall back to document
UNSUPPORTED_IMAGE_EXTENSIONS = {".heic", ".heif", ".tiff", ".tif", ".svg"}

_INLINE_TEXT_MAX = 3000  # bytes — small text files rendered inline
_MEDIA_GROUP_MAX = 10    # Telegram limit per media group


# ===================================================================
# SEND ACTION
# ===================================================================

@dataclass
class SendAction:
    """Represents a single Telegram API call to send media."""
    method: str                     # "photo", "video", "audio", "document",
                                    # "text", "media_group"
    files: list[Path] = field(default_factory=list)
    group_type: str = ""            # "photo_video", "audio", "document"
    caption: str = ""               # optional caption (first item in group)
    text_content: str = ""          # for inline text rendering


# ===================================================================
# CLASSIFICATION
# ===================================================================

def _classify(path: Path) -> str:
    """Classify a file path into a Telegram send category.

    Returns one of: "photo", "video", "audio", "text", "document"
    """
    ext = path.suffix.lower()
    if ext in PHOTO_EXTENSIONS:
        return "photo"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in UNSUPPORTED_IMAGE_EXTENSIONS:
        return "document"
    # Small text files get inlined
    from Stage_1.registry import get_modality
    modality = get_modality(ext)
    if modality == "text":
        try:
            if path.stat().st_size <= _INLINE_TEXT_MAX:
                return "text"
        except OSError:
            pass
    return "document"


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def prepare_media_actions(
    paths: list[str],
    max_file_size: int = 50 * 1024 * 1024,
) -> list[SendAction]:
    """Classify paths and build SendAction list for Telegram.

    Groups photos+videos into photo_video media groups, audio into
    audio media groups, and everything else into document media groups.
    Single files get individual send actions. Groups >10 are chunked.

    Parameters:
        paths:          List of absolute file path strings.
        max_file_size:  Skip files larger than this (default 50 MB).

    Returns:
        List of SendAction objects ready for _execute_send_actions().
    """
    # Phase 1: validate and classify
    photo_video: list[Path] = []
    audio: list[Path] = []
    documents: list[Path] = []
    text_actions: list[SendAction] = []

    for path_str in paths:
        p = Path(path_str)
        if not p.exists():
            logger.warning(f"Skipping non-existent path: {p}")
            continue
        if not p.is_file():
            logger.info(f"Skipping non-file: {p}")
            continue
        try:
            if p.stat().st_size > max_file_size:
                logger.warning(f"Skipping oversized file ({p.stat().st_size} bytes): {p.name}")
                continue
        except OSError:
            continue

        category = _classify(p)

        if category == "photo" or category == "video":
            photo_video.append(p)
        elif category == "audio":
            audio.append(p)
        elif category == "text":
            # Inline text — each gets its own action
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                escaped = html.escape(content)
                header = f"<b>{html.escape(p.name)}</b>\n<pre>"
                footer = "</pre>"
                available = 4096 - len(header) - len(footer)
                if len(escaped) > available:
                    escaped = escaped[:available - 20] + "\n... (truncated)"
                text_actions.append(SendAction(
                    method="text",
                    text_content=header + escaped + footer,
                ))
            except Exception as e:
                logger.error(f"Failed to read text file {p.name}: {e}")
                documents.append(p)
        else:
            documents.append(p)

    # Phase 2: build actions from grouped files
    actions: list[SendAction] = []

    # Photo + Video group
    actions.extend(_build_group_actions(photo_video, "photo_video"))

    # Audio group
    actions.extend(_build_group_actions(audio, "audio"))

    # Document group
    actions.extend(_build_group_actions(documents, "document"))

    # Inline text actions (always individual)
    actions.extend(text_actions)

    return actions


def _build_group_actions(
    files: list[Path],
    group_type: str,
) -> list[SendAction]:
    """Build SendAction(s) for a list of files in the same media category.

    - 0 files → empty list
    - 1 file  → single send action (photo/video/audio/document)
    - 2-10    → one media_group action
    - >10     → chunked into media_groups of 10
    """
    if not files:
        return []

    if len(files) == 1:
        p = files[0]
        if group_type == "photo_video":
            method = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "photo"
        elif group_type == "audio":
            method = "audio"
        else:
            method = "document"
        return [SendAction(method=method, files=[p])]

    # Multiple files — chunk into groups of 10
    actions = []
    for i in range(0, len(files), _MEDIA_GROUP_MAX):
        chunk = files[i:i + _MEDIA_GROUP_MAX]
        if len(chunk) == 1:
            # Remainder of 1 — send individually
            p = chunk[0]
            if group_type == "photo_video":
                method = "video" if p.suffix.lower() in VIDEO_EXTENSIONS else "photo"
            elif group_type == "audio":
                method = "audio"
            else:
                method = "document"
            actions.append(SendAction(method=method, files=[p]))
        else:
            actions.append(SendAction(
                method="media_group",
                files=chunk,
                group_type=group_type,
            ))
    return actions
