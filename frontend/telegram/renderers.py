"""
Telegram media renderers.

Takes a list of file paths, classifies each by Telegram-compatible media type,
groups them into Media Group actions, and returns structured SendAction objects
that telegram.py can execute via the Telegram Bot API.

Design philosophy mirrors frontend/gui/renderers.py — group by modality,
render appropriately for the medium. Telegram Media Groups bundle 2-10 files
into a single collage (Photo/Video, Audio, or Document).
"""

import html
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

logger = logging.getLogger("TelegramRenderers")


# ===================================================================
# TELEGRAM-NATIVE EXTENSION SETS
# ===================================================================

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac"}

# These image formats can't be sent as Telegram photos — fall back to document
UNSUPPORTED_IMAGE_EXTENSIONS = {".heic", ".heif", ".tiff", ".tif", ".svg"}

# Google proxy files — JSON shortcuts with a doc_id, rendered as links
_GOOGLE_LINK_MAP = {
    ".gdoc":    "https://docs.google.com/document/d/{doc_id}",
    ".gsheet":  "https://docs.google.com/spreadsheets/d/{doc_id}",
    ".gslides": "https://docs.google.com/presentation/d/{doc_id}",
    ".gdraw":   "https://docs.google.com/drawings/d/{doc_id}",
    ".gform":   "https://docs.google.com/forms/d/{doc_id}",
}

_INLINE_TEXT_MAX = 3000   # bytes — small text files rendered inline
_MEDIA_GROUP_MAX = 10     # Telegram limit per media group
_PHOTO_MAX_SIZE = 10 * 1024 * 1024  # 10 MB — Telegram photo upload limit


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
# PHOTO RESIZING
# ===================================================================

def prepare_photo_bytes(path: Path) -> io.BytesIO:
    """Return a BytesIO ready for Telegram's send_photo.

    If the file is under 10 MB, returns the raw bytes.
    If over 10 MB, progressively downscales until it fits.
    """
    size = path.stat().st_size
    if size <= _PHOTO_MAX_SIZE:
        buf = io.BytesIO(path.read_bytes())
        buf.name = path.name
        return buf

    logger.info(f"Resizing {path.name} ({size / 1024 / 1024:.1f} MB) to fit 10 MB photo limit")
    img = Image.open(path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Shrink by 25% each pass until it fits
    quality = 85
    for scale in [0.75, 0.5, 0.35, 0.25]:
        w, h = img.size
        resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= _PHOTO_MAX_SIZE:
            buf.seek(0)
            buf.name = path.stem + ".jpg"
            logger.info(f"Resized to {buf.tell() / 1024 / 1024:.1f} MB ({int(w * scale)}x{int(h * scale)})")
            return buf

    # Last resort: very small
    buf.seek(0)
    buf.name = path.stem + ".jpg"
    return buf


# ===================================================================
# GOOGLE PROXY LINKS
# ===================================================================

def _google_link(path: Path) -> str | None:
    """If *path* is a Google proxy file, return its web URL. Otherwise None."""
    ext = path.suffix.lower()
    template = _GOOGLE_LINK_MAP.get(ext)
    if not template:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        doc_id = data.get("doc_id") or data.get("url", "").split("/d/")[-1].split("/")[0]
        if doc_id:
            return template.format(doc_id=doc_id)
    except Exception as e:
        logger.warning(f"Could not read Google proxy file {path.name}: {e}")
    return None


# ===================================================================
# CLASSIFICATION
# ===================================================================

def _classify(path: Path) -> str:
    """Classify a file path into a Telegram send category.

    Returns one of: "photo", "video", "audio", "text", "document", "google_link"
    """
    ext = path.suffix.lower()
    if ext in _GOOGLE_LINK_MAP:
        return "google_link"
    if ext in PHOTO_EXTENSIONS:
        return "photo"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in UNSUPPORTED_IMAGE_EXTENSIONS:
        return "document"
    # Small text files get inlined
    from Stage_1.parser_registry import get_modality
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
    skipped: list[str] = []

    for path_str in paths:
        p = Path(path_str)
        if not p.exists():
            logger.warning(f"Skipping non-existent path: {p}")
            continue
        if not p.is_file():
            logger.info(f"Skipping non-file: {p}")
            continue
        try:
            size = p.stat().st_size
            if size > max_file_size:
                skipped.append(f"{p.name} ({size / 1024 / 1024:.1f} MB — exceeds 50 MB limit)")
                continue
        except OSError:
            continue

        category = _classify(p)

        if category == "google_link":
            url = _google_link(p)
            if url:
                text_actions.append(SendAction(
                    method="text",
                    text_content=f'<a href="{html.escape(url)}">{html.escape(p.name)}</a>',
                ))
            else:
                skipped.append(f"{p.name} (could not extract Google link)")
            continue

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

    # Notify about skipped files
    if skipped:
        note = "Skipped files:\n" + "\n".join(f"- {s}" for s in skipped)
        actions.append(SendAction(method="text", text_content=note))

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
