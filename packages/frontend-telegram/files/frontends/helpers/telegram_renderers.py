"""Frontend plugin for Telegram renderers."""

from __future__ import annotations

import html
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

logger = logging.getLogger("TelegramRenderers")

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".m4a", ".aac"}
UNSUPPORTED_IMAGE_EXTENSIONS = {".heic", ".heif", ".tiff", ".tif", ".svg"}
_GOOGLE_LINK_MAP = {
    ".gdoc": "https://docs.google.com/document/d/{doc_id}",
    ".gsheet": "https://docs.google.com/spreadsheets/d/{doc_id}",
    ".gslides": "https://docs.google.com/presentation/d/{doc_id}",
    ".gdraw": "https://docs.google.com/drawings/d/{doc_id}",
    ".gform": "https://docs.google.com/forms/d/{doc_id}",
}
_INLINE_TEXT_MAX = 3000
_MEDIA_GROUP_MAX = 10
_PHOTO_MAX_SIZE = 10 * 1024 * 1024
_PHOTO_MAX_DIMENSION_SUM = 10_000
_PHOTO_MAX_RATIO = 20


@dataclass
class SendAction:
    """Send action."""
    method: str
    files: list[Path] = field(default_factory=list)
    group_type: str = ""
    text_content: str = ""


def file_bytes(path: Path) -> io.BytesIO:
    """Handle file bytes."""
    buf = io.BytesIO(path.read_bytes())
    buf.name = path.name
    return buf


def prepare_photo_bytes(path: Path) -> io.BytesIO:
    """Handle prepare photo bytes."""
    size = path.stat().st_size
    img = Image.open(path)
    w, h = img.size
    ratio = max(w, h) / max(1, min(w, h))
    if size <= _PHOTO_MAX_SIZE and w + h <= _PHOTO_MAX_DIMENSION_SUM and ratio <= _PHOTO_MAX_RATIO:
        return file_bytes(path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    dimension_scale = min(1, _PHOTO_MAX_DIMENSION_SUM / (w + h))
    for scale in [dimension_scale, dimension_scale * 0.75, dimension_scale * 0.5, dimension_scale * 0.35, dimension_scale * 0.25]:
        if scale <= 0:
            continue
        resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        rw, rh = resized.size
        resized_ratio = max(rw, rh) / max(1, min(rw, rh))
        if buf.tell() <= _PHOTO_MAX_SIZE and rw + rh <= _PHOTO_MAX_DIMENSION_SUM and resized_ratio <= _PHOTO_MAX_RATIO:
            buf.seek(0)
            buf.name = path.stem + ".jpg"
            return buf
    raise ValueError(f"{path.name} could not be resized within Telegram's photo limits")


def _google_link(path: Path) -> str | None:
    """Internal helper to handle google link."""
    template = _GOOGLE_LINK_MAP.get(path.suffix.lower())
    if not template:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        doc_id = data.get("doc_id") or data.get("url", "").split("/d/")[-1].split("/")[0]
        return template.format(doc_id=doc_id) if doc_id else None
    except Exception as e:
        logger.warning(f"Could not read Google proxy file {path.name}: {e}")
        return None


def _classify(path: Path) -> str:
    """Internal helper to handle classify."""
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
    from plugins.services.helpers.parser_registry import get_modality
    try:
        return "text" if get_modality(ext) == "text" and path.stat().st_size <= _INLINE_TEXT_MAX else "document"
    except OSError:
        return "document"


def prepare_media_actions(paths: list[str], max_file_size: int = 50 * 1024 * 1024) -> list[SendAction]:
    """Handle prepare media actions."""
    photo_video, audio, documents, text_actions, skipped = [], [], [], [], []
    for path_str in paths:
        p = Path(path_str)
        if not p.exists() or not p.is_file():
            continue
        try:
            size = p.stat().st_size
            if size > max_file_size:
                skipped.append(f"{p.name} ({size / 1024 / 1024:.1f} MB exceeds 50 MB limit)")
                continue
        except OSError:
            continue
        category = _classify(p)
        if category == "google_link":
            url = _google_link(p)
            text_actions.append(SendAction("text", text_content=f'<a href="{html.escape(url)}">{html.escape(p.name)}</a>')) if url else skipped.append(f"{p.name} (could not extract Google link)")
        elif category in {"photo", "video"}:
            photo_video.append(p)
        elif category == "audio":
            audio.append(p)
        elif category == "text":
            try:
                escaped = html.escape(p.read_text(encoding="utf-8", errors="replace"))
                header, footer = f"<b>{html.escape(p.name)}</b>\n<pre>", "</pre>"
                available = 4096 - len(header) - len(footer)
                text_actions.append(SendAction("text", text_content=header + (escaped[:available - 20] + "\n... (truncated)" if len(escaped) > available else escaped) + footer))
            except Exception:
                documents.append(p)
        else:
            documents.append(p)
    actions = _build_group_actions(photo_video, "photo_video") + _build_group_actions(audio, "audio") + _build_group_actions(documents, "document") + text_actions
    if skipped:
        actions.append(SendAction("text", text_content="Skipped files:\n" + "\n".join(f"- {s}" for s in skipped)))
    return actions


def _build_group_actions(files: list[Path], group_type: str) -> list[SendAction]:
    """Internal helper to build group actions."""
    if not files:
        return []
    actions = []
    for i in range(0, len(files), _MEDIA_GROUP_MAX):
        chunk = files[i:i + _MEDIA_GROUP_MAX]
        if len(chunk) == 1:
            p = chunk[0]
            actions.append(SendAction("video" if group_type == "photo_video" and p.suffix.lower() in VIDEO_EXTENSIONS else "photo" if group_type == "photo_video" else "audio" if group_type == "audio" else "document", [p]))
        else:
            actions.append(SendAction("media_group", chunk, group_type))
    return actions
