"""
Attachment cache.

Front-door for files that arrive from frontends (e.g. Telegram) rather than
from a user-controlled sync directory. Writes each attachment to
``trees.attachment_cache()`` with a stable, unique filename, then runs LRU
eviction to keep the folder under a configured size cap.

The cache folder is registered as a sync_directory by default (see config_data.py),
so the Stage_2 watcher picks up saved files and drives them through the normal
pipeline (extract_text, chunk, embed, OCR, lexical index).

It sits inside the agent's own tree (``DATA_DIR/workspace/attachments``), which
is what lets the agent work on what it was handed without a dialog per verb —
see :func:`trees.attachment_cache`.

**The sync default draws a provenance line, using folder membership to do it.**
Only inbound files are meant to be indexed; the rest of the workspace is code
the agent wrote and is better served by ``fs.search`` than by embeddings. That
was structurally true while :func:`save` was the only thing that could write
here, and since the move it is a convention — the agent may now write to this
folder, and whatever it leaves would be indexed as though a person had sent it.
Nothing does, so this is a note rather than a bug. If something ever should,
mark provenance on the ``files`` row at insert and have the watcher skip
workspace files it did not record; do not widen the sync to the whole tree,
which would index every ``fs.temp`` staging file (real suffix, so the name
filters miss it) once on the way in and ghost it on the way out.
"""

import logging
import re
import time
from pathlib import Path

import trees

logger = logging.getLogger("AttachmentCache")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_NAME_LEN = 120


def _sanitize(name: str) -> str:
    """Internal helper to handle sanitize."""
    name = _SAFE_NAME.sub("_", name).strip("_") or "attachment"
    if len(name) > _MAX_NAME_LEN:
        stem, _, ext = name.rpartition(".")
        if stem and len(ext) <= 10:
            name = stem[: _MAX_NAME_LEN - len(ext) - 1] + "." + ext
        else:
            name = name[:_MAX_NAME_LEN]
    return name


def save(filename_hint: str, data: bytes, size_cap_gb: float = 2.0) -> Path:
    """
    Write bytes into the attachment cache and return the saved path.

    The filename is `{unix_ts}_{sanitized_hint}` so ordering by name matches
    insertion order and collisions are impossible within the same second.
    Triggers LRU eviction after writing.
    """
    directory = trees.attachment_cache()
    ts = int(time.time())
    safe = _sanitize(filename_hint)
    path = directory / f"{ts}_{safe}"

    n = 1
    while path.exists():
        path = directory / f"{ts}_{n}_{safe}"
        n += 1

    path.write_bytes(data)
    logger.info(f"Saved attachment: {path.name} ({len(data)} bytes)")

    _evict_if_over_cap(int(size_cap_gb * 1024 * 1024 * 1024))
    return path


def _evict_if_over_cap(cap_bytes: int) -> None:
    """Internal helper to handle evict if over cap."""
    directory = trees.attachment_cache()
    if not directory.is_dir():
        return
    entries = [(p, p.stat()) for p in directory.iterdir() if p.is_file()]
    total = sum(st.st_size for _, st in entries)
    if total <= cap_bytes:
        return

    entries.sort(key=lambda e: e[1].st_mtime)
    freed = 0
    for p, st in entries:
        if total - freed <= cap_bytes:
            break
        try:
            p.unlink()
            freed += st.st_size
            logger.info(f"Evicted from attachment cache: {p.name} ({st.st_size} bytes)")
        except OSError as e:
            logger.warning(f"Failed to evict {p}: {e}")

    if freed:
        logger.info(f"Attachment cache eviction freed {freed} bytes")
