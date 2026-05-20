"""Shareable canvas links.

Every share/save/get-link mints a row in `canvas_shares` keyed by a short
opaque `share_id`. The row owns enough state to replay the canvas: image
path, replay chain, palette id, size, plus the title/artist that identify
the piece.

`kind` is one of:
- 'gallery'   — published to the Shared Gallery (via /api/share)
- 'archive'   — saved to the user's private archive (via /api/save)
- 'ephemeral' — Get-link-only; image lives under DATA_DIR/shared_links/

The unique (kind, image_path) index means we lazily backfill legacy gallery
or archive items: the first "Get link" request for an unmapped path inserts
a row and returns its share_id. Subsequent requests return the same id.
"""

from __future__ import annotations

import io
import json
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import qrcode

from paths import DATA_DIR

SHARED_LINKS_DIR = DATA_DIR / "shared_links"
SHARE_ID_LEN = 8
_SHARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,32}$")


def is_valid_share_id(share_id: str) -> bool:
    return bool(share_id) and bool(_SHARE_ID_RE.match(share_id))


def _new_share_id() -> str:
    # token_urlsafe(6) → 8 base64url chars. Plenty of entropy for our scale.
    return secrets.token_urlsafe(6)[:SHARE_ID_LEN]


def _row_to_dict(row) -> dict:
    d = dict(row)
    chain_raw = d.get("chain_json") or "[]"
    try:
        d["chain"] = json.loads(chain_raw)
    except Exception:
        d["chain"] = []
    return d


def create_share(
    db,
    *,
    kind: str,
    image_path: str,
    title: str | None,
    artist: str | None,
    chain: list | None,
    palette_id: str | None,
    size: int | None,
    owner_id: str | None,
) -> str:
    """Insert a canvas_shares row and return its share_id.

    Idempotent on (kind, image_path): if a row already exists, return its
    existing id. Title/artist/chain on the existing row are left alone —
    the first share wins, subsequent backfills don't clobber metadata.
    """
    if kind not in ("gallery", "archive", "ephemeral"):
        raise ValueError(f"unknown share kind: {kind!r}")
    path_str = str(Path(image_path).resolve())
    with db.lock:
        existing = db.conn.execute(
            "SELECT share_id FROM canvas_shares WHERE kind = ? AND image_path = ?",
            (kind, path_str),
        ).fetchone()
        if existing:
            return existing["share_id"]
        chain_json = json.dumps(list(chain or []))
        # Generate ids until one sticks. Collisions are astronomically rare
        # at 8 base64url chars, but the loop is correct and cheap.
        for _ in range(8):
            sid = _new_share_id()
            try:
                db.conn.execute(
                    """
                    INSERT INTO canvas_shares
                        (share_id, kind, image_path, title, artist, chain_json,
                         palette_id, size, owner_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sid, kind, path_str, title, artist, chain_json,
                     palette_id, int(size) if size else None, owner_id, time.time()),
                )
                db.conn.commit()
                return sid
            except Exception:
                continue
        raise RuntimeError("could not allocate share_id")


def find_or_create_for_path(
    db,
    *,
    kind: str,
    image_path: str,
    title: str | None = None,
    artist: str | None = None,
    chain: list | None = None,
    palette_id: str | None = None,
    size: int | None = None,
    owner_id: str | None = None,
) -> str:
    """Path-keyed lookup — used by `/api/get_link` when the user clicks
    Get-link on a gallery or archive tile that pre-dates this feature."""
    return create_share(
        db, kind=kind, image_path=image_path,
        title=title, artist=artist, chain=chain,
        palette_id=palette_id, size=size, owner_id=owner_id,
    )


def lookup_share(db, share_id: str) -> dict | None:
    if not is_valid_share_id(share_id):
        return None
    with db.lock:
        row = db.conn.execute(
            "SELECT * FROM canvas_shares WHERE share_id = ?",
            (share_id,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def bump_view_count(db, share_id: str) -> None:
    if not is_valid_share_id(share_id):
        return
    with db.lock:
        db.conn.execute(
            "UPDATE canvas_shares SET view_count = view_count + 1 WHERE share_id = ?",
            (share_id,),
        )
        db.conn.commit()


def snapshot_current_canvas(
    src_path: str | Path,
    share_id: str,
) -> Path:
    """Copy the live canvas image into the shared_links dir under a stable
    name. This is what makes an 'ephemeral' (get-link-without-publishing)
    link survive future canvas edits in the same session."""
    SHARED_LINKS_DIR.mkdir(parents=True, exist_ok=True)
    # Mirror the source extension so old PNG composites keep working
    # during the WebP transition; new shares pick up .webp automatically.
    ext = Path(src_path).suffix.lower() or ".webp"
    dest = SHARED_LINKS_DIR / f"{share_id}{ext}"
    shutil.copy2(src_path, dest)
    return dest


def generate_qr_png(url: str, *, box_size: int = 8, border: int = 2) -> bytes:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_share_url(base_url: str, share_id: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/share/{share_id}"


def build_qr_url(base_url: str, share_id: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/share/{share_id}/qr.png"
