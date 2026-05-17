"""Gallery / share helpers. Canvas state lives in layered_canvas now."""

from __future__ import annotations

from array import array
import json
import math
import re
import shutil
import time
from pathlib import Path

from PIL import Image

from paths import DATA_DIR
from plugins.tools.helpers.color_theory import visual_stats
# Re-export canvas/state helpers from layered_canvas for back-compat.
from plugins.tools.helpers.layered_canvas import (  # noqa: F401
    canvas,
    current,
    reset_canvas,
    set_current,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
GALLERY_DIR = DATA_DIR / "shared_fractals"


def clean_text(value, default):
    return re.sub(r"[^A-Za-z0-9 _.-]+", "", str(value or default)).strip()[:80] or default


def is_image(path):
    return Path(path).is_file() and Path(path).suffix.lower() in IMAGE_EXTS


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def mark_original(path, meta):
    sidecar = Path(path).with_suffix(".json")
    data = {**read_json(sidecar), **meta, "original": True}
    write_json(sidecar, data)


def save_share(src, title="untitled", artist="anonymous", chain=None):
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    title, artist = clean_text(title, "untitled"), clean_text(artist, "anonymous")
    dest = GALLERY_DIR / f"{clean_text(title, 'untitled').replace(' ', '_')}-{Path(src).stem[-15:]}.png"
    shutil.copy2(src, dest)
    meta = {"title": title, "artist": artist, "source_path": str(Path(src).resolve()),
            "path": str(dest.resolve()), "created_at": time.time(),
            "chain": list(chain or [])}
    write_json(dest.with_suffix(".json"), meta)
    return dest, meta


def share_current(session_key, title="untitled", artist="anonymous"):
    item = current(session_key)
    if not item:
        raise ValueError("No canvas to share yet — make something first.")
    # Pull the live replay chain so remixes can attribute scoring back to the
    # skills that produced this image.
    from plugins.tools.helpers.layered_canvas import canvas as _canvas
    snapshot = _canvas(session_key) or {}
    chain = snapshot.get("chain") or []
    return save_share(item["path"], title, artist, chain=chain)


def image_stats(path):
    return visual_stats(Image.open(path))


def gallery_rows(db=None):
    rows = []
    if not GALLERY_DIR.exists():
        return rows
    remix_counts: dict[str, int] = {}
    if db is not None:
        try:
            from plugins.helpers import skill_scoring
            remix_counts = skill_scoring.remix_counts_by_path(db)
        except Exception:
            remix_counts = {}
    for p in GALLERY_DIR.glob("*.png"):
        if is_image(p):
            m = read_json(p.with_suffix(".json"))
            path_str = str(p.resolve())
            rows.append({
                "path": path_str,
                "title": m.get("title") or "untitled",
                "artist": m.get("artist") or "anonymous",
                "created_at": m.get("created_at") or p.stat().st_mtime,
                "remixes": int(remix_counts.get(path_str, 0)),
                "score": 0.0,
            })
    # Most-remixed first, then most-recent. Cold-start (no remixes yet) falls
    # back to recency for everything.
    return sorted(rows, key=lambda r: (r["remixes"], r["created_at"]), reverse=True)


def similar_rows(target, context=None, limit=24):
    rows = _embedded_rows(target, context) or _visual_rows(target)
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:limit]


def image_vector(path, size=12):
    img = Image.open(path).convert("RGB").resize((size, size))
    vals = []
    for r, g, b in img.getdata():
        vals += [r / 255, g / 255, b / 255]
    mag = math.sqrt(sum(v * v for v in vals)) or 1
    return [v / mag for v in vals]


def cosine(a, b):
    ma = math.sqrt(sum(x * x for x in a)) or 1
    mb = math.sqrt(sum(y * y for y in b)) or 1
    return sum(x * y for x, y in zip(a, b)) / (ma * mb)


def _visual_rows(path):
    target = image_vector(path)
    return [{**r, "score": float(cosine(target, image_vector(r["path"])))}
            for r in gallery_rows()
            if Path(r["path"]).resolve() != Path(path).resolve()]


def _embedded_rows(path, context):
    emb = getattr(context, "services", {}).get("image_embedder") if context else None
    db = getattr(context, "db", None)
    if not emb or not getattr(emb, "loaded", False) or not db:
        return []
    try:
        target = list(emb.encode([Image.open(path).convert("RGB")])[0])
        with db.lock:
            rows = db.conn.execute(
                "SELECT path, embedding FROM image_embeddings WHERE path LIKE ?",
                (str(GALLERY_DIR) + "%",),
            ).fetchall()
        out = []
        by_path = {r["path"]: r for r in gallery_rows()}
        for r in rows:
            vec = array("f"); vec.frombytes(r["embedding"])
            if r["path"] in by_path:
                out.append({**by_path[r["path"]], "score": float(cosine(target, vec))})
        return out
    except Exception:
        return []
