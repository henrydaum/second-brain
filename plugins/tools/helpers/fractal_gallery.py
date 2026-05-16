"""Shared fractal-demo state helpers."""

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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
STATE_PATH = DATA_DIR / "fractals" / "current_images.json"
CANVAS_PATH = DATA_DIR / "fractals" / "canvases.json"
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


def set_current(session_key, path, original=False, meta=None):
    if not session_key or not is_image(path):
        return
    path = Path(path).resolve()
    state = read_json(STATE_PATH)
    state[session_key] = {"path": str(path), "original": bool(original), **(meta or {})}
    write_json(STATE_PATH, state)
    canvases = read_json(CANVAS_PATH)
    old = canvases.get(session_key, {})
    hist = (old.get("history") or [])[-24:]
    if old.get("path") != str(path):
        hist = hist[-23:] + [{"path": str(path), "op": (meta or {}).get("kind") or "image", "at": time.time()}]
    canvases[session_key] = {"path": str(path), "original": bool(original), "meta": meta or {}, "stats": image_stats(path), "history": hist}
    write_json(CANVAS_PATH, canvases)


def current(session_key):
    item = read_json(STATE_PATH).get(session_key or "")
    return item if item and is_image(item.get("path")) else None


def canvas(session_key):
    item = read_json(CANVAS_PATH).get(session_key or "")
    return item if item and is_image(item.get("path")) else None


def reset_canvas(session_key):
    for path in (STATE_PATH, CANVAS_PATH):
        data = read_json(path)
        data.pop(session_key or "", None)
        write_json(path, data)


def mark_original(path, meta):
    sidecar = Path(path).with_suffix(".json")
    data = {**read_json(sidecar), **meta, "original": True}
    write_json(sidecar, data)


def save_share(src, title="untitled", artist="anonymous"):
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    title, artist = clean_text(title, "untitled"), clean_text(artist, "anonymous")
    dest = GALLERY_DIR / f"{clean_text(title, 'untitled').replace(' ', '_')}-{Path(src).stem[-15:]}.png"
    shutil.copy2(src, dest)
    meta = {"title": title, "artist": artist, "source_path": str(Path(src).resolve()), "path": str(dest.resolve()), "created_at": time.time()}
    write_json(dest.with_suffix(".json"), meta)
    return dest, meta


def share_current(session_key, title="untitled", artist="anonymous"):
    item = current(session_key)
    if not item or not item.get("original"):
        raise ValueError("Only the current original/generated canvas can be shared.")
    return save_share(item["path"], title, artist)


def image_stats(path):
    return visual_stats(Image.open(path))


def gallery_rows():
    rows = []
    for p in GALLERY_DIR.glob("*.png"):
        if is_image(p):
            m = read_json(p.with_suffix(".json"))
            rows.append({"path": str(p.resolve()), "title": m.get("title") or "untitled", "artist": m.get("artist") or "anonymous", "created_at": m.get("created_at") or p.stat().st_mtime, "score": 0.0})
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


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
    return [{**r, "score": float(cosine(target, image_vector(r["path"])))} for r in gallery_rows() if Path(r["path"]).resolve() != Path(path).resolve()]


def _embedded_rows(path, context):
    emb = getattr(context, "services", {}).get("image_embedder") if context else None
    db = getattr(context, "db", None)
    if not emb or not getattr(emb, "loaded", False) or not db:
        return []
    try:
        target = list(emb.encode([Image.open(path).convert("RGB")])[0])
        with db.lock:
            rows = db.conn.execute("SELECT path, embedding FROM image_embeddings WHERE path LIKE ?", (str(GALLERY_DIR) + "%",)).fetchall()
        out = []
        by_path = {r["path"]: r for r in gallery_rows()}
        for r in rows:
            vec = array("f"); vec.frombytes(r["embedding"])
            if r["path"] in by_path:
                out.append({**by_path[r["path"]], "score": float(cosine(target, vec))})
        return out
    except Exception:
        return []
