"""Shared fractal-demo state helpers."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from PIL import Image

from paths import DATA_DIR

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
STATE_PATH = DATA_DIR / "fractals" / "current_images.json"
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
    state = read_json(STATE_PATH)
    state[session_key] = {"path": str(Path(path).resolve()), "original": bool(original), **(meta or {})}
    write_json(STATE_PATH, state)


def current(session_key):
    item = read_json(STATE_PATH).get(session_key or "")
    return item if item and is_image(item.get("path")) else None


def mark_original(path, meta):
    sidecar = Path(path).with_suffix(".json")
    data = {**read_json(sidecar), **meta, "original": True}
    write_json(sidecar, data)


def save_share(src, title="untitled", artist="anonymous"):
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    title, artist = clean_text(title, "untitled"), clean_text(artist, "anonymous")
    dest = GALLERY_DIR / f"{clean_text(title, 'untitled').replace(' ', '_')}-{Path(src).stem[-15:]}.png"
    shutil.copy2(src, dest)
    meta = {"title": title, "artist": artist, "source_path": str(Path(src).resolve()), "path": str(dest.resolve())}
    write_json(dest.with_suffix(".json"), meta)
    return dest, meta


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
