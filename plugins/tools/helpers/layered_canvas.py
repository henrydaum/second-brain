"""Single-image canvas state.

A canvas per session holds: the current image path, a centralized palette id,
a square size, a structured replay chain (skills run in order), and a
short history list for the UI. Skills inherit `palette` and `size` from
this state at execution time.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from paths import DATA_DIR
from plugins.helpers.palettes import DEFAULT_PALETTE_ID, palette_exists

CANVAS_ROOT = DATA_DIR / "canvas"
STATE_PATH = CANVAS_ROOT / "state.json"
COMPOSITE_DIR = CANVAS_ROOT / "composites"
DEFAULT_SIZE = 768
MIN_SIZE = 256
MAX_SIZE = 1536

_state_lock = threading.RLock()


def _slug(session_key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (session_key or "").lower()).strip("_") or "anon"


def _read_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(data: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _blank_entry() -> dict:
    return {
        "size": DEFAULT_SIZE,
        "palette_id": DEFAULT_PALETTE_ID,
        "image_path": None,
        "last_chain": [],
        "history": [],
    }


def _ensure_dir(session_key: str) -> Path:
    composite_dir = COMPOSITE_DIR / _slug(session_key)
    composite_dir.mkdir(parents=True, exist_ok=True)
    return composite_dir


def _push_history(entry: dict, op: str) -> None:
    hist = (entry.get("history") or [])[-24:]
    hist.append({"op": op, "at": time.time()})
    entry["history"] = hist


def get_state(session_key: str) -> dict:
    """Return a snapshot of the canvas state for this session."""
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        # Backfill any missing keys (forward-compat after schema bumps).
        merged = {**_blank_entry(), **entry}
        return merged


def image_path(session_key: str) -> Path:
    """Stable path for the current composite image."""
    return _ensure_dir(session_key) / "current.png"


def commit_image(session_key: str, pil_image, op: str, chain_entry: dict | None = None) -> dict:
    """Save a PIL image as the new composite. If chain_entry is given, update the replay chain.

    chain_entry is a dict like {slug, params, kind, seed}. For 'creation' kind the chain is
    reset and replaced with [entry]; for 'transform' the entry is appended. For non-skill
    operations (e.g. user wipes), pass chain_entry=None to leave the chain untouched.
    """
    dest = image_path(session_key)
    pil_image.save(dest, format="PNG")
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry = {**_blank_entry(), **entry}
        entry["image_path"] = str(dest.resolve())
        if chain_entry is not None:
            kind = chain_entry.get("kind")
            if kind == "creation":
                entry["last_chain"] = [dict(chain_entry)]
            elif kind == "transform":
                # Don't append a transform if there's no base.
                if entry["last_chain"]:
                    entry["last_chain"] = list(entry["last_chain"]) + [dict(chain_entry)]
                else:
                    entry["last_chain"] = [dict(chain_entry)]
        _push_history(entry, op)
        store[session_key] = entry
        _write_state(store)
        return entry


def set_palette(session_key: str, palette_id: str) -> dict:
    """Update the session-default palette id and propagate it onto every chain
    entry that declared a palette control. Returns the new state."""
    if not palette_exists(palette_id):
        raise ValueError(f"unknown palette: {palette_id}")
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry = {**_blank_entry(), **entry}
        entry["palette_id"] = palette_id
        chain = list(entry.get("last_chain") or [])
        for step in chain:
            if "palette" in (step.get("controls") or {}):
                step["controls"]["palette"] = palette_id
        entry["last_chain"] = chain
        store[session_key] = entry
        _write_state(store)
        return entry


def set_skill_control(session_key: str, chain_index: int, name: str, value) -> dict:
    """Update one control value on a chain entry. Returns the new state.
    Caller is responsible for replaying the chain afterward."""
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry = {**_blank_entry(), **entry}
        chain = list(entry.get("last_chain") or [])
        if not (0 <= chain_index < len(chain)):
            raise ValueError(f"chain_index {chain_index} out of range (len={len(chain)})")
        step = dict(chain[chain_index])
        controls = dict(step.get("controls") or {})
        controls[name] = value
        step["controls"] = controls
        chain[chain_index] = step
        entry["last_chain"] = chain
        # Mirror palette selection onto the session default so future skills inherit.
        if name == "palette" and isinstance(value, str) and palette_exists(value):
            entry["palette_id"] = value
        store[session_key] = entry
        _write_state(store)
        return entry


def randomize_seed(session_key: str, chain_index: int, new_seed: int) -> dict:
    """Replace the seed on one chain entry."""
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry = {**_blank_entry(), **entry}
        chain = list(entry.get("last_chain") or [])
        if not (0 <= chain_index < len(chain)):
            raise ValueError(f"chain_index {chain_index} out of range (len={len(chain)})")
        step = dict(chain[chain_index])
        step["seed"] = int(new_seed)
        chain[chain_index] = step
        entry["last_chain"] = chain
        store[session_key] = entry
        _write_state(store)
        return entry


def replace_state(session_key: str, entry: dict) -> None:
    with _state_lock:
        store = _read_state()
        store[session_key] = {**_blank_entry(), **(entry or {})}
        _write_state(store)


def set_size(session_key: str, size: int) -> dict:
    """Update the centralized square resolution. Clamped to [MIN_SIZE, MAX_SIZE]."""
    size = max(MIN_SIZE, min(MAX_SIZE, int(size)))
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry = {**_blank_entry(), **entry}
        entry["size"] = size
        store[session_key] = entry
        _write_state(store)
        return entry


def clear_chain(session_key: str) -> None:
    """Drop the replay chain without touching the image (used after a manual override)."""
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key)
        if not entry:
            return
        entry["last_chain"] = []
        store[session_key] = entry
        _write_state(store)


def reset(session_key: str) -> None:
    """Wipe canvas state and on-disk image for a session."""
    with _state_lock:
        store = _read_state()
        store.pop(session_key, None)
        _write_state(store)
        d = COMPOSITE_DIR / _slug(session_key)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def reset_canvas(session_key: str) -> None:
    """Compat alias."""
    reset(session_key)


def canvas(session_key: str) -> dict | None:
    """Frontend-facing canvas snapshot. Returns None when the canvas has no image yet."""
    entry = get_state(session_key)
    ip = entry.get("image_path")
    if not ip or not Path(ip).is_file():
        # Still return palette/size so the UI can show the active palette ring.
        return {
            "path": None,
            "palette_id": entry.get("palette_id") or DEFAULT_PALETTE_ID,
            "size": entry.get("size") or DEFAULT_SIZE,
            "history": entry.get("history") or [],
            "chain": entry.get("last_chain") or [],
        }
    return {
        "path": ip,
        "palette_id": entry.get("palette_id") or DEFAULT_PALETTE_ID,
        "size": entry.get("size") or DEFAULT_SIZE,
        "history": entry.get("history") or [],
        "chain": entry.get("last_chain") or [],
    }


def current(session_key: str) -> dict | None:
    """Compat helper used by share/remix flows."""
    state = canvas(session_key)
    if not state or not state.get("path"):
        return None
    return {"path": state["path"], "original": True, "kind": "composite"}


def set_current(session_key: str, path: Any, original: bool = False, meta: dict | None = None) -> None:
    """Compat shim: load an external image as the current canvas (e.g. from remix)."""
    from PIL import Image
    p = Path(path)
    if not p.is_file():
        return
    reset(session_key)
    img = Image.open(p).convert("RGBA")
    commit_image(session_key, img, op=(meta or {}).get("kind") or "load", chain_entry=None)
