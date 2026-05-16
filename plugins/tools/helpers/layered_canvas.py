"""Layered canvas state: 4 image slots + palette/atmosphere passes + composite cache."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from paths import DATA_DIR

CANVAS_ROOT = DATA_DIR / "canvas"
STATE_PATH = CANVAS_ROOT / "state.json"
LAYERS_DIR = CANVAS_ROOT / "layers"
COMPOSITE_DIR = CANVAS_ROOT / "composites"
DEFAULT_SIZE = (960, 720)
IMAGE_SLOTS = ("background", "form", "texture", "accent")
PASS_SLOTS = ("palette", "atmosphere")
ALL_SLOTS = IMAGE_SLOTS + PASS_SLOTS

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
        "size": list(DEFAULT_SIZE),
        "layers": {slot: None for slot in IMAGE_SLOTS},
        "palette": None,
        "atmosphere": None,
        "composite_path": None,
        "history": [],
    }


def _ensure_dirs(session_key: str) -> tuple[Path, Path]:
    slug = _slug(session_key)
    layer_dir = LAYERS_DIR / slug
    composite_dir = COMPOSITE_DIR / slug
    layer_dir.mkdir(parents=True, exist_ok=True)
    composite_dir.mkdir(parents=True, exist_ok=True)
    return layer_dir, composite_dir


def get_state(session_key: str) -> dict:
    """Return a deep-copyable snapshot of the canvas state for this session."""
    with _state_lock:
        store = _read_state()
        return store.get(session_key) or _blank_entry()


def _push_history(entry: dict, op: str) -> None:
    hist = (entry.get("history") or [])[-24:]
    hist.append({"op": op, "at": time.time()})
    entry["history"] = hist


def layer_path(session_key: str, slot: str) -> Path:
    """Stable path for a given image slot in this session."""
    layer_dir, _ = _ensure_dirs(session_key)
    return layer_dir / f"{slot}.png"


def composite_path(session_key: str) -> Path:
    """Stable path for the composited canvas image."""
    _, composite_dir = _ensure_dirs(session_key)
    return composite_dir / "current.png"


def set_image_layer(session_key: str, slot: str, tool: str, params: dict, image_path: str | Path) -> dict:
    """Record that ``slot`` was rendered to ``image_path`` and stamp it into the state."""
    assert slot in IMAGE_SLOTS, f"unknown image slot: {slot}"
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry["layers"][slot] = {
            "path": str(Path(image_path).resolve()),
            "tool": tool,
            "params": dict(params or {}),
            "updated_at": time.time(),
        }
        _push_history(entry, tool)
        store[session_key] = entry
        _write_state(store)
        return entry


def set_pass(session_key: str, name: str, tool: str, params: dict) -> dict:
    """Record a palette or atmosphere pass (not an image, just parameters)."""
    assert name in PASS_SLOTS, f"unknown pass: {name}"
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry[name] = {"tool": tool, "params": dict(params or {}), "updated_at": time.time()}
        _push_history(entry, tool)
        store[session_key] = entry
        _write_state(store)
        return entry


def clear_slot(session_key: str, slot: str) -> dict:
    """Remove a slot's contribution to the canvas."""
    if slot not in ALL_SLOTS:
        raise ValueError(f"unknown layer: {slot}")
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        if slot in IMAGE_SLOTS:
            entry["layers"][slot] = None
            p = layer_path(session_key, slot)
            if p.exists():
                try: p.unlink()
                except Exception: pass
        else:
            entry[slot] = None
        _push_history(entry, f"clear:{slot}")
        store[session_key] = entry
        _write_state(store)
        return entry


def record_composite(session_key: str, path: Path, stats: dict | None = None) -> dict:
    """Record the latest composite path + visual stats."""
    with _state_lock:
        store = _read_state()
        entry = store.get(session_key) or _blank_entry()
        entry["composite_path"] = str(Path(path).resolve())
        if stats is not None:
            entry["stats"] = stats
        store[session_key] = entry
        _write_state(store)
        return entry


def reset(session_key: str) -> None:
    """Wipe canvas state and on-disk layers/composite for a session."""
    with _state_lock:
        store = _read_state()
        store.pop(session_key, None)
        _write_state(store)
        slug = _slug(session_key)
        for d in (LAYERS_DIR / slug, COMPOSITE_DIR / slug):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)


def reset_canvas(session_key: str) -> None:
    """Compat alias used by the web frontend."""
    reset(session_key)


def canvas(session_key: str) -> dict | None:
    """Frontend-facing canvas snapshot (the shape /api/canvas expects)."""
    entry = get_state(session_key)
    cp = entry.get("composite_path")
    if not cp or not Path(cp).is_file():
        return None
    layers_summary = {
        slot: (entry["layers"][slot] or {}).get("params", {}).get("style") or (entry["layers"][slot] or {}).get("params", {}).get("type") or ("set" if entry["layers"][slot] else None)
        for slot in IMAGE_SLOTS
    }
    return {
        "path": cp,
        "original": True,
        "kind": "composite",
        "stats": entry.get("stats") or {},
        "layers": layers_summary,
        "palette": (entry.get("palette") or {}).get("params"),
        "atmosphere": (entry.get("atmosphere") or {}).get("params"),
        "history": entry.get("history") or [],
    }


def current(session_key: str) -> dict | None:
    """Compat alias used by the web frontend for the gallery-share path check."""
    state = canvas(session_key)
    if not state:
        return None
    return {"path": state["path"], "original": True, "kind": "composite"}


def set_current(session_key: str, path: Any, original: bool = False, meta: dict | None = None) -> None:
    """Compat shim for the remix flow: load an external image as the background slot."""
    p = Path(path)
    if not p.is_file():
        return
    reset(session_key)
    # Copy the source into our background slot path so future recomposes own it.
    dest = layer_path(session_key, "background")
    shutil.copy2(p, dest)
    meta = dict(meta or {})
    meta.setdefault("source", str(p.resolve()))
    set_image_layer(session_key, "background", meta.get("kind") or "remix", {"style": "remix", **meta}, dest)


def filled_slots(session_key: str) -> dict[str, str | None]:
    """Return a compact map of which slots are filled and with what."""
    entry = get_state(session_key)
    out: dict[str, str | None] = {}
    for slot in IMAGE_SLOTS:
        data = entry["layers"].get(slot)
        if not data:
            out[slot] = None
        else:
            params = data.get("params") or {}
            out[slot] = params.get("style") or params.get("type") or data.get("tool") or "set"
    for name in PASS_SLOTS:
        data = entry.get(name)
        if not data:
            out[name] = None
        else:
            params = data.get("params") or {}
            out[name] = params.get("scheme") or params.get("style") or "set"
    return out


def layers_summary_text(session_key: str) -> str:
    """One-line text summary of the current layer stack, for the LLM."""
    f = filled_slots(session_key)
    parts = []
    for slot in IMAGE_SLOTS:
        parts.append(f"{slot}={f[slot] or 'empty'}")
    pass_parts = []
    for name in PASS_SLOTS:
        pass_parts.append(f"{name}={f[name] or 'empty'}")
    return "Layers: " + ", ".join(parts) + " | " + ", ".join(pass_parts)
