"""Canvas state shims + composite PNG disk management.

The authoritative canvas state now lives on ``ConversationState.canvas``
(``state_machine/canvas.py``). This module is the thin bridge between
session keys and live ``Canvas`` instances: it resolves the cs from the
bound runtime, performs the requested pure mutation on ``cs.canvas``,
and emits ``CANVAS_COMMITTED`` after a successful image commit.

Persistence of the chain/palette/size now rides on the existing
state-machine marker (see ``state_machine/serialization.py``); the
legacy ``DATA_DIR/canvas/state.json`` is consumed once per
session_key as a one-time migration.
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
from events.event_bus import bus
from events.event_channels import CANVAS_COMMITTED
from state_machine.canvas import (
    Canvas,
    DEFAULT_SIZE,
    MIN_SIZE,
    MAX_SIZE,
    MAX_CHAIN_LENGTH,
)

CANVAS_ROOT = DATA_DIR / "canvas"
LEGACY_STATE_PATH = CANVAS_ROOT / "state.json"
COMPOSITE_DIR = CANVAS_ROOT / "composites"

_state_lock = threading.RLock()
_runtime_ref: Any = None
# Detached canvases for sessions without a live runtime cs (tests,
# autonomous flows). Lazily migrated from state.json on first access.
_detached: dict[str, Canvas] = {}
_migrated_keys: set[str] = set()


def bind_runtime(runtime: Any) -> None:
    """Hook called once at bootstrap so session lookups find the cs."""
    global _runtime_ref
    _runtime_ref = runtime


def _slug(session_key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (session_key or "").lower()).strip("_") or "anon"


def _ensure_dir(session_key: str) -> Path:
    composite_dir = COMPOSITE_DIR / _slug(session_key)
    composite_dir.mkdir(parents=True, exist_ok=True)
    return composite_dir


def image_path(session_key: str) -> Path:
    """Stable path for the current composite image."""
    return _ensure_dir(session_key) / "current.png"


# ── legacy state.json migration ────────────────────────────────────

def _read_legacy_state() -> dict:
    try:
        return json.loads(LEGACY_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _consume_legacy(session_key: str) -> Canvas | None:
    if session_key in _migrated_keys:
        return None
    _migrated_keys.add(session_key)
    store = _read_legacy_state()
    entry = store.get(session_key)
    if not entry:
        return None
    return Canvas.from_dict(entry)


# ── cs resolution ──────────────────────────────────────────────────

def _resolve_canvas(session_key: str) -> Canvas:
    """Return the live ``Canvas`` for this session, creating/migrating as needed."""
    if _runtime_ref is not None:
        try:
            sessions = getattr(_runtime_ref, "sessions", {}) or {}
            session = sessions.get(session_key)
            if session is None and hasattr(_runtime_ref, "get_session"):
                session = _runtime_ref.get_session(session_key)
            if session is not None and getattr(session, "cs", None) is not None:
                canvas = session.cs.canvas
                # Lazy migration: empty canvas + legacy entry → populate.
                if not canvas.last_chain and canvas.image_path is None:
                    legacy = _consume_legacy(session_key)
                    if legacy is not None:
                        canvas.restore(legacy.to_dict())
                return canvas
        except Exception:
            pass
    # Fallback for tests / detached callers.
    if session_key not in _detached:
        legacy = _consume_legacy(session_key)
        _detached[session_key] = legacy if legacy is not None else Canvas()
    return _detached[session_key]


# ── read API ───────────────────────────────────────────────────────

def get_state(session_key: str) -> dict:
    """Return a dict snapshot of the canvas state for this session."""
    with _state_lock:
        return _resolve_canvas(session_key).to_dict()


def to_frontend_shape(c) -> dict:
    """Convert a ``Canvas`` instance into the dict shape the UI expects."""
    ip = c.image_path
    has_image = bool(ip) and Path(ip).is_file()
    return {
        "path": ip if has_image else None,
        "palette_id": c.palette_id or DEFAULT_PALETTE_ID,
        "size": c.size or DEFAULT_SIZE,
        "history": list(c.history),
        "chain": list(c.last_chain),
    }


def canvas(session_key: str) -> dict | None:
    """Frontend-facing canvas snapshot for the session's live ``cs.canvas``."""
    with _state_lock:
        return to_frontend_shape(_resolve_canvas(session_key))


def current(session_key: str) -> dict | None:
    """Compat helper used by share/remix flows."""
    state = canvas(session_key)
    if not state or not state.get("path"):
        return None
    return {"path": state["path"], "original": True, "kind": "composite"}


# ── write API (operate on cs.canvas) ───────────────────────────────

def commit_image(session_key: str, pil_image, op: str, chain_entry: dict | None = None, *, canvas=None) -> dict:
    """Save a PIL image as the new composite WebP, update the chain on
    ``canvas`` (or the session's live ``cs.canvas`` when no canvas is
    given), emit ``CANVAS_COMMITTED``, and return a dict snapshot.

    Passing ``canvas`` lets callers operate on a draft (swap-on-success):
    the draft's chain/image_path/history are updated, but the live
    ``cs.canvas`` is untouched until the caller assigns the draft."""
    dest = image_path(session_key)
    pil_image.save(dest, format="PNG")
    with _state_lock:
        c = canvas if canvas is not None else _resolve_canvas(session_key)
        c.image_path = str(dest.resolve())
        if chain_entry is not None:
            kind = chain_entry.get("kind")
            if kind == "creation":
                c.last_chain = [dict(chain_entry)]
            elif kind == "transform":
                c.last_chain = list(c.last_chain) + [dict(chain_entry)]
        c.push_history(op)
        snapshot = c.to_dict()
    bus.emit(CANVAS_COMMITTED, {
        "session_key": session_key,
        "image_path": snapshot["image_path"],
        "chain": list(snapshot.get("last_chain") or []),
        "op": op,
    })
    return snapshot


def set_palette(session_key: str, palette_id: str) -> dict:
    if not palette_exists(palette_id):
        raise ValueError(f"unknown palette: {palette_id}")
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.apply_palette(palette_id)
        return c.to_dict()


def set_skill_control(session_key: str, chain_index: int, name: str, value) -> dict:
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.apply_control(chain_index, name, value)
        return c.to_dict()


def delete_chain_entry(session_key: str, chain_index: int) -> dict:
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.delete_entry(chain_index)
        return c.to_dict()


def move_chain_entry(session_key: str, from_index: int, to_index: int) -> dict:
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.move_entry(from_index, to_index)
        return c.to_dict()


def randomize_seed(session_key: str, chain_index: int, new_seed: int) -> dict:
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.randomize_seed_at(chain_index, new_seed)
        return c.to_dict()


def replace_state(session_key: str, entry: dict) -> None:
    """Restore a canvas from a snapshot dict (used for rollback on failure)."""
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.restore(Canvas.from_dict(entry).to_dict())


def set_size(session_key: str, size: int) -> dict:
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.set_size(size)
        return c.to_dict()


def clear_chain(session_key: str) -> None:
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.clear_chain()


def _wipe_composite(session_key: str) -> None:
    """Delete the on-disk composite directory; leave state untouched.

    Used by canvas actions whose draft canvas has already been reset —
    they own state, this owns disk."""
    with _state_lock:
        d = COMPOSITE_DIR / _slug(session_key)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    bus.emit(CANVAS_COMMITTED, {
        "session_key": session_key,
        "image_path": None,
        "chain": [],
        "op": "reset",
    })


def reset(session_key: str) -> None:
    """Wipe canvas state and on-disk image for a session (legacy path
    used by remix and similar flows outside the canvas-action framework)."""
    with _state_lock:
        c = _resolve_canvas(session_key)
        c.reset()
    _wipe_composite(session_key)


def reset_canvas(session_key: str) -> None:
    """Compat alias."""
    reset(session_key)


def set_current(session_key: str, path: Any, original: bool = False, meta: dict | None = None) -> None:
    """Compat shim: load an external image as the current canvas (e.g. from remix)."""
    from PIL import Image
    p = Path(path)
    if not p.is_file():
        return
    reset(session_key)
    img = Image.open(p).convert("RGBA")
    commit_image(session_key, img, op=(meta or {}).get("kind") or "load", chain_entry=None)
