"""Canvas state machine — the backbone of the image editing toolkit.

Holds canvases in memory with SQLite persistence. Each canvas is a layer chain
(an ordered list of layer dicts), plus dimensions, a palette, and an undo/redo
stack. The service owns all state and mutations; rendering is handled by
``canvas_render.py``, which walks the chain and calls each layer's script.

Quick start
    Get the session canvas (creates one if needed)::

        state = sdk.services.call("canvas", "get_or_create")
        canvas_id = state["canvas_id"]

    Add a background::

        sdk.services.call("canvas", "add_layer", canvas_id,
                          "canvas_solid", "background",
                          {"color": "#1a1a2e"})

    Add a filter::

        sdk.services.call("canvas", "add_layer", canvas_id,
                          "canvas_blur", "filter", {"radius": 8.0})

    Inspect the chain::

        state = sdk.services.call("canvas", "get_state", canvas_id)
        for layer in state["layers"]:
            print(layer["script"], layer["kind"], layer.get("controls"))

Layer model
    Each layer is a dict::

        {
            "id": "stable-unique-id",     # survives reorders and deletes
            "script": "canvas_blur",      # name of a script in scripts/
            "kind": "filter",             # "background" | "filter" | "object"
            "controls": {"radius": 5.0},  # kwargs passed to the script
        }

    Three kinds, three contracts with ``canvas_render.py``:

    - ``background`` — produces an image from nothing. Only layer 0 may be a
      background. The script receives no input image, just dimensions, seed,
      palette, and controls.
    - ``filter`` — reads the prior layer's output PNG, returns a transformed
      PNG. Lens blurs, color grades, glitch effects, warp transforms.
    - ``object`` — reads the prior layer's output, returns an RGBA image that
      is alpha-composited on top. Text, shapes, sprites, overlays.

    Layer 0 must always be a background. ``move_layer`` enforces this.

Persistence
    Canvas state is written to ``canvas_states`` in SQLite on every mutation.
    Canvases survive restarts and are lazy-loaded on first access.

Undo / redo
    Every mutation snapshots the full canvas state before applying. Up to 50
    undo steps are kept. Redo is cleared on new mutations.

This service never touches pixels. Call ``get_state`` to read the chain, then
run ``canvas_render.py`` (or the ``render_canvas`` tool) to produce a PNG. The
separation keeps the service inspectable and the renderer cacheable.

Exports
    create, get_state, get_or_create, add_layer, remove_layer, move_layer,
    set_control, set_palette, set_dimensions, set_render_seed, clear, undo,
    redo, for_session, list_canvases, delete_canvas, list_palettes
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from guest.bases import BaseService

DEFAULT_SIZE = 1024
MIN_SIZE = 16
MAX_SIZE = 8192
UNDO_LIMIT = 50
DEFAULT_PALETTE = "default"


def _new_id() -> str:
    """Url-safe short id."""
    return secrets.token_urlsafe(8).rstrip("=")


def _clamp_dimension(dim: int) -> int:
    return max(MIN_SIZE, min(MAX_SIZE, int(dim)))


# ── palette catalogue ──────────────────────────────────────────────────────

_PALETTES: dict[str, dict] = {}


def _ensure_palettes(sdk) -> None:
    """Load palettes from the data file if not already loaded."""
    global _PALETTES
    if _PALETTES:
        return
    _PALETTES["default"] = {
        "id": "default",
        "name": "Default",
        "kind": "neutral",
        "colors": {
            "background": "#ffffff",
            "primary": "#1a1a2e",
            "secondary": "#16213e",
            "tertiary": "#0f3460",
            "accent": "#e94560",
        },
    }
    # Load additional palettes from a JSON catalogue alongside this service.
    try:
        import json
        palettes_path = sdk.path.join(
            sdk.paths.get("installed"),
            "services", "helpers", "palettes.json",
        )
        raw = sdk.fs.read(palettes_path)
        for p in json.loads(raw):
            _PALETTES[p["id"]] = p
    except Exception:
        pass


# ── canvas dataclass (pure data, in-memory) ─────────────────────────────────


class Canvas:

    def __init__(
        self,
        canvas_id: str | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        palette_id: str = DEFAULT_PALETTE,
        layers: list[dict] | None = None,
        render_seed: int | None = None,
    ):
        self.canvas_id = canvas_id or _new_id()
        self.width = _clamp_dimension(width)
        self.height = _clamp_dimension(height)
        self.palette_id = palette_id
        self.layers: list[dict] = list(layers or [])
        self.render_seed = render_seed
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

    # ── serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "canvas_id": self.canvas_id,
            "width": self.width,
            "height": self.height,
            "palette_id": self.palette_id,
            "layers": [dict(step) for step in self.layers],
            "render_seed": self.render_seed,
            "undo_stack": list(self.undo_stack),
            "redo_stack": list(self.redo_stack),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Canvas":
        if not data:
            return cls()
        c = cls(
            canvas_id=data.get("canvas_id"),
            width=data.get("width", DEFAULT_SIZE),
            height=data.get("height", DEFAULT_SIZE),
            palette_id=data.get("palette_id", DEFAULT_PALETTE),
            layers=data.get("layers"),
            render_seed=data.get("render_seed"),
        )
        c.undo_stack = list(data.get("undo_stack") or [])
        c.redo_stack = list(data.get("redo_stack") or [])
        # Backfill ids for layers that predate the id field.
        for step in c.layers:
            if isinstance(step, dict) and not step.get("id"):
                step["id"] = _new_id()
        return c

    # ── snapshot ───────────────────────────────────────────────────────

    def _snapshot(self) -> dict:
        return {
            "canvas": {
                "width": self.width,
                "height": self.height,
                "palette_id": self.palette_id,
                "layers": [dict(step) for step in self.layers],
            },
            "render_seed": self.render_seed,
        }

    def _snapshot_state(self) -> dict:
        """Snapshot the canvas shape for undo/redo."""
        return {
            "canvas_id": self.canvas_id,
            "width": self.width,
            "height": self.height,
            "palette_id": self.palette_id,
            "layers": [dict(step) for step in self.layers],
            "render_seed": self.render_seed,
        }

    # ── mutations ──────────────────────────────────────────────────────

    def push_undo(self) -> None:
        self.undo_stack.append(self._snapshot_state())
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack = self.undo_stack[-UNDO_LIMIT:]
        self.redo_stack.clear()

    def apply_palette(self, palette_id: str) -> None:
        self.palette_id = palette_id
        for step in self.layers:
            if "palette" in (step.get("controls") or {}):
                step["controls"]["palette"] = palette_id

    def apply_control(self, chain_index: int, name: str, value: Any) -> None:
        if not (0 <= chain_index < len(self.layers)):
            raise ValueError(
                f"chain_index {chain_index} out of range "
                f"(len={len(self.layers)})"
            )
        step = dict(self.layers[chain_index])
        controls = dict(step.get("controls") or {})
        controls[name] = value
        step["controls"] = controls
        self.layers[chain_index] = step

    def delete_entry(self, chain_index: int) -> None:
        if not (0 <= chain_index < len(self.layers)):
            raise ValueError(
                f"chain_index {chain_index} out of range "
                f"(len={len(self.layers)})"
            )
        del self.layers[chain_index]

    def move_entry(self, from_index: int, to_index: int) -> None:
        n = len(self.layers)
        if not (0 <= from_index < n) or not (0 <= to_index < n):
            raise ValueError(f"index out of range (len={n})")
        if from_index != to_index and (from_index == 0 or to_index == 0):
            raise ValueError(
                "layer 0 must be a background; reorder rejected"
            )
        step = self.layers.pop(from_index)
        self.layers.insert(to_index, step)

    def push_layer(self, entry: dict) -> None:
        """Append a layer, or replace layer 0 if it is a background."""
        kind = entry.get("kind")
        if kind == "background":
            if self.layers:
                self.layers[0] = dict(entry)
            else:
                self.layers = [dict(entry)]
        elif kind in ("filter", "object"):
            self.layers = list(self.layers) + [dict(entry)]
        else:
            raise ValueError(f"unknown layer kind: {kind!r}")

    def set_dimensions(self, width: int, height: int) -> None:
        self.width = _clamp_dimension(width)
        self.height = _clamp_dimension(height)

    def reset(self) -> None:
        self.layers = []
        self.render_seed = None


# ── the service ─────────────────────────────────────────────────────────────


class CanvasService(BaseService):
    name = "canvas"
    description = (
        "Canvas state machine for image manipulation and generative art. "
        "Holds a layer chain per canvas; rendering is handled by scripts."
    )

    exports = [
        "create",
        "get_state",
        "get_or_create",
        "add_layer",
        "remove_layer",
        "move_layer",
        "set_control",
        "set_palette",
        "set_dimensions",
        "set_render_seed",
        "clear",
        "undo",
        "redo",
        "for_session",
        "list_canvases",
        "delete_canvas",
        "list_palettes",
    ]

    # ── lifecycle ──────────────────────────────────────────────────────

    def start(self, sdk):
        self._canvases: dict[str, Canvas] = {}
        self._session_to_canvas: dict[str, str] = {}
        _ensure_palettes(sdk)
        self._ensure_schema(sdk)
        sdk.log("canvas: started")
        return True

    def stop(self, sdk):
        self._canvases = {}
        self._session_to_canvas = {}
        sdk.log("canvas: stopped")

    # ── schema ─────────────────────────────────────────────────────────

    def _ensure_schema(self, sdk):
        sdk.db.define(
            """
            CREATE TABLE IF NOT EXISTS canvas_states (
                canvas_id  TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )

    def _persist(self, sdk, canvas: Canvas) -> None:
        import json

        now = time.time()
        payload = json.dumps(canvas.to_dict(), separators=(",", ":"))
        try:
            sdk.db.write(
                "INSERT INTO canvas_states (canvas_id, state_json, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(canvas_id) DO UPDATE SET "
                "  state_json = excluded.state_json, "
                "  updated_at = excluded.updated_at",
                [canvas.canvas_id, payload, now],
            )
        except Exception:
            sdk.log(f"canvas: persist failed for {canvas.canvas_id}")

    def _load(self, sdk, canvas_id: str) -> Canvas | None:
        import json

        rows = sdk.db.query(
            "SELECT state_json FROM canvas_states WHERE canvas_id = ?",
            [canvas_id],
        )
        if not rows:
            return None
        try:
            data = json.loads(rows[0]["state_json"])
        except (TypeError, ValueError):
            return None
        return Canvas.from_dict(data)

    # ── helpers ────────────────────────────────────────────────────────

    def _get(self, sdk, canvas_id: str) -> Canvas | None:
        """Get an in-memory canvas, lazy-loading from the DB."""
        c = self._canvases.get(canvas_id)
        if c is not None:
            return c
        loaded = self._load(sdk, canvas_id)
        if loaded is not None:
            self._canvases[loaded.canvas_id] = loaded
        return loaded

    # ── exports ────────────────────────────────────────────────────────

    def create(self, sdk, width=DEFAULT_SIZE, height=DEFAULT_SIZE,
               palette_id=DEFAULT_PALETTE, canvas_id=None):
        """Allocate a fresh canvas. Returns the canvas_id."""
        c = Canvas(
            canvas_id=canvas_id,
            width=width,
            height=height,
            palette_id=palette_id,
        )
        self._canvases[c.canvas_id] = c
        self._persist(sdk, c)
        sdk.log(f"canvas: created {c.canvas_id} ({c.width}x{c.height})")
        return c.canvas_id

    def get_state(self, sdk, canvas_id):
        """Return the full canvas state dict, or None."""
        c = self._get(sdk, canvas_id)
        if c is None:
            return None
        return c.to_dict()

    def get_or_create(self, sdk, width=DEFAULT_SIZE, height=DEFAULT_SIZE,
                      palette_id=DEFAULT_PALETTE):
        """Return the session's canvas, creating it if needed."""
        session_key = self._session_key(sdk)
        cid = self._session_to_canvas.get(session_key)
        if cid:
            c = self._get(sdk, cid)
            if c is not None:
                return c.to_dict()
            self._session_to_canvas.pop(session_key, None)
        cid = self.create(sdk, width=width, height=height,
                          palette_id=palette_id)
        self._session_to_canvas[session_key] = cid
        return self._get(sdk, cid).to_dict()

    def for_session(self, sdk, canvas_id=None):
        """Bind the session to a canvas, or return the current binding.

        With no canvas_id, returns the current canvas state dict (or None).
        With a canvas_id, binds the session to that canvas and returns its state.
        """
        session_key = self._session_key(sdk)
        if canvas_id is None:
            cid = self._session_to_canvas.get(session_key)
            if cid is None:
                return None
            c = self._get(sdk, cid)
            return c.to_dict() if c else None
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        self._session_to_canvas[session_key] = canvas_id
        return c.to_dict()

    def add_layer(self, sdk, canvas_id, script, kind, controls=None):
        """Append a layer, or replace the background if kind='background'.

        ``script`` is the name of a script in the scripts/ directory (e.g.
        ``"fractal_flame"``). ``kind`` is ``"background"``, ``"filter"``,
        or ``"object"``. ``controls`` is an optional dict of keyword arguments
        passed to the script when rendering.
        """
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        if kind not in ("background", "filter", "object"):
            raise ValueError(
                f"kind must be 'background', 'filter', or 'object' "
                f"(got {kind!r})"
            )
        if not script or not isinstance(script, str):
            raise ValueError("add_layer requires a 'script' name")
        c.push_undo()
        entry = {
            "id": _new_id(),
            "script": str(script),
            "kind": kind,
            "controls": dict(controls or {}),
        }
        c.push_layer(entry)
        self._persist(sdk, c)
        return c.to_dict()

    def remove_layer(self, sdk, canvas_id, chain_index):
        """Delete the layer at ``chain_index``.

        Deleting layer 0 clears the entire canvas.
        """
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        if not isinstance(chain_index, int):
            raise ValueError("chain_index must be an integer")
        if chain_index < 0 or chain_index >= len(c.layers):
            raise ValueError(
                f"chain_index {chain_index} out of range "
                f"(len={len(c.layers)})"
            )
        c.push_undo()
        if chain_index == 0:
            c.reset()
        else:
            c.delete_entry(chain_index)
        self._persist(sdk, c)
        return c.to_dict()

    def move_layer(self, sdk, canvas_id, from_index, to_index):
        """Reorder layers. Layer 0 must stay a background."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        if not isinstance(from_index, int) or not isinstance(to_index, int):
            raise ValueError("from_index and to_index must be integers")
        c.push_undo()
        c.move_entry(from_index, to_index)
        self._persist(sdk, c)
        return c.to_dict()

    def set_control(self, sdk, canvas_id, chain_index, name, value):
        """Update one control on one layer."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        if not isinstance(chain_index, int):
            raise ValueError("chain_index must be an integer")
        if not name:
            raise ValueError("name is required")
        c.push_undo()
        c.apply_control(chain_index, str(name), value)
        self._persist(sdk, c)
        return c.to_dict()

    def set_palette(self, sdk, canvas_id, palette_id):
        """Change the canvas-wide palette."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        if not palette_id:
            raise ValueError("palette_id is required")
        c.push_undo()
        c.apply_palette(str(palette_id))
        self._persist(sdk, c)
        return c.to_dict()

    def set_dimensions(self, sdk, canvas_id, width, height):
        """Resize the canvas. Each dimension is clamped to 16..8192."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        c.push_undo()
        c.set_dimensions(width, height)
        self._persist(sdk, c)
        return c.to_dict()

    def set_render_seed(self, sdk, canvas_id, seed):
        """Store the seed used for the last render, for cache reuse."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        c.render_seed = int(seed)
        self._persist(sdk, c)
        return c.render_seed

    def clear(self, sdk, canvas_id):
        """Reset the layer chain. Palette and dimensions are preserved."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        c.push_undo()
        c.reset()
        self._persist(sdk, c)
        return c.to_dict()

    def undo(self, sdk, canvas_id):
        """Restore the most recent prior canvas state."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        if not c.undo_stack:
            raise ValueError("nothing to undo")
        c.redo_stack.append(c._snapshot_state())
        snapshot = c.undo_stack.pop()
        c.layers = [dict(step) for step in snapshot["layers"]]
        c.width = snapshot["width"]
        c.height = snapshot["height"]
        c.palette_id = snapshot["palette_id"]
        c.render_seed = snapshot["render_seed"]
        self._persist(sdk, c)
        return c.to_dict()

    def redo(self, sdk, canvas_id):
        """Re-apply the most recently undone canvas state."""
        c = self._get(sdk, canvas_id)
        if c is None:
            raise ValueError(f"unknown canvas: {canvas_id!r}")
        if not c.redo_stack:
            raise ValueError("nothing to redo")
        c.undo_stack.append(c._snapshot_state())
        snapshot = c.redo_stack.pop()
        c.layers = [dict(step) for step in snapshot["layers"]]
        c.width = snapshot["width"]
        c.height = snapshot["height"]
        c.palette_id = snapshot["palette_id"]
        c.render_seed = snapshot["render_seed"]
        self._persist(sdk, c)
        return c.to_dict()

    def list_canvases(self, sdk):
        """Return all known canvas ids, newest-updated first."""
        rows = sdk.db.query(
            "SELECT canvas_id FROM canvas_states ORDER BY updated_at DESC"
        )
        return [r["canvas_id"] for r in rows]

    def delete_canvas(self, sdk, canvas_id):
        """Drop a canvas from memory and persistence."""
        self._canvases.pop(canvas_id, None)
        try:
            sdk.db.write(
                "DELETE FROM canvas_states WHERE canvas_id = ?",
                [canvas_id],
            )
        except Exception:
            pass
        # Unbind any session pointing at this canvas.
        to_unbind = [
            k for k, v in self._session_to_canvas.items()
            if v == canvas_id
        ]
        for k in to_unbind:
            self._session_to_canvas.pop(k, None)
        return True

    def list_palettes(self, sdk):
        """Return the palette catalogue."""
        return list(_PALETTES.values())

    # ── internal ───────────────────────────────────────────────────────

    @staticmethod
    def _session_key(sdk) -> str:
        """Derive a session key from the current session."""
        session = sdk.session.get() or {}
        return session.get("session_key") or "local"