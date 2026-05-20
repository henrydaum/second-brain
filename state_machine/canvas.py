"""Canvas state as a first-class ConversationState attribute.

Pure data + pure transformations. No frontend, no PIL, no subprocess, no
disk I/O. The state machine owns the chain, palette, size, image path,
and history; rendering (running skills, writing PNGs, base64 caching) is
the renderer's job, invoked from the canvas actions in action.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SIZE = 1024
MIN_SIZE = 256
MAX_SIZE = 1536
MAX_CHAIN_LENGTH = 4
DEFAULT_PALETTE_ID = "japandi"  # mirrored from plugins.helpers.palettes
HISTORY_LIMIT = 24


@dataclass
class Canvas:
    """One canvas: its centralized palette/size, current composite path,
    structured chain of skills (creation + transforms), and short op log."""

    size: int = DEFAULT_SIZE
    palette_id: str = DEFAULT_PALETTE_ID
    image_path: str | None = None
    last_chain: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)

    # ── serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "palette_id": self.palette_id,
            "image_path": self.image_path,
            "last_chain": list(self.last_chain),
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Canvas":
        if not data:
            return cls()
        return cls(
            size=int(data.get("size") or DEFAULT_SIZE),
            palette_id=str(data.get("palette_id") or DEFAULT_PALETTE_ID),
            image_path=data.get("image_path"),
            last_chain=list(data.get("last_chain") or []),
            history=list(data.get("history") or []),
        )

    def snapshot(self) -> dict[str, Any]:
        """Deep-ish copy for rollback on render failure."""
        return self.to_dict()

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore from a snapshot in place (used after a failed mutation)."""
        self.size = int(snap.get("size") or DEFAULT_SIZE)
        self.palette_id = str(snap.get("palette_id") or DEFAULT_PALETTE_ID)
        self.image_path = snap.get("image_path")
        self.last_chain = list(snap.get("last_chain") or [])
        self.history = list(snap.get("history") or [])

    # ── pure mutations ───────────────────────────────────────────────

    def push_history(self, op: str) -> None:
        hist = self.history[-HISTORY_LIMIT:]
        hist.append({"op": op, "at": time.time()})
        self.history = hist

    def apply_palette(self, palette_id: str) -> None:
        """Set the session palette and propagate it onto every chain entry
        that declared a palette control."""
        self.palette_id = palette_id
        for step in self.last_chain:
            if "palette" in (step.get("controls") or {}):
                step["controls"]["palette"] = palette_id

    def apply_control(self, chain_index: int, name: str, value: Any) -> None:
        if not (0 <= chain_index < len(self.last_chain)):
            raise ValueError(f"chain_index {chain_index} out of range (len={len(self.last_chain)})")
        step = dict(self.last_chain[chain_index])
        controls = dict(step.get("controls") or {})
        controls[name] = value
        step["controls"] = controls
        self.last_chain[chain_index] = step
        if name == "palette" and isinstance(value, str):
            self.palette_id = value

    def randomize_seed_at(self, chain_index: int, new_seed: int) -> None:
        if not (0 <= chain_index < len(self.last_chain)):
            raise ValueError(f"chain_index {chain_index} out of range (len={len(self.last_chain)})")
        step = dict(self.last_chain[chain_index])
        step["seed"] = int(new_seed)
        self.last_chain[chain_index] = step

    def delete_entry(self, chain_index: int) -> None:
        if not (0 <= chain_index < len(self.last_chain)):
            raise ValueError(f"chain_index {chain_index} out of range (len={len(self.last_chain)})")
        del self.last_chain[chain_index]

    def move_entry(self, from_index: int, to_index: int) -> None:
        n = len(self.last_chain)
        if not (0 <= from_index < n) or not (0 <= to_index < n):
            raise ValueError(f"index out of range (len={n})")
        step = self.last_chain.pop(from_index)
        self.last_chain.insert(to_index, step)
        if self.last_chain and self.last_chain[0].get("kind") != "creation":
            raise ValueError("layer 0 must be a creation; reorder rejected")

    def push_chain_entry(self, entry: dict) -> None:
        """Append/replace as a creation or transform."""
        kind = entry.get("kind")
        if kind == "creation":
            self.last_chain = [dict(entry)]
        elif kind == "transform":
            self.last_chain = list(self.last_chain) + [dict(entry)]
        else:
            raise ValueError(f"unknown chain entry kind: {kind!r}")

    def reseed_chain(self, fresh_seeds: list[int]) -> None:
        if len(fresh_seeds) != len(self.last_chain):
            raise ValueError("fresh_seeds length must match chain length")
        self.last_chain = [{**step, "seed": int(s)} for step, s in zip(self.last_chain, fresh_seeds)]

    def set_size(self, size: int) -> None:
        self.size = max(MIN_SIZE, min(MAX_SIZE, int(size)))

    def clear_chain(self) -> None:
        self.last_chain = []

    def reset(self) -> None:
        self.image_path = None
        self.last_chain = []
        # palette_id, size, history preserved
