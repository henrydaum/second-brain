"""Helpers for technique control schemas (the per-technique UI knobs).

Lives next to the technique registry rather than in any frontend module so
state-machine actions can import it without reaching upward through the
plugin layers.
"""

from __future__ import annotations

from typing import Any


def coerce_control_value(spec: dict, value: Any) -> Any:
    """Coerce an incoming control value to match its declared type."""
    t = spec.get("type")
    if t == "slider":
        v = float(value)
        lo, hi = float(spec.get("min", v)), float(spec.get("max", v))
        return max(lo, min(hi, v))
    if t == "bool":
        return bool(value)
    if t == "enum":
        allowed = [opt.get("value") for opt in (spec.get("options") or [])]
        if value not in allowed:
            raise ValueError(f"enum value {value!r} not in allowed options")
        return value
    if t == "palette":
        return str(value)
    if t == "text":
        s = "" if value is None else str(value)
        cap = int(spec.get("max_length", 120))
        return s[:cap]
    # pan controls deliver values via their underlying numeric param names; this
    # path only fires when the frontend sends {name: x_param, value: number}.
    return value
