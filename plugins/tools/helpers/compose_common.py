"""Shared helpers for the compose_* tools."""

from __future__ import annotations

import random

from plugins.BaseTool import ToolResult
from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.compositor import recompose
from plugins.tools.helpers.clip_inspect import format_inspect, inspect


def coerce_seed(value) -> int:
    try:
        v = int(value)
        if 1 <= v <= 2_147_483_647:
            return v
    except Exception:
        pass
    return random.randint(1, 2_147_483_647)


def build_result(context, session_key: str, action_summary: str) -> ToolResult:
    """Recompose, inspect, and produce a tool result."""
    if not session_key:
        return ToolResult.failed("No session key — cannot compose without an active conversation.")
    try:
        composite_path, stats = recompose(session_key)
    except Exception as e:
        return ToolResult.failed(f"Compositor failed: {e}")
    descriptors = inspect(context, composite_path)
    layers_line = lc.layers_summary_text(session_key)
    desc_line = format_inspect(descriptors)
    summary_parts = [action_summary, layers_line]
    if desc_line:
        summary_parts.append(desc_line)
    summary_parts.append(
        f"Visual: brightness={stats.get('brightness')}, contrast={stats.get('contrast')}, "
        f"detail={stats.get('detail')}, guidance={stats.get('guidance')}."
    )
    return ToolResult(
        data={
            "composite_path": str(composite_path),
            "stats": stats,
            "descriptors": descriptors,
            "layers": lc.filled_slots(session_key),
        },
        llm_summary=" ".join(summary_parts),
        attachment_paths=[str(composite_path)],
    )
