"""manage_layers: agent-side control over the canvas chain (delete/move/clear)."""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers.palettes import get_palette
from plugins.skills.helpers.skill_runner import replay_chain
from plugins.tools.helpers import layered_canvas as lc

logger = logging.getLogger("ManageLayers")


class ManageLayers(BaseTool):
    name = "manage_layers"
    description = (
        "Edit the canvas layer chain (max 4 layers: 1 creation + up to 3 "
        "transforms). action=delete removes layer at chain_index (0 is the "
        "creation — deleting it clears the canvas). action=move reorders from "
        "from_index to to_index; layer 0 must stay a creation. action=clear "
        "wipes the canvas entirely. Surviving layers are replayed end-to-end "
        "to rebuild the image. Reach for this when the chain hits the cap and "
        "you need to free a slot, or when a transform belongs in a different "
        "order."
    )
    max_calls = 4
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["delete", "move", "clear"]},
            "chain_index": {"type": "integer", "description": "Target layer index for delete."},
            "from_index": {"type": "integer", "description": "Source layer index for move."},
            "to_index": {"type": "integer", "description": "Destination layer index for move."},
        },
        "required": ["action"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        session_key = getattr(context, "session_key", None) or "local"
        action = str(kwargs.get("action") or "").lower()
        state = lc.get_state(session_key)
        chain = list(state.get("last_chain") or [])
        if action == "clear":
            lc.reset(session_key)
            return ToolResult(data={"canvas": lc.canvas(session_key)}, llm_summary="Cleared the canvas.")
        if action == "delete":
            idx = int(kwargs.get("chain_index", -1))
            if not (0 <= idx < len(chain)):
                return ToolResult.failed(f"chain_index {idx} out of range (chain has {len(chain)} layer(s)).")
            if idx == 0:
                lc.reset(session_key)
                return ToolResult(data={"canvas": lc.canvas(session_key)}, llm_summary="Deleted the creation layer; canvas cleared.")
            try:
                new_state = lc.delete_chain_entry(session_key, idx)
                return _replay_and_commit(context, session_key, new_state, f"manage_layers:delete:{idx}", f"Deleted layer {idx}.")
            except Exception as e:
                logger.exception("manage_layers delete failed")
                return ToolResult.failed(str(e))
        if action == "move":
            fi = int(kwargs.get("from_index", -1))
            ti = int(kwargs.get("to_index", -1))
            if not chain:
                return ToolResult.failed("Chain is empty; nothing to move.")
            if not (0 <= fi < len(chain)) or not (0 <= ti < len(chain)):
                return ToolResult.failed(f"index out of range (chain has {len(chain)} layer(s)).")
            if fi == ti:
                return ToolResult(data={"canvas": lc.canvas(session_key)}, llm_summary="No-op (from_index == to_index).")
            try:
                new_state = lc.move_chain_entry(session_key, fi, ti)
                return _replay_and_commit(context, session_key, new_state, f"manage_layers:move:{fi}->{ti}", f"Moved layer {fi} to position {ti}.")
            except ValueError as e:
                return ToolResult.failed(str(e))
            except Exception as e:
                logger.exception("manage_layers move failed")
                return ToolResult.failed(str(e))
        return ToolResult.failed(f"Unknown action '{action}'. Use delete, move, or clear.")


def _replay_and_commit(context, session_key: str, state: dict, op: str, summary: str) -> ToolResult:
    chain = list(state.get("last_chain") or [])
    if not chain:
        lc.reset(session_key)
        return ToolResult(data={"canvas": lc.canvas(session_key)}, llm_summary=summary + " Canvas cleared.")
    out = lc.image_path(session_key).with_name("_manage_layers.png")
    registry = getattr(context, "skill_registry", None)
    loader = registry.get_record if registry is not None else (lambda _slug: None)
    replay_chain(
        chain,
        palette=get_palette(state.get("palette_id")),
        size=int(state.get("size") or lc.DEFAULT_SIZE),
        output_image_path=out,
        workdir=out.parent,
        skill_loader=loader,
    )
    with Image.open(out) as img:
        lc.commit_image(session_key, img.convert("RGBA"), op, None)
    new_state = lc.get_state(session_key)
    new_state["last_chain"] = chain
    lc.replace_state(session_key, new_state)
    final = lc.canvas(session_key)
    attach = [final["path"]] if final and final.get("path") else []
    return ToolResult(data={"canvas": final, "chain": chain}, llm_summary=summary, attachment_paths=attach)
