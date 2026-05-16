"""execute_skill: run a canvas skill."""

from __future__ import annotations

import random
import time
import logging
from pathlib import Path

from PIL import Image

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers.palettes import get_palette
from plugins.helpers.skill_runner import SkillRunError, make_chain_entry, run_skill
from plugins.helpers.skill_store import read_skill
from plugins.tools.helpers import layered_canvas as lc

logger = logging.getLogger("SkillTools")


class ExecuteSkill(BaseTool):
    name = "execute_skill"
    description = "Execute a creation or transform skill on the canvas. Creations start a new chain; transforms require an existing canvas."
    max_calls = 6
    parameters = {"type": "object", "properties": {"slug": {"type": "string"}, "params": {"type": "object", "default": {}}}, "required": ["slug"]}

    def run(self, context, **kwargs) -> ToolResult:
        session_key, slug, params = getattr(context, "session_key", None) or "local", str(kwargs.get("slug") or ""), dict(kwargs.get("params") or {})
        skill = read_skill(slug)
        if not skill:
            return ToolResult.failed(f"No skill named '{slug}'.")
        state = lc.get_state(session_key)
        if skill.kind == "transform" and not state.get("image_path"):
            return ToolResult.failed("Transform skills require a current canvas image.")
        seed = int(params.get("seed") or random.randint(1, 2_147_483_647))
        tmp = lc.image_path(session_key).with_name(f"_skill_{slug}_{int(time.time()*1000)}.png")
        try:
            run_skill(skill, params=params, palette=get_palette(state.get("palette_id")), size=int(state.get("size") or lc.DEFAULT_SIZE), seed=seed, input_image_path=Path(state["image_path"]) if state.get("image_path") else None, output_image_path=tmp)
            with Image.open(tmp) as img:
                entry = {**make_chain_entry(skill, params, seed), "palette_id": state.get("palette_id")}
                new_state = lc.commit_image(session_key, img.convert("RGBA"), f"skill:{slug}", entry)
            final = lc.canvas(session_key)["path"]
            return ToolResult(data={"canvas": lc.canvas(session_key), "chain": new_state.get("last_chain")}, llm_summary=f"Executed {skill.kind} skill '{slug}' on the canvas.", attachment_paths=[final])
        except (SkillRunError, Exception) as e:
            logger.exception("execute_skill failed: slug=%s session=%s params=%s", slug, session_key, params)
            return ToolResult.failed(str(e))
