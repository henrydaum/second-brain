"""read_skill_guide: return the long-form generative-art guide.

The web demo agent calls this once per session before authoring new skills.
The content lives in plugins/helpers/skill_guide.md so it can be iterated on
without touching code.
"""

from __future__ import annotations

from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult


_GUIDE_PATH = Path(__file__).resolve().parents[1] / "helpers" / "skill_guide.md"


class ReadSkillGuide(BaseTool):
    name = "read_skill_guide"
    description = (
        "Return the canvas-skill authoring guide — the taste half of the "
        "reference. The system prompt has the formulas (fractals, L-systems, "
        "attractors, noise) and the sandbox rules; this guide has style, "
        "composition, palette discipline, and the chaining strategy that makes "
        "an image feel finished. Call once per session before your first "
        "create_skill if you need a refresher; skip if you're cloning a known "
        "skill via read_skill."
    )
    max_calls = 2
    background_safe = True
    parameters = {"type": "object", "properties": {}, "required": []}

    def run(self, context, **kwargs) -> ToolResult:
        try:
            text = _GUIDE_PATH.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"could not read skill guide: {e}")
        return ToolResult(data={"length": len(text)}, llm_summary=text)
