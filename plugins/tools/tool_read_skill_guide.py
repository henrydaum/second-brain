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
        "Return the canvas-skill authoring guide: the skill template, the canvas/"
        "art_kit API reference, established generative methods (Vogel spirals, "
        "Voronoi, flow fields, L-systems), composition rules, palette discipline, "
        "and determinism rules. Read this once per session before authoring a new "
        "skill."
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
