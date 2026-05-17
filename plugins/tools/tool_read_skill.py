"""read_skill: return the source of a stored skill by slug.

Lets the agent clone-and-adjust an existing skill — a much better starting
point than authoring from scratch when the built-in library already has a
close reference.
"""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult
from plugins.helpers import skill_store


class ReadSkill(BaseTool):
    name = "read_skill"
    description = (
        "Return the full source of a stored canvas skill by slug. Use this to "
        "clone-and-adjust an existing skill instead of writing one from scratch."
    )
    max_calls = 6
    background_safe = True
    parameters = {
        "type": "object",
        "properties": {"slug": {"type": "string"}},
        "required": ["slug"],
    }

    def run(self, context, **kwargs) -> ToolResult:
        slug = str(kwargs.get("slug") or "").strip()
        if not slug:
            return ToolResult.failed("slug is required")
        skill = skill_store.read_skill(slug)
        if skill is None:
            return ToolResult.failed(f"No skill named '{slug}'.")
        header = (
            f"# {skill.name} ({skill.kind}, owner={skill.owner or 'unknown'})\n"
            f"# {skill.description}\n\n"
        )
        return ToolResult(
            data={"slug": skill.slug, "kind": skill.kind, "owner": skill.owner, "name": skill.name},
            llm_summary=header + skill.code,
        )
