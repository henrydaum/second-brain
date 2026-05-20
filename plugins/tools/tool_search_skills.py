"""read_skill: return the source of a stored skill by slug.

Lets the agent clone-and-adjust an existing skill — a much better starting
point than authoring from scratch when the built-in library already has a
close reference.
"""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult


class SearchSkills(BaseTool):
    name = "search_skills"
    description = (
        "Search for stored canvas skills by slug. Returns a list of matching skills."
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
        registry = getattr(context, "skill_registry", None)
        if registry is None:
            return ToolResult.failed("skill registry not available")
        skills = registry.search_records(slug)
        if not skills:
            return ToolResult.failed(f"No skills found for slug '{slug}'.")
        return ToolResult(data={"skills": skills})
        "always cheaper and better than authoring from scratch."