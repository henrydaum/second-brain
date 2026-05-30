"""read_technique: return the source of a stored technique by slug.

Lets the agent clone-and-adjust an existing technique — a much better starting
point than authoring from scratch when the built-in library already has a
close reference.
"""

from __future__ import annotations

from plugins.BaseTool import BaseTool, ToolResult


class ReadTechnique(BaseTool):
    name = "read_technique"
    description = (
        "Return the full Python source of a stored canvas technique by slug. The "
        "primary path for reuse: clone the closest hit from search_techniques, "
        "adjust what differs, and write the variant via create_technique. Almost "
        "always cheaper and better than authoring from scratch."
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
        registry = getattr(context, "technique_registry", None)
        if registry is None:
            return ToolResult.failed("technique registry not available")
        technique = registry.get_record(slug)
        if technique is None:
            return ToolResult.failed(f"No technique named '{slug}'.")
        header = (
            f"# {technique.name} ({technique.kind}, owner={technique.owner or 'unknown'})\n"
            f"# {technique.description}\n\n"
        )
        return ToolResult(
            data={"slug": technique.slug, "kind": technique.kind, "owner": technique.owner, "name": technique.name},
            llm_summary=header + technique.code,
        )
