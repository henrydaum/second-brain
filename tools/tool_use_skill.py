"""Load a skill's full instructions into context.

Counterpart to the skills service's prompt index: the index carries only
name + description per skill; this tool returns the whole SKILL.md when the
agent decides a skill applies.
"""

dependencies_files = ['services/service_skills.py']
dependencies_pip = []

from plugins.BaseTool import BaseTool, ToolResult


class UseSkill(BaseTool):
    """Fetch one skill's full SKILL.md text."""
    name = "use_skill"
    description = (
        "Load the full instructions of an installed skill by name. Call this as soon "
        "as a request matches a skill in the Skills index, before attempting the task."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name exactly as listed in the Skills index."},
        },
        "required": ["name"],
    }
    requires_services = ["skills"]
    max_calls = 5
    background_safe = True

    def run(self, context, **kwargs) -> ToolResult:
        """Return the skill's full text."""
        service = (context.services or {}).get("skills")
        if service is None or not getattr(service, "loaded", False):
            return ToolResult.failed("Skills service is not loaded.")
        name = (kwargs.get("name") or "").strip()
        text = service.read(name)
        if text is None:
            known = ", ".join(s.name for s in service.list()) or "(none installed)"
            return ToolResult.failed(f"No skill named '{name}'. Installed skills: {known}")
        return ToolResult(llm_summary=text)
