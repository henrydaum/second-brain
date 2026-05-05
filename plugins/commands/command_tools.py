from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_tools


class ToolsCommand(BaseCommand):
    name = "tools"
    description = "List registered tools"
    category = "System"

    def run(self, _args, context):
        registry = getattr(context, "tool_registry", None)
        schemas = [tool.to_schema()["function"] for tool in registry.tools.values()] if registry else []
        return format_tools([{
            "name": s["name"],
            "description": s.get("description", ""),
            "parameters": s.get("parameters", {}),
            "requires_services": getattr(registry.tools.get(s["name"]), "requires_services", []),
        } for s in schemas])
