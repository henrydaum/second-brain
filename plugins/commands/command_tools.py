from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_tool_result, format_tools
from state_machine.conversationClass import FormStep
from state_machine.forms import schema_to_form_steps


ACTIONS = ["call"]


class ToolsCommand(BaseCommand):
    name = "tools"
    description = "Select a tool, then call it"
    category = "System"

    def form(self, args, context):
        registry = getattr(context, "tool_registry", None)
        tools = getattr(registry, "tools", {}) or {}
        steps = [FormStep("tool_name", "Tool", True, enum=sorted(tools))]
        tool = tools.get(args.get("tool_name"))
        if tool:
            steps.append(FormStep("action", _describe(tool), True, enum=ACTIONS))
        if tool and args.get("action") == "call":
            steps += schema_to_form_steps(tool.to_schema()["function"].get("parameters"))
        return steps

    def run(self, args, context):
        registry = getattr(context, "tool_registry", None)
        if args.get("tool_name"):
            tool = (getattr(registry, "tools", {}) or {}).get(args["tool_name"]) if registry else None
            if not tool:
                return "Unknown tool."
            if args.get("action") == "call":
                fields = tool.to_schema()["function"].get("parameters", {}).get("properties", {}).keys()
                return format_tool_result(registry.call(args["tool_name"], **{k: args[k] for k in fields if k in args}))
            return f"Unknown action: {args.get('action')}"
        schemas = [tool.to_schema()["function"] for tool in registry.tools.values()] if registry else []
        return format_tools([{
            "name": s["name"],
            "description": s.get("description", ""),
            "parameters": s.get("parameters", {}),
            "requires_services": getattr(registry.tools.get(s["name"]), "requires_services", []),
        } for s in schemas])


def _describe(tool):
    schema = tool.to_schema()["function"]
    params = schema.get("parameters", {})
    required = set(params.get("required", []))
    fields = [f"{name}{'*' if name in required else ''}" for name in (params.get("properties") or {})]
    return f"{tool.name}\n{schema.get('description', '')}\nArgs: {', '.join(fields) or '(none)'}"
