"""Live-unregister sandbox plugins without deleting their files."""

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.tool_register_plugin import _PLUGIN_CONFIG, _command_registry, _frontend_manager


class UnregisterPlugin(BaseTool):
    name = "unregister_plugin"
    description = (
        "Remove a sandbox plugin from the live runtime without deleting its file. "
        "Use edit_file(operation='delete', ...) as a separate step if the plugin "
        "should not load again on startup."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plugin_type": {"type": "string", "enum": list(_PLUGIN_CONFIG.keys()), "description": "Kind of plugin."},
            "plugin_name": {"type": "string", "description": "Registered plugin name to remove from the live runtime."},
        },
        "required": ["plugin_type", "plugin_name"],
    }
    requires_services = []
    max_calls = 10
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        plugin_type = kwargs.get("plugin_type", "")
        plugin_name = (kwargs.get("plugin_name") or "").strip()
        if plugin_type not in _PLUGIN_CONFIG:
            return ToolResult.failed(f"Invalid plugin_type '{plugin_type}'. Must be one of: {', '.join(_PLUGIN_CONFIG)}.")
        if not plugin_name:
            return ToolResult.failed("plugin_name is required.")
        missing = _missing_handle(plugin_type, context)
        if missing:
            return ToolResult.failed(missing)
        from plugins.plugin_discovery import unload_plugin
        try:
            unload_plugin(
                plugin_type, plugin_name,
                tool_registry=context.tool_registry,
                orchestrator=context.orchestrator,
                services=context.services,
                command_registry=_command_registry(context),
                frontend_manager=_frontend_manager(context),
            )
        except Exception as e:
            return ToolResult.failed(f"Unregister failed: {e}")
        return ToolResult(llm_summary=f"Unregistered {plugin_type} '{plugin_name}'.")


def _missing_handle(plugin_type: str, context) -> str | None:
    if plugin_type == "tool" and not getattr(context, "tool_registry", None):
        return "No tool registry available."
    if plugin_type == "task" and not getattr(context, "orchestrator", None):
        return "No orchestrator available."
    if plugin_type == "service" and getattr(context, "services", None) is None:
        return "No services registry available."
    if plugin_type == "command" and not _command_registry(context):
        return "No command registry available."
    if plugin_type == "frontend" and not _frontend_manager(context):
        return "No frontend manager available."
    return None
