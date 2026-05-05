"""
Register Plugin tool.

Activates or deactivates a sandbox plugin in the live registry. The agent
authors the file with the general file-editing tools; this tool only
manipulates the registry.

Plugin types: tool, task, service, command, frontend.
"""

import logging
import sys
from pathlib import Path

from plugins.BaseTool import BaseTool, ToolResult
from paths import (
    SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES,
    SANDBOX_COMMANDS, SANDBOX_FRONTENDS,
)

logger = logging.getLogger("RegisterPlugin")

# plugin_type -> (sandbox dir, naming prefix or None, sandbox-namespace template)
_PLUGIN_CONFIG = {
    "tool":     (SANDBOX_TOOLS,     "tool_",     "sandbox_tools_{stem}"),
    "task":     (SANDBOX_TASKS,     "task_",     "sandbox_tasks_{stem}"),
    "service":  (SANDBOX_SERVICES,  None,        "sandbox_services_{stem}"),
    "command":  (SANDBOX_COMMANDS,  "command_",  "sandbox_commands_{stem}"),
    "frontend": (SANDBOX_FRONTENDS, "frontend_", "sandbox_frontends_{stem}"),
}


class RegisterPlugin(BaseTool):
    name = "register_plugin"
    description = (
        "Register or unregister a sandbox plugin in the live registry. "
        "Use action='register' with file_name to load the file from its "
        "sandbox directory and add it to the registry. Use action='unregister' "
        "with plugin_name to remove a plugin from the registry. "
        "File creation, editing, and deletion are done with the general file "
        "tools, not this one. Built-in plugins cannot be (un)registered here."
    )
    parameters = {
        "type": "object",
        "properties": {
            "plugin_type": {
                "type": "string",
                "enum": list(_PLUGIN_CONFIG.keys()),
                "description": "Kind of plugin.",
            },
            "action": {
                "type": "string",
                "enum": ["register", "unregister"],
                "description": "Whether to add the plugin to the registry or remove it.",
            },
            "file_name": {
                "type": "string",
                "description": (
                    "Sandbox file name (e.g. tool_get_weather.py). Required for "
                    "action='register'. Must follow the naming convention for "
                    "the plugin_type."
                ),
            },
            "plugin_name": {
                "type": "string",
                "description": (
                    "Registered name of the plugin. Required for "
                    "action='unregister'."
                ),
            },
        },
        "required": ["plugin_type", "action"],
    }
    requires_services = []
    max_calls = 10
    background_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        plugin_type = kwargs.get("plugin_type", "")
        action = kwargs.get("action", "")

        if plugin_type not in _PLUGIN_CONFIG:
            return ToolResult.failed(
                f"Invalid plugin_type '{plugin_type}'. Must be one of: "
                f"{', '.join(_PLUGIN_CONFIG)}."
            )
        if action not in ("register", "unregister"):
            return ToolResult.failed(
                f"Invalid action '{action}'. Must be 'register' or 'unregister'."
            )

        if action == "register":
            return self._register(plugin_type, kwargs.get("file_name", "").strip(), context)
        return self._unregister(plugin_type, kwargs.get("plugin_name", "").strip(), context)

    def _register(self, plugin_type: str, file_name: str, context) -> ToolResult:
        if not file_name:
            return ToolResult.failed("file_name is required for action='register'.")

        err = _check_naming(plugin_type, file_name)
        if err:
            return ToolResult.failed(err)

        sandbox_dir, _, _ = _PLUGIN_CONFIG[plugin_type]
        sandbox_path = sandbox_dir / file_name
        if not sandbox_path.exists():
            return ToolResult.failed(
                f"'{file_name}' was not found in {sandbox_dir.name}/. "
                f"Create the file first using the file-editing tools."
            )

        from plugins.plugin_discovery import load_single_plugin, get_plugin_settings
        # Re-import: drop any stale cached module so a freshly edited file
        # actually picks up its new source.
        _drop_module(sandbox_path, plugin_type)

        name, error = load_single_plugin(
            plugin_type, sandbox_path,
            tool_registry=context.tool_registry,
            orchestrator=context.orchestrator,
            services=context.services,
            config=context.config,
            command_registry=_command_registry(context),
            frontend_manager=_frontend_manager(context),
        )
        if error:
            return ToolResult.failed(f"Registration failed: {error}")

        try:
            import config.config_manager as cm
            cm.reconcile_plugin_config(context.config, get_plugin_settings())
        except Exception as e:
            logger.warning(f"reconcile_plugin_config failed: {e}")

        return ToolResult(llm_summary=f"Registered {plugin_type} '{name}' from {file_name}.")

    def _unregister(self, plugin_type: str, plugin_name: str, context) -> ToolResult:
        if not plugin_name:
            return ToolResult.failed("plugin_name is required for action='unregister'.")

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


def _check_naming(plugin_type: str, file_name: str) -> str | None:
    if not file_name.endswith(".py"):
        return f"File name must end with .py, got '{file_name}'."
    _, prefix, _ = _PLUGIN_CONFIG[plugin_type]
    if prefix and not file_name.startswith(prefix):
        return f"{plugin_type.title()} files must start with '{prefix}', got '{file_name}'."
    if plugin_type == "service" and file_name.startswith("_"):
        return f"Service files must not start with '_', got '{file_name}'."
    return None


def _drop_module(sandbox_path: Path, plugin_type: str) -> None:
    _, _, ns_template = _PLUGIN_CONFIG[plugin_type]
    sys.modules.pop(ns_template.format(stem=sandbox_path.stem), None)


def _command_registry(context):
    return (
        getattr(context, "command_registry", None)
        or getattr(getattr(context, "runtime", None), "command_registry", None)
    )


def _frontend_manager(context):
    return (
        getattr(context, "frontend_manager", None)
        or getattr(getattr(context, "runtime", None), "frontend_manager", None)
    )
