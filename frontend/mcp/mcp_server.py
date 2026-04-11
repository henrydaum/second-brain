"""
MCP (Model Context Protocol) server for Second Brain.

Presents Second Brain's tools, system state, and admin commands as a
standardized MCP interface.  Any MCP-compatible host — Claude Desktop,
Cursor, OpenClaw, Hermes Agent — can connect and call tools natively.

Architecture:
    - Tools from ToolRegistry are auto-registered as MCP tools.
    - System state is exposed as MCP resources.
    - Slash commands are funneled through a single ``system_command`` tool.
    - When sandbox plugins are created or removed, connected clients are
      notified via ``tools/list_changed``.

Usage:
    Called from main.pyw at startup:

        from mcp_server import start_mcp_server
        mcp = start_mcp_server(tool_registry, db, config, ...)

    The server runs on a daemon thread using streamable-HTTP transport.
"""

import json
import logging
import threading
from pathlib import Path

from fastmcp import FastMCP, Context
from fastmcp.tools import Tool
from fastmcp.utilities.types import Image, Audio, File
from fastmcp.prompts import Prompt

from Stage_3.system_prompt import build_system_prompt

# Modality → MCP media helper mapping
_MEDIA_MAP = {
    "image": lambda p: Image(path=p),
    "audio": lambda p: Audio(path=p),
}

# Tools that only make sense in the GUI
_SKIP_TOOLS = {"render_files"}

logger = logging.getLogger("MCP")


# ── Tool wrappers ────────────────────────────────────────────────────

# JSON Schema type → Python type name (used in dynamically generated signatures)
_TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def _make_dispatch_fn(tool_registry):
    """Return a dispatch function that calls through the ToolRegistry.

    Returns either a plain string (text-only) or a list of mixed content
    blocks (text + images/audio/files) if the ToolResult contains
    ``gui_display_paths``.  FastMCP automatically serializes each item
    into the appropriate MCP content type, and the host renders them
    natively — images inline in Claude, as attachments in Telegram, etc.
    """
    from Stage_1.registry import get_modality as _get_modality

    def _mcp_approve(command: str, justification: str) -> bool:
        """Approval gate for shell commands via MCP.

        Controlled by the ``mcp_auto_approve_commands`` config flag.
        When False (default), all commands are denied with a log warning.
        """
        if tool_registry.config.get("mcp_auto_approve_commands", False):
            logger.warning(f"MCP: auto-approving command: {command}")
            return True
        logger.warning(
            f"MCP: denied command (mcp_auto_approve_commands is off): {command}"
        )
        return False

    def dispatch(tool_name: str, mcp_context: Context, kwargs: dict):
        result = tool_registry.call(tool_name, mcp_context=mcp_context,
                                    approve_command=_mcp_approve, **kwargs)
        if not result.success:
            return json.dumps({"error": result.error})

        text = result.llm_summary or json.dumps(result.data, default=str)

        # If there are no display paths, return text only
        if not result.gui_display_paths:
            return text

        # Build a mixed content response: text + media blocks
        content = [text]
        for file_path in result.gui_display_paths:
            p = Path(file_path)
            if not p.exists():
                continue
            modality = _get_modality(p.suffix)
            builder = _MEDIA_MAP.get(modality)
            if builder:
                content.append(builder(str(p)))
            else:
                # Anything else (video, PDF, etc.) → generic file embed
                content.append(File(path=str(p)))
        return content

    return dispatch


def _build_typed_handler(tool_name: str, dispatch_fn, parameters: dict, description: str):
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    # Build required params first, then optional, then Context at the end.
    # Context must come last — putting a defaulted param before required
    # params is a SyntaxError.  FastMCP detects Context by type annotation
    # regardless of position.
    required_strs = []
    optional_strs = []
    dict_entries = []
    for prop_name, prop_info in properties.items():
        py_type = _TYPE_MAP.get(prop_info.get("type", "string"), "str")
        if prop_name in required:
            required_strs.append(f"{prop_name}: {py_type}")
        else:
            default = prop_info.get("default")
            optional_strs.append(f"{prop_name}: {py_type} = {repr(default)}")
        dict_entries.append(f"{repr(prop_name)}: {prop_name}")

    param_strs = required_strs + optional_strs + ["__ctx: Context = None"]
    sig = ", ".join(param_strs)
    dict_str = "{" + ", ".join(dict_entries) + "}"

    code = (
        f"def {tool_name}({sig}) -> str | list:\n"
        f"    _kwargs = {dict_str}\n"
        f"    return _dispatch({tool_name!r}, __ctx, {{k: v for k, v in _kwargs.items() if v is not None}})\n"
    )
    ns = {"_dispatch": dispatch_fn, "Context": Context}
    exec(code, ns)
    fn = ns[tool_name]
    fn.__doc__ = description
    return fn


def register_tools_from_registry(mcp: FastMCP, tool_registry):
    """Discover every enabled tool and register it as an MCP tool.

    Each tool gets a dynamically generated handler whose signature
    matches the tool's JSON Schema, so FastMCP validation works.
    """
    dispatch = _make_dispatch_fn(tool_registry)
    for name, tool in tool_registry.tools.items():
        if not tool.agent_enabled:
            continue
        if name in _SKIP_TOOLS:
            logger.debug(f"Skipping GUI-only tool '{name}' for MCP")
            continue
        _register_one_tool(mcp, tool_registry, name, tool, dispatch)


def _register_one_tool(mcp: FastMCP, tool_registry, name: str, tool, dispatch=None):
    """Register a single Second Brain tool as an MCP tool."""
    if dispatch is None:
        dispatch = _make_dispatch_fn(tool_registry)
    handler = _build_typed_handler(name, dispatch, tool.parameters, tool.description or name)

    mcp_tool = Tool.from_function(
        handler,
        name=name,
        description=tool.description or name,
    )
    mcp.add_tool(mcp_tool)


# ── System command tool ──────────────────────────────────────────────

def register_system_command(mcp: FastMCP, registry):
    """Register a single ``system_command`` tool that dispatches slash
    commands, keeping the tool list clean.

    The external LLM calls:
        system_command(command="load llm")
        system_command(command="stats")
        system_command(command="services")
    """

    help_text = "\n".join(
        f"  {cmd.name:20}"
        for cmd in registry.all_commands()
    )

    @mcp.tool(
        name="system_command",
        description=(
            "Run a Second Brain system command. These manage services, tasks, "
            "tools, config, and the processing pipeline.\n\n"
            "Available commands:\n" + help_text + "\n\n"
            "Type '/help' for more information about each command."
            "Examples:\n"
            '  system_command(command="services")\n'
            '  system_command(command="load llm")\n'
        ),
    )
    def system_command(command: str) -> str:
        """Run a system administration command."""
        command = command.strip().lstrip("/")
        parts = command.split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        result = registry.dispatch(cmd_name, arg)
        return result or "(no output)"


# ── Resources ────────────────────────────────────────────────────────

def register_resources(mcp: FastMCP, ctrl, db):
    """Expose system state as MCP resources.

    Resources are read-only context the host can pull without the LLM
    making a tool call — useful for populating context windows.
    """

    @mcp.resource(
        "secondbrain://stats",
        name="System Statistics",
        description="File counts by modality and task pipeline status.",
        mime_type="application/json",
    )
    def resource_stats() -> str:
        return json.dumps(ctrl.stats(), indent=2, default=str)

    @mcp.resource(
        "secondbrain://services",
        name="Services",
        description="All registered services and their load status.",
        mime_type="application/json",
    )
    def resource_services() -> str:
        return json.dumps(ctrl.list_services(), indent=2, default=str)

    @mcp.resource(
        "secondbrain://tasks",
        name="Tasks",
        description="All registered tasks and their status.",
        mime_type="application/json",
    )
    def resource_tasks() -> str:
        return json.dumps(ctrl.list_tasks(), indent=2, default=str)

    @mcp.resource(
        "secondbrain://tools",
        name="Tools",
        description="All registered tools with descriptions and parameters.",
        mime_type="application/json",
    )
    def resource_tools() -> str:
        return json.dumps(ctrl.list_tools(), indent=2, default=str)

    @mcp.resource(
        "secondbrain://pipeline",
        name="Pipeline Status",
        description="Task pipeline status with dependency graph and queue counts.",
        mime_type="application/json",
    )
    def resource_pipeline() -> str:
        return json.dumps(ctrl.pipeline_status(), indent=2, default=str)

    @mcp.resource(
        "secondbrain://schema/{table_name}",
        name="Table Schema",
        description="Database table schema. Use to understand column types before writing SQL queries.",
        mime_type="application/json",
    )
    def resource_schema(table_name: str) -> str:
        try:
            result = db.query(f"PRAGMA table_info({table_name})")
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource(
        "secondbrain://tables",
        name="Database Tables",
        description="List of all tables in the database.",
        mime_type="application/json",
    )
    def resource_tables() -> str:
        try:
            result = db.query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            names = [row[0] for row in result["rows"]]
            return json.dumps(names, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ── Dynamic tool registration hook ──────────────────────────────────

def hook_tool_registry(mcp: FastMCP, tool_registry):
    """Monkey-patch ToolRegistry.register() and .unregister() so that
    new sandbox tools created via build_plugin are automatically
    reflected in the MCP server, and connected clients get notified.
    """
    _orig_register = tool_registry.register
    _orig_unregister = tool_registry.unregister

    def _patched_register(tool):
        _orig_register(tool)
        if tool.agent_enabled:
            try:
                # Remove first to prevent FastMCP duplicate tool conflicts.
                try:
                    mcp.remove_tool(tool.name)
                except Exception:
                    pass
                _register_one_tool(mcp, tool_registry, tool.name, tool)
                logger.info(f"MCP: auto-registered tool '{tool.name}'")
            except Exception as e:
                logger.debug(f"MCP: failed to auto-register '{tool.name}': {e}")

    def _patched_unregister(name):
        _orig_unregister(name)
        try:
            mcp.remove_tool(name)
            logger.info(f"MCP: auto-unregistered tool '{name}'")
        except Exception as e:
            logger.debug(f"MCP: failed to auto-unregister '{name}': {e}")

    tool_registry.register = _patched_register
    tool_registry.unregister = _patched_unregister


# ── System Prompt ─────────────────────────────────────────────────

def register_prompts(mcp: FastMCP, db, orchestrator, tool_registry, services):
    @mcp.prompt(
        name="second_brain_identity",
        description="Load the Second Brain persona, authoring guidance, and rules."
    )
    def prompt_second_brain() -> str:
        return build_system_prompt(db, orchestrator, tool_registry, services)


# ── Server lifecycle ─────────────────────────────────────────────────

def start_mcp_server(tool_registry, db, config, services, orchestrator,
                     ctrl=None, root_dir=None, command_registry=None) -> FastMCP:
    """
    Build and start the MCP server on a daemon thread.

    Returns the FastMCP instance (for shutdown or introspection).

    Parameters:
        tool_registry:    ToolRegistry with all discovered tools.
        db:               Database instance.
        config:           Global config dict.
        services:         Dict of {name: service_instance}.
        orchestrator:     Orchestrator instance.
        ctrl:             Controller instance (needed for resources + admin commands).
        root_dir:         Project root (needed for /reload).
        command_registry: CommandRegistry with slash commands registered.
    """
    port = config.get("mcp_port", 5123)

    mcp = FastMCP("Second Brain")

    # 1. Register all tools from the ToolRegistry
    register_tools_from_registry(mcp, tool_registry)

    # 2. Register the unified system_command tool
    if command_registry:
        register_system_command(mcp, command_registry)

    # 3. Register read-only resources
    if ctrl:
        register_resources(mcp, ctrl, db)

    # 4. Register system prompt
    register_prompts(mcp, db, orchestrator, tool_registry, services)

    # 5. Hook tool registry for dynamic notifications
    hook_tool_registry(mcp, tool_registry)

    # 7. Start on daemon thread
    def _run():
        try:
            mcp.run(transport="streamable-http", host="127.0.0.1", port=port)
        except Exception as e:
            logger.error(f"MCP server failed: {e}")

    thread = threading.Thread(target=_run, daemon=True, name="MCP-Server")
    thread.start()
    logger.info(f"MCP server listening on http://127.0.0.1:{port}/mcp")
    return mcp
