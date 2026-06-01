"""Slash command plugin for `/mcp` — manage Model Context Protocol servers.

Wraps the ``mcp_servers`` config (a JSON dict the ``mcp`` service reads) in a
form so users can add/remove/enable servers and inspect the tools each one
registered, without hand-editing JSON. Mirrors the ``/llm`` command's shape:
a dynamic form, plugin-config persistence, and a live service reload so changes
take effect immediately.
"""

import json

from config import config_manager
from plugins.BaseCommand import BaseCommand
from plugins.services.service_mcp import qualified_tool_name
from state_machine.conversation import FormStep

ACTIONS = ["tools", "toggle", "reconnect", "remove"]


class McpCommand(BaseCommand):
    """Slash-command handler for `/mcp`."""

    name = "mcp"
    description = "Add, remove, enable, or inspect MCP servers and their tools"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        servers = context.config.get("mcp_servers", {}) or {}
        names = [*sorted(servers), "add"]
        steps = [FormStep(
            "server", _intro(context, servers), True,
            enum=names, enum_labels=[_server_label(context, n, servers) for n in names],
        )]
        sel = args.get("server")

        if sel == "add":
            steps += [
                FormStep("new_name",
                         "Name this server. It becomes the tool prefix, e.g. 'github' -> mcp__github__*.",
                         True),
                FormStep("transport", "How does Second Brain reach this server?", True,
                         enum=["stdio", "http"],
                         enum_labels=["Local command (stdio)", "Remote URL (HTTP)"],
                         default="stdio"),
            ]
            if args.get("transport") == "http":
                steps += [
                    FormStep("url", "Server URL, e.g. https://example.com/mcp.", True),
                    FormStep("headers",
                             'Optional headers as JSON, e.g. {"Authorization": "Bearer ..."}. Blank for none.',
                             False, default="", prompt_when_missing=True),
                ]
            else:
                steps += [
                    FormStep("command",
                             "Command that launches the server, e.g. npx, uvx, or python.", True),
                    FormStep("cmd_args",
                             "Arguments, space-separated, e.g. -y @modelcontextprotocol/server-github. "
                             "Blank for none.",
                             False, default="", prompt_when_missing=True),
                    FormStep("env",
                             'Optional environment variables as JSON, e.g. {"GITHUB_TOKEN": "..."}. '
                             "Blank for none.",
                             False, default="", prompt_when_missing=True),
                ]
            return steps

        if sel and sel in servers:
            toggle = "Enable" if servers[sel].get("disabled") else "Disable"
            steps.append(FormStep(
                "action", _describe(context, sel, servers), True,
                enum=ACTIONS, enum_labels=["View tools", toggle, "Reconnect", "Remove"],
            ))
        return steps

    def run(self, args, context):
        """Execute `/mcp` for the active session."""
        servers = context.config.setdefault("mcp_servers", {})
        sel = args.get("server")
        if not sel:
            return _list(context, servers)

        if sel == "add":
            name = (args.get("new_name") or "").strip()
            if not name:
                return "Server name is required."
            if name in servers:
                return f"Server '{name}' already exists. Remove it first, or pick another name."
            spec, err = _build_spec(args)
            if err:
                return err
            servers[name] = spec
            _save(context.config)
            reload_err = _reload_mcp(context)
            return reload_err or f"Added MCP server '{name}'.{_tool_note(context, name)}"

        if sel not in servers:
            return "Unknown server."
        action = args.get("action")

        if action == "remove":
            servers.pop(sel, None)
            _save(context.config)
            reload_err = _reload_mcp(context)
            return reload_err or f"Removed MCP server '{sel}'."

        if action == "toggle":
            servers[sel]["disabled"] = not servers[sel].get("disabled", False)
            state = "disabled" if servers[sel]["disabled"] else "enabled"
            _save(context.config)
            reload_err = _reload_mcp(context)
            return reload_err or f"Server '{sel}' {state}.{_tool_note(context, sel)}"

        if action == "reconnect":
            reload_err = _reload_mcp(context)
            return reload_err or f"Reconnected MCP servers.{_tool_note(context, sel)}"

        if action == "tools":
            return _tools_for(context, sel)

        return f"Unknown action: {action}"


# ── spec building / parsing ──────────────────────────────────────────

def _build_spec(args) -> tuple[dict | None, str | None]:
    """Turn collected form args into an mcp_servers spec. Returns (spec, error)."""
    transport = args.get("transport") or "stdio"
    if transport == "http":
        url = (args.get("url") or "").strip()
        if not url:
            return None, "A URL is required for an HTTP server."
        spec = {"transport": "http", "url": url}
        headers, err = _parse_json_dict(args.get("headers"), "headers")
        if err:
            return None, err
        if headers:
            spec["headers"] = headers
        return spec, None

    command = (args.get("command") or "").strip()
    if not command:
        return None, "A command is required for a stdio server."
    spec = {"command": command}
    cmd_args = _parse_args(args.get("cmd_args"))
    if cmd_args:
        spec["args"] = cmd_args
    env, err = _parse_json_dict(args.get("env"), "env")
    if err:
        return None, err
    if env:
        spec["env"] = env
    return spec, None


def _parse_args(raw) -> list[str]:
    """Parse args as a JSON array or a whitespace-separated string."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            val = json.loads(raw)
            if isinstance(val, list):
                return [str(x) for x in val]
        except json.JSONDecodeError:
            pass
    return raw.split()


def _parse_json_dict(raw, label) -> tuple[dict | None, str | None]:
    """Parse an optional JSON object of string->string. Returns (dict, error)."""
    raw = (raw or "").strip()
    if not raw:
        return {}, None
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Could not parse {label} as JSON: {e}"
    if not isinstance(val, dict):
        return None, f"{label} must be a JSON object."
    return {str(k): str(v) for k, v in val.items()}, None


# ── service reload + persistence ─────────────────────────────────────

def _reload_mcp(context) -> str | None:
    """Reload the mcp service so config changes take effect. Returns an error
    string on failure, else None."""
    svc = (context.services or {}).get("mcp")
    if svc is None:
        return "The mcp service is not registered (is service_mcp.py present?)."
    try:
        if getattr(svc, "loaded", False):
            svc.unload()
        if svc.load() is False:
            return ("Saved, but the mcp service failed to load. "
                    "Is the 'mcp' package installed? Run: pip install mcp")
    except Exception as e:
        return f"Saved, but reloading the mcp service failed: {e}"
    return None


def _save(config):
    """Persist mcp_servers to plugin config so it survives restart."""
    saved = config_manager.load_plugin_config()
    saved["mcp_servers"] = config.get("mcp_servers", {})
    config_manager.save_plugin_config(saved)


# ── display helpers ──────────────────────────────────────────────────

def _intro(context, servers) -> str:
    svc = (context.services or {}).get("mcp")
    state = "loaded" if (svc and getattr(svc, "loaded", False)) else "not loaded"
    return f"Select an MCP server, or add one. (mcp service: {state})"


def _server_label(context, name, servers) -> str:
    if name == "add":
        return "Add a server"
    spec = servers.get(name, {})
    if spec.get("disabled"):
        return f"{name} (disabled)"
    svc = (context.services or {}).get("mcp")
    if svc and getattr(svc, "loaded", False) and hasattr(svc, "is_connected"):
        if svc.is_connected(name):
            return f"{name} ({len(svc.registered_tools(name))} tools)"
        return f"{name} (not connected)"
    return name


def _describe(context, name, servers) -> str:
    spec = servers.get(name, {})
    transport = spec.get("transport") or ("http" if spec.get("url") else "stdio")
    if transport == "http":
        target = spec.get("url", "")
    else:
        target = " ".join([spec.get("command", ""), *(spec.get("args") or [])]).strip()
    status = "—"
    svc = (context.services or {}).get("mcp")
    if spec.get("disabled"):
        status = "disabled"
    elif svc and getattr(svc, "loaded", False) and hasattr(svc, "is_connected"):
        status = (f"connected, {len(svc.registered_tools(name))} tool(s)"
                  if svc.is_connected(name) else "not connected")
    return (f"{name}\nTransport: {transport}\nTarget: {target or '-'}\n"
            f"Status: {status}\n\nWhat do you want to do?")


def _list(context, servers) -> str:
    if not servers:
        return "No MCP servers configured. Run /mcp and choose 'Add a server'."
    lines = ["MCP servers:"]
    lines += [f"- {_server_label(context, name, servers)}" for name in sorted(servers)]
    svc = (context.services or {}).get("mcp")
    if not (svc and getattr(svc, "loaded", False)):
        lines.append("\n(The mcp service is not loaded — add a server or use /services to activate it.)")
    return "\n".join(lines)


def _tools_for(context, server) -> str:
    svc = (context.services or {}).get("mcp")
    if not svc or not getattr(svc, "loaded", False):
        return "The mcp service is not loaded."
    if not hasattr(svc, "registered_tools"):
        return "This mcp service build does not support tool listing."
    names = svc.registered_tools(server)
    if not names:
        return f"Server '{server}' registered no tools (not connected, or it exposes none)."
    return f"Tools from '{server}':\n" + "\n".join(f"- {n}" for n in names)


def _tool_note(context, server) -> str:
    """A trailing ' (N tools)' note when the server is connected, else ''."""
    svc = (context.services or {}).get("mcp")
    if svc and getattr(svc, "loaded", False) and hasattr(svc, "registered_tools"):
        n = len(svc.registered_tools(server))
        if n:
            return f" Registered {n} tool(s); the agent can use them now."
    return ""
