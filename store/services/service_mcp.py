"""MCP (Model Context Protocol) client service.

Connects to one or more MCP servers (stdio subprocess or streamable-HTTP),
discovers their tools, and registers each discovered tool as a first-class
``BaseTool`` in the tool registry — so the agent sees MCP tools alongside
native ones, with no special-casing in the agent loop.

This is the *client* half of MCP: Second Brain consuming external MCP servers
(GitHub, Postgres, Slack, filesystem, etc.). Exposing Second Brain *as* an MCP
server is a separate, future piece.

Configuration (``mcp_servers``, a JSON dict keyed by server name)::

    {
      "everything": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-everything"],
        "env": {"FOO": "bar"}
      },
      "remote": {
        "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer ..."}
      }
    }

A spec with a ``command`` uses the stdio transport; one with a ``url`` uses
streamable HTTP. Add ``"disabled": true`` to skip a server, or ``"transport"``
to force ``"stdio"``/``"http"`` explicitly.

Async lives behind a single background event-loop thread. Each server is held
open by one long-lived coroutine that sets up the connection, lists tools, then
parks on a stop event — so setup and teardown happen in the same task, which is
what anyio's cancel scopes require. Synchronous callers (tools, ``unload``)
marshal into that loop via ``run_coroutine_threadsafe`` / ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading

from plugins.BaseService import BaseService
from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("MCPService")

# OpenAI caps function names at 64 chars; keep MCP tool names within that.
_MAX_TOOL_NAME = 64
_NAME_SANITIZE = re.compile(r"[^a-zA-Z0-9_-]")


def qualified_tool_name(server: str, tool: str) -> str:
    """Namespace an MCP tool as ``mcp__<server>__<tool>`` (sanitized, capped).

    Mirrors the ``mcp__server__tool`` convention used by other MCP hosts so the
    agent can tell at a glance that a tool came from a server.
    """
    raw = f"mcp__{server}__{tool}"
    safe = _NAME_SANITIZE.sub("_", raw)
    return safe[:_MAX_TOOL_NAME]


def result_to_text(call_result) -> str:
    """Flatten an MCP ``CallToolResult``'s content blocks into model-facing text."""
    parts: list[str] = []
    for block in getattr(call_result, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            if text:
                parts.append(text)
        elif btype == "image":
            parts.append("[image content returned by MCP tool]")
        elif btype == "resource":
            res = getattr(block, "resource", None)
            uri = getattr(res, "uri", "") if res else ""
            parts.append(f"[resource: {uri}]" if uri else "[resource content]")
        else:
            parts.append(f"[{btype or 'unknown'} content]")
    return "\n".join(parts)


class MCPTool(BaseTool):
    """A single MCP server tool, surfaced as a native Second Brain tool.

    Instantiated by ``MCPService`` (never auto-discovered — it needs the live
    service + server name), so ``auto_register`` is False.
    """

    auto_register = False
    max_calls = 5

    def __init__(self, service: "MCPService", server: str, mcp_tool):
        """Build a tool wrapper from an MCP tool definition."""
        self._service = service
        self._server = server
        self._mcp_name = getattr(mcp_tool, "name", "") or ""
        self.name = qualified_tool_name(server, self._mcp_name)
        desc = (getattr(mcp_tool, "description", "") or "").strip()
        self.description = (
            f"[MCP server '{server}'] {desc}"
            if desc
            else f"MCP tool '{self._mcp_name}' from server '{server}'."
        )
        schema = getattr(mcp_tool, "inputSchema", None)
        self.parameters = schema if isinstance(schema, dict) and schema else {
            "type": "object", "properties": {},
        }
        # Refuse to run if the MCP service has been unloaded.
        self.requires_services = ["mcp"]

    def run(self, context, **kwargs) -> ToolResult:
        """Dispatch the call to the MCP server and map its result to a ToolResult."""
        try:
            call_result = self._service.call_mcp_tool(self._server, self._mcp_name, kwargs)
        except TimeoutError:
            return ToolResult.failed(
                f"MCP tool '{self.name}' timed out waiting for server '{self._server}'."
            )
        except Exception as e:  # connection dropped, bad args, server error
            return ToolResult.failed(f"MCP tool '{self.name}' failed: {e}")

        text = result_to_text(call_result)
        if getattr(call_result, "isError", False):
            return ToolResult.failed(text or "MCP tool reported an error.")

        structured = getattr(call_result, "structuredContent", None)
        return ToolResult(
            success=True,
            llm_summary=text or "(MCP tool returned no text content)",
            data=structured if structured is not None else None,
        )


class MCPService(BaseService):
    """Manages MCP server connections and registers their tools."""

    model_name = "mcp"
    shared = True

    config_settings = [
        ("MCP Servers", "mcp_servers",
         "MCP servers to connect to, keyed by name. Each value is "
         "{command, args, env} for a stdio server or {url, headers} for an "
         "HTTP server. Add \"disabled\": true to skip one.",
         {},
         {"type": "json_dict"}),

        ("MCP Tool Timeout", "mcp_tool_timeout",
         "Seconds to wait for an MCP tool call before giving up.",
         60,
         {"type": "slider", "range": (5, 600, 119), "is_float": False}),

        ("MCP Connect Timeout", "mcp_connect_timeout",
         "Seconds to wait for an MCP server to connect and list its tools.",
         30,
         {"type": "slider", "range": (5, 120, 115), "is_float": False}),
    ]

    def __init__(self, config: dict):
        """Initialize the MCP service (does not connect — see _load)."""
        super().__init__()
        self.config = config
        self._tool_registry = None
        self._conv_runtime = None                 # ConversationRuntime, for interactive OAuth
        # Background asyncio loop that owns every MCP session.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # Per-server runtime state.
        self._sessions: dict = {}            # server -> live ClientSession
        self._stops: dict = {}               # server -> asyncio.Event (set to disconnect)
        self._server_futures: dict = {}      # server -> concurrent.futures.Future of _serve
        self._server_tools: dict = {}        # server -> list of MCP tool defs
        self._registered: set[str] = set()   # qualified tool names we put in the registry
        self._tools_registered = False
        self._lock = threading.RLock()

    # ── lifecycle ────────────────────────────────────────────────────

    @property
    def _tool_timeout(self) -> int:
        return int(self.config.get("mcp_tool_timeout", 60) or 60)

    @property
    def _connect_timeout(self) -> int:
        return int(self.config.get("mcp_connect_timeout", 30) or 30)

    def _active_servers(self) -> dict:
        servers = self.config.get("mcp_servers") or {}
        if not isinstance(servers, dict):
            return {}
        return {
            name: spec for name, spec in servers.items()
            if isinstance(spec, dict) and not spec.get("disabled")
        }

    def _load(self) -> bool:
        """Connect to configured servers and (if possible) register their tools.

        Returns True even with zero servers configured — the service is then
        "on but idle", and tools appear after servers are added and the service
        is reloaded. Returns False only if servers are configured but the MCP
        SDK is missing.
        """
        servers = self._active_servers()
        if not servers:
            logger.info("MCP: no servers configured; service idle.")
            self.loaded = True
            return True

        try:
            import mcp  # noqa: F401
        except Exception as e:
            logger.error(
                "MCP servers are configured but the 'mcp' package is not installed. "
                "Run: pip install mcp  (%s)", e,
            )
            return False

        self._start_loop()
        connected = 0
        for name, spec in servers.items():
            if self._is_oauth(spec):
                # OAuth servers may need to prompt the user, which requires the
                # session lock. load() is often called *under* that lock (from a
                # command), so connecting here would deadlock. Schedule it on the
                # loop and return; it self-registers once authorization completes.
                self._schedule_serve(name, spec)
                logger.info("MCP server '%s': connecting in background (OAuth).", name)
            elif self._connect_server(name, spec):
                connected += 1
        logger.info("MCP: connected %d/%d server(s) synchronously.", connected, len(servers))

        self.loaded = True
        # If the registry is already bound (on-demand load after startup), wire
        # tools now. During autoload it isn't yet — bind_runtime() handles it.
        if self._tool_registry is not None:
            self._register_tools()
        return True

    def bind_runtime(self, *, tool_registry=None, runtime=None, orchestrator=None,
                     command_registry=None, frontend_manager=None, **_):
        """Receive the tool registry and runtime. Registers tools if we connected first.

        Called once at startup (after autoload) and again on reload. Either
        ordering — load-then-bind or bind-then-load — ends with tools registered.
        The runtime is needed so an OAuth server can prompt the user in chat.
        """
        if tool_registry is not None:
            self._tool_registry = tool_registry
        if runtime is not None:
            self._conv_runtime = runtime
        if self.loaded and self._server_tools and not self._tools_registered:
            self._register_tools()

    def unload(self):
        """Disconnect every server, stop the loop, and unregister all tools."""
        # 1. Unregister tools first so the agent stops seeing them immediately.
        if self._tool_registry is not None:
            for name in list(self._registered):
                try:
                    self._tool_registry.unregister(name)
                except Exception as e:
                    logger.debug("MCP: failed to unregister '%s': %s", name, e)
        self._registered.clear()
        self._tools_registered = False

        # 2. Signal each serve-coroutine to exit, letting its async-with blocks
        #    unwind in their own task (anyio cancel-scope safety).
        loop = self._loop
        if loop is not None and not loop.is_closed():
            for stop in list(self._stops.values()):
                try:
                    loop.call_soon_threadsafe(stop.set)
                except Exception:
                    pass
            for fut in list(self._server_futures.values()):
                try:
                    fut.result(timeout=5)
                except Exception as e:
                    logger.debug("MCP: serve task did not exit cleanly: %s", e)
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=5)

        self._loop = None
        self._thread = None
        self._sessions.clear()
        self._stops.clear()
        self._server_futures.clear()
        self._server_tools.clear()
        self.loaded = False
        logger.info("MCP: unloaded.")

    # ── event loop plumbing ──────────────────────────────────────────

    def _start_loop(self):
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-loop", daemon=True,
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    def _is_oauth(self, spec: dict) -> bool:
        """Whether a server should use the interactive OAuth flow.

        An HTTP server with no static ``headers`` and ``oauth`` not disabled —
        the SDK's provider only actually triggers a flow on a 401, so attaching
        it is harmless for servers that don't need auth.
        """
        transport = spec.get("transport") or ("http" if spec.get("url") else "stdio")
        if transport != "http" or spec.get("headers"):
            return False
        return spec.get("oauth", True) is not False

    def _schedule_serve(self, name: str, spec: dict):
        """Start the serve coroutine on the loop. Returns ``(ready, state)``."""
        ready = threading.Event()
        state: dict = {"error": None, "tools": []}
        fut = asyncio.run_coroutine_threadsafe(
            self._serve(name, spec, state, ready), self._loop,
        )
        self._server_futures[name] = fut
        return ready, state

    def _connect_server(self, name: str, spec: dict) -> bool:
        """Schedule the serve coroutine and block until it's ready or fails."""
        try:
            ready, state = self._schedule_serve(name, spec)
        except Exception as e:
            logger.warning("MCP server '%s': could not schedule connect: %s", name, e)
            return False

        if not ready.wait(timeout=self._connect_timeout):
            state["error"] = state["error"] or "timed out connecting"
        if state["error"]:
            logger.warning("MCP server '%s' failed to connect: %s", name, state["error"])
            return False

        logger.info("MCP server '%s': discovered %d tool(s).", name, len(state["tools"]))
        return True

    async def _serve(self, name: str, spec: dict, state: dict, ready: threading.Event):
        """Hold one MCP server connection open until its stop event is set.

        Setup (connect, initialize, list_tools) and teardown both run in this
        single task, which keeps anyio's cancel scopes happy.
        """
        try:
            transport = spec.get("transport") or ("http" if spec.get("url") else "stdio")
            stop = asyncio.Event()
            self._stops[name] = stop

            if transport == "stdio":
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client

                env = spec.get("env")
                params = StdioServerParameters(
                    command=spec["command"],
                    args=list(spec.get("args") or []),
                    env={**os.environ, **env} if env else None,
                )
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await self._handshake(name, session, state, ready)
                        await stop.wait()
            else:
                from mcp import ClientSession
                from mcp.client.streamable_http import streamablehttp_client

                url = spec["url"]
                headers = spec.get("headers") or None
                auth = self._oauth_provider(name, spec) if self._is_oauth(spec) else None
                async with streamablehttp_client(url, headers=headers, auth=auth) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await self._handshake(name, session, state, ready)
                        await stop.wait()
        except Exception as e:
            state["error"] = str(e)
            logger.debug("MCP server '%s' serve error: %s", name, e, exc_info=True)
        finally:
            self._sessions.pop(name, None)
            ready.set()  # never leave _connect_server blocked

    async def _handshake(self, name, session, state, ready):
        await session.initialize()
        self._sessions[name] = session
        tools = await self._list_all_tools(session)
        self._server_tools[name] = tools
        state["tools"] = tools
        # Register before signaling ready so a synchronous connect sees the
        # tools immediately; for background (OAuth) servers this is how they
        # register at all once authorization completes.
        self._register_server(name)
        ready.set()

    def _oauth_provider(self, name: str, spec: dict):
        """Build the OAuth provider for a server, or None if unavailable."""
        try:
            from plugins.services.helpers.mcp_oauth import build_oauth_provider
        except Exception as e:
            logger.warning("MCP '%s': OAuth support unavailable (%s); connecting without auth.", name, e)
            return None
        return build_oauth_provider(spec["url"], name, spec.get("scope"), self._oauth_prompt(name))

    def _oauth_prompt(self, name: str):
        """Return a blocking ``prompt(auth_url) -> str | None`` for this server."""
        def prompt(auth_url: str) -> str | None:
            from runtime.interactive_auth import authorize_via_frontend
            runtime = self._conv_runtime
            session_key = getattr(runtime, "active_session_key", None) if runtime else None
            if not runtime or not session_key:
                logger.error(
                    "MCP '%s' needs authorization but no conversation is active to ask in. "
                    "Open a chat and run /mcp -> reconnect.", name,
                )
                return None
            return authorize_via_frontend(
                runtime, session_key, auth_url,
                title=f"Authorize MCP server '{name}'",
                instructions=(
                    "Open this link, sign in, and approve access. You'll land on a page that "
                    "won't load — copy the full URL from your browser's address bar (or just "
                    "the code) and send it back here:"
                ),
                timeout=float(self.config.get("mcp_oauth_timeout", 600) or 600),
            )
        return prompt

    @staticmethod
    async def _list_all_tools(session) -> list:
        """List every tool, following pagination cursors."""
        tools: list = []
        cursor = None
        while True:
            result = await session.list_tools(cursor) if cursor else await session.list_tools()
            tools.extend(getattr(result, "tools", []) or [])
            cursor = getattr(result, "nextCursor", None)
            if not cursor:
                break
        return tools

    # ── tool registration ────────────────────────────────────────────

    def _register_server(self, server: str) -> list[str]:
        """Register one MCPTool per tool the server exposes. Idempotent.

        Thread-safe: also called from the loop thread when a background (OAuth)
        server finishes connecting. Returns the names newly registered.
        """
        if self._tool_registry is None:
            return []
        added: list[str] = []
        with self._lock:
            for mcp_tool in self._server_tools.get(server, []):
                try:
                    tool = MCPTool(self, server, mcp_tool)
                except Exception as e:
                    logger.warning("MCP: skipping a tool from '%s': %s", server, e)
                    continue
                tool._source_path = ""  # not file-backed
                if tool.name in self._registered:
                    continue
                self._tool_registry.register(tool)
                self._registered.add(tool.name)
                added.append(tool.name)
        if added:
            logger.info("MCP '%s': registered %d tool(s): %s", server, len(added), ", ".join(added))
        return added

    def _register_tools(self):
        """Register tools for every connected server. Idempotent."""
        if self._tool_registry is None:
            return
        for server in list(self._server_tools):
            self._register_server(server)
        self._tools_registered = True

    # ── synchronous call surface (used by MCPTool.run) ───────────────

    def call_mcp_tool(self, server: str, tool_name: str, arguments: dict):
        """Call a tool on a connected MCP server, blocking for the result.

        Raises ``TimeoutError`` if the server doesn't respond in time, or
        ``RuntimeError`` if the server isn't connected.
        """
        session = self._sessions.get(server)
        loop = self._loop
        if session is None or loop is None or loop.is_closed():
            raise RuntimeError(f"MCP server '{server}' is not connected.")
        fut = asyncio.run_coroutine_threadsafe(
            session.call_tool(tool_name, arguments or {}), loop,
        )
        try:
            return fut.result(timeout=self._tool_timeout)
        except asyncio.TimeoutError as e:  # pragma: no cover - timing dependent
            fut.cancel()
            raise TimeoutError(str(e)) from e

    # ── introspection (used by the /mcp command) ─────────────────────

    def registered_tools(self, server: str | None = None) -> list[str]:
        """Qualified names of currently registered tools, optionally one server's."""
        if server is None:
            return sorted(self._registered)
        prefix = qualified_tool_name(server, "")
        return sorted(name for name in self._registered if name.startswith(prefix))

    def is_connected(self, server: str) -> bool:
        """Whether a live session exists for this server."""
        return server in self._sessions


def build_services(config: dict) -> dict:
    """Build services."""
    return {"mcp": MCPService(config)}
