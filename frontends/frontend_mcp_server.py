"""Expose Second Brain as an MCP server over streamable HTTP.

MCP is the USB of LLM apps: with this frontend enabled, any MCP client
(Claude Code, Claude Desktop, other agents) can connect to your Second Brain
and use it as a tool — ask it questions answered by your agent with your
memory and tools, and browse your conversations.

Transport: streamable HTTP on localhost (configurable host/port). stdio is
deliberately not offered — the REPL owns this process's stdout. The server
runs on the frontend's daemon thread via uvicorn; ``stop()`` signals it down.

Sessions and identity: each MCP ``client_id`` maps to the session
``mcp:<client_id>``. The default client acts as the operator (base user) so
your own tools see your own brain; a non-default ``client_id`` is identified
as its own user (``per_user`` binding), giving that client an isolated
conversation space.

Attendance: MCP clients are agents, not humans, so sessions are explicitly
marked unattended and actions are submitted ``user_driven=False`` — the
active REPL/Telegram session keeps foreground status, interactive
(``background_safe=False``) tools refuse cleanly, and a turn that opens a
form or approval is cancelled with an explanatory note instead of hanging.
Every MCP-originated action lands in the action ledger under ``mcp:*``
session keys.
"""

from __future__ import annotations

dependencies_files = []
dependencies_pip = ["mcp"]

import logging

from plugins.BaseFrontend import BaseFrontend
from pipeline.database import DEFAULT_USER_ID
from state_machine.conversation_phases import BASE_PHASE

logger = logging.getLogger("MCPServer")

_DEFAULT_CLIENT = "default"
_CONTENT_CAP = 4000  # chars per message when rendering a transcript


class MCPServerFrontend(BaseFrontend):
    """Serve Second Brain to external MCP clients (chat + browse tools)."""

    name = "mcp_server"
    description = "MCP server: lets external agents chat with Second Brain and browse conversations over streamable HTTP."
    user_binding = "per_user"
    default_user_id = DEFAULT_USER_ID

    config_settings = [
        ("MCP Server Host", "mcp_server_host",
         "Interface the MCP server listens on. Keep 127.0.0.1 unless you "
         "understand the exposure — there is no authentication layer.",
         "127.0.0.1",
         {"type": "text"}),
        ("MCP Server Port", "mcp_server_port",
         "Port for the streamable-HTTP MCP endpoint (/mcp). Requires restart.",
         8766,
         {"type": "slider", "range": (1024, 65535, 100), "is_float": False}),
        ("MCP Expose Browse Tools", "mcp_server_expose_browse",
         "Also expose read-only list_conversations / read_conversation tools "
         "to MCP clients (ownership-guarded). Off = chat tool only.",
         True,
         {"type": "bool"}),
    ]

    def __init__(self):
        super().__init__()
        self._server = None
        self._identified: set[str] = set()

    # ── Lifecycle ───────────────────────────────────────────────────────

    def session_key(self, ctx) -> str:
        return f"mcp:{ctx or _DEFAULT_CLIENT}"

    def start(self) -> None:
        # Lazy imports: ``mcp`` (and its uvicorn dependency) arrive via
        # dependencies_pip; discovery can import this module without them.
        import uvicorn
        from mcp.server.fastmcp import FastMCP

        host = str(self.config.get("mcp_server_host") or "127.0.0.1")
        port = int(self.config.get("mcp_server_port") or 8766)
        try:
            mcp = FastMCP("second-brain", stateless_http=True)
        except TypeError:  # older SDK without the stateless flag
            mcp = FastMCP("second-brain")
        self._register_tools(mcp)

        config = uvicorn.Config(mcp.streamable_http_app(), host=host, port=port,
                                log_level="warning")
        self._server = uvicorn.Server(config)
        logger.info(f"MCP server listening on http://{host}:{port}/mcp")
        # Blocking by design: FrontendManager runs start() on a daemon thread.
        self._server.run()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            self._server = None

    # ── MCP tools ───────────────────────────────────────────────────────

    def _register_tools(self, mcp) -> None:
        import anyio

        @mcp.tool()
        async def ask_second_brain(prompt: str, client_id: str = _DEFAULT_CLIENT) -> str:
            """Send a message to Second Brain and get its agent's reply.

            The reply is produced by Second Brain's own agent with its memory
            and tools. Conversation context persists per client_id, so
            follow-up questions work. Omit client_id to talk as the operator;
            pass a stable client_id for an isolated identity.
            """
            return await anyio.to_thread.run_sync(self._ask, prompt, client_id)

        if not self.config.get("mcp_server_expose_browse", True):
            return

        @mcp.tool()
        async def list_conversations(client_id: str = _DEFAULT_CLIENT) -> str:
            """List this client's Second Brain conversations (id, title, kind)."""
            return await anyio.to_thread.run_sync(self._list_conversations, client_id)

        @mcp.tool()
        async def read_conversation(conversation_id: int, client_id: str = _DEFAULT_CLIENT) -> str:
            """Read one conversation's transcript (ownership-guarded)."""
            return await anyio.to_thread.run_sync(self._read_conversation, conversation_id, client_id)

    # ── Tool bodies (sync, run in worker threads) ───────────────────────

    def _session_for(self, client_id: str) -> str:
        """Bind (once) and return the session key for an MCP client."""
        key = self.session_key(client_id)
        self._tag_session(key)
        # An MCP client is an agent, not a human: pin the session unattended
        # so interactive tools refuse and replies never wait on a prompt.
        self.mark_unattended(key)
        if client_id != _DEFAULT_CLIENT and key not in self._identified:
            self.identify(key, client_id)
            self._identified.add(key)
        return key

    def _ask(self, prompt: str, client_id: str) -> str:
        key = self._session_for(client_id)
        # user_driven=False: never steal the active-session slot from a human
        # frontend. send_text is used directly — slash commands and forms are
        # human affordances, not part of the MCP surface.
        out = self.runtime.handle_action(key, "send_text", prompt, user_driven=False)

        note = ""
        session = self.runtime.get_session(key)
        if getattr(session.cs, "phase", BASE_PHASE) != BASE_PHASE:
            self.runtime.handle_action(key, "cancel", None, user_driven=False)
            note = ("\n\n[Second Brain opened an interactive prompt, which is "
                    "not supported over MCP; it was cancelled.]")

        messages = [m for m in (out.messages or []) if m]
        if not messages and out.error:
            messages = [f"Error: {out.error.get('message') or out.error.get('code') or 'request failed'}"]
        return ("\n\n".join(messages) or "(no reply)") + note

    def _list_conversations(self, client_id: str) -> str:
        key = self._session_for(client_id)
        db = getattr(self.runtime, "db", None)
        if db is None:
            return "No database available."
        user_id = self.runtime.session_user_id(key)
        rows = db.list_user_conversations(limit=50, user_id=user_id)
        if not rows:
            return "No conversations."
        lines = [f"{row.get('id')}: {row.get('title') or '(untitled)'}"
                 + (f"  [{row.get('kind')}]" if row.get("kind") not in (None, "user") else "")
                 for row in rows]
        return "\n".join(lines)

    def _read_conversation(self, conversation_id: int, client_id: str) -> str:
        key = self._session_for(client_id)
        db = getattr(self.runtime, "db", None)
        if db is None:
            return "No database available."
        # Same non-leaking refusal the kernel uses for cross-user access.
        if not self.runtime.assert_conversation_access(key, int(conversation_id)):
            return "No such conversation."
        lines = []
        for row in db.get_conversation_messages(int(conversation_id)):
            role = row.get("role")
            if role not in ("user", "assistant", "tool"):
                continue  # state/compaction markers are kernel-internal
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            if len(content) > _CONTENT_CAP:
                content = content[:_CONTENT_CAP] + f" …[truncated {len(content) - _CONTENT_CAP} chars]"
            label = f"tool:{row.get('tool_name')}" if role == "tool" else role
            lines.append(f"[{label}] {content}")
        return "\n\n".join(lines) or "Conversation is empty."

    # ── Rendering: replies travel back as MCP tool results, not pushes. ──

    def render_messages(self, session_key, messages):
        pass

    def render_attachments(self, session_key, paths):
        pass

    def render_form_field(self, session_key, form):
        pass

    def render_approval_request(self, session_key, req):
        pass

    def render_buttons(self, session_key, buttons):
        pass

    def render_error(self, session_key, error):
        pass
