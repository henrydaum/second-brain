"""
Session management for the backend service.

A **Session** owns one Agent instance, one conversation, and one
CommandRegistry.  The SessionManager creates, looks up, and cleans up
sessions.  All agent lifecycle, conversation persistence, and approval
flow is centralised here — frontends never touch these internals.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from backend.protocol import (
    new_id, serialize,
    SessionCreated, SessionResumed, SessionError,
    AgentThinking, AgentToolCall, AgentToolResult,
    AgentMessage, AgentDone, AgentError, AgentCancelled,
    ApprovalRequest,
)
from backend.bridge import push_event, block_on_async

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

logger = logging.getLogger("backend.session")


# -------------------------------------------------------------------
# Session
# -------------------------------------------------------------------

class Session:
    """Server-side session — owns an Agent, a conversation, and connected clients."""

    def __init__(
        self,
        session_id: str,
        *,
        db: Any,
        config: dict,
        services: dict,
        tool_registry: Any,
        orchestrator: Any,
        ctrl: Any,
        root_dir: Any,
        loop: asyncio.AbstractEventLoop,
        conversation_id: int | None = None,
    ):
        self.id = session_id
        self.db = db
        self.config = config
        self.services = services
        self.tool_registry = tool_registry
        self.orchestrator = orchestrator
        self.ctrl = ctrl
        self.root_dir = root_dir
        self.loop = loop

        self.conversation_id = conversation_id
        self.agent = None                                  # created lazily
        self.clients: set[ServerConnection] = set()
        self.chat_lock = asyncio.Lock()
        self.pending_approvals: dict[str, asyncio.Future] = {}
        self._current_turn_id: str | None = None
        self._cancelled = False

    # ------- agent creation -------

    def _ensure_agent(self):
        """Create the Agent lazily on first chat message."""
        if self.agent is not None:
            return self.agent

        from Stage_3.agent import Agent
        from Stage_3.system_prompt import build_system_prompt

        llm = self.services.get("llm")
        if llm is None or not llm.loaded:
            return None

        self.agent = Agent(
            llm,
            self.tool_registry,
            self.config,
            system_prompt=lambda: build_system_prompt(
                self.db, self.orchestrator,
                self.tool_registry, self.services,
            ),
            on_tool_result=self._on_tool_result,
            on_message=self._on_message,
        )
        # Wire up approval callback for this session
        self.tool_registry.on_approve_command = self._approve_command
        return self.agent

    # ------- broadcasting -------

    async def broadcast(self, event) -> None:
        """Send a serialised event to every connected client."""
        data = serialize(event)
        closed = []
        for ws in self.clients:
            try:
                await ws.send(data)
            except Exception:
                closed.append(ws)
        for ws in closed:
            self.clients.discard(ws)

    # ------- agent callbacks (run on executor thread) -------

    def _on_tool_result(self, tool_name: str, result) -> None:
        """Fires after each tool execution inside Agent.chat()."""
        if self._current_turn_id is None:
            return
        evt = AgentToolResult(
            session_id=self.id,
            turn_id=self._current_turn_id,
            tool_call_id="",       # filled by the handler wrapper
            tool_name=tool_name,
            success=result.success,
            error=result.error or "",
            llm_summary=result.llm_summary or "",
            data=result.data,
            display_paths=result.gui_display_paths or [],
        )
        push_event(self.broadcast(evt), self.loop)

    def _on_message(self, msg: dict) -> None:
        """Fires for every message appended to agent history."""
        # 1. Persist to DB
        self._persist_message(msg)
        # 2. Broadcast to connected clients
        if self._current_turn_id is None:
            return
        evt = AgentMessage(
            session_id=self.id,
            turn_id=self._current_turn_id,
            role=msg.get("role", ""),
            content=msg.get("content") or "",
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id"),
            tool_name=msg.get("name"),
        )
        push_event(self.broadcast(evt), self.loop)

    def _persist_message(self, msg: dict) -> None:
        """Save a message to the conversations table."""
        role = msg.get("role", "")
        content = msg.get("content") or ""
        tool_call_id = msg.get("tool_call_id")
        tool_name = msg.get("name")

        # Lazy creation: first message creates the DB row
        if self.conversation_id is None:
            title = (content[:80].replace("\n", " ").strip()
                     if role == "user" else "New conversation")
            self.conversation_id = self.db.create_conversation(title)

        # Serialize tool_calls for assistant messages
        if msg.get("tool_calls"):
            content = json.dumps({
                "content": content,
                "tool_calls": msg["tool_calls"],
            })

        self.db.save_message(
            self.conversation_id, role, content,
            tool_call_id=tool_call_id, tool_name=tool_name,
        )

    # ------- approval flow (sync, blocks executor thread) -------

    def _approve_command(self, command: str, justification: str) -> bool:
        """Called by tools that need user approval (e.g. run_command).

        Blocks the tool thread until the frontend responds or timeout.
        """
        approval_id = new_id()

        async def _request():
            future = self.loop.create_future()
            self.pending_approvals[approval_id] = future
            await self.broadcast(ApprovalRequest(
                session_id=self.id,
                approval_id=approval_id,
                command=command,
                justification=justification,
            ))
            return await future

        try:
            return block_on_async(_request(), self.loop, timeout=60.0)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(f"Approval timed out for command: {command}")
            self.pending_approvals.pop(approval_id, None)
            return False

    def resolve_approval(self, approval_id: str, approved: bool) -> None:
        """Called by the handler when a frontend sends approval.response."""
        future = self.pending_approvals.pop(approval_id, None)
        if future and not future.done():
            future.set_result(approved)

    # ------- chat turn -------

    async def run_chat(self, message: str, turn_id: str) -> None:
        """Run a full agent chat turn, broadcasting events as they occur.

        Must be called while holding ``self.chat_lock``.
        """
        self._current_turn_id = turn_id
        self._cancelled = False

        agent = self._ensure_agent()
        if agent is None:
            await self.broadcast(AgentError(
                session_id=self.id, turn_id=turn_id,
                error="LLM is not loaded. Use /load llm to load it, "
                      "or /services to check status.",
            ))
            return

        from backend.bridge import run_sync

        try:
            response = await run_sync(agent.chat, message)
        except Exception as e:
            logger.error(f"Agent error in session {self.id}: {e}")
            await self.broadcast(AgentError(
                session_id=self.id, turn_id=turn_id, error=str(e),
            ))
            return
        finally:
            self._current_turn_id = None

        if self._cancelled:
            await self.broadcast(AgentCancelled(
                session_id=self.id, turn_id=turn_id,
            ))
        else:
            await self.broadcast(AgentDone(
                session_id=self.id, turn_id=turn_id,
                content=response or "",
            ))

    def cancel(self) -> None:
        """Request cancellation of the current agent turn."""
        self._cancelled = True
        if self.agent:
            self.agent.cancelled = True

    # ------- cleanup -------

    def disconnect_all(self) -> None:
        """Resolve all pending approvals as denied and clear clients."""
        for future in self.pending_approvals.values():
            if not future.done():
                future.set_result(False)
        self.pending_approvals.clear()
        self.clients.clear()

    # ------- slash commands -------

    def get_command_registry(self):
        """Build a CommandRegistry for this session (lazily cached)."""
        if not hasattr(self, "_cmd_registry"):
            from frontend.shared.commands import CommandRegistry, register_core_commands
            registry = CommandRegistry()
            register_core_commands(
                registry, self.ctrl, self.services,
                self.tool_registry, self.root_dir,
                get_agent=lambda: self.agent,
            )
            self._cmd_registry = registry
        return self._cmd_registry


# -------------------------------------------------------------------
# SessionManager
# -------------------------------------------------------------------

class SessionManager:
    """Creates, looks up, and cleans up sessions."""

    def __init__(self, *, db, config, services, tool_registry,
                 orchestrator, ctrl, root_dir, loop):
        self._db = db
        self._config = config
        self._services = services
        self._tool_registry = tool_registry
        self._orchestrator = orchestrator
        self._ctrl = ctrl
        self._root_dir = root_dir
        self._loop = loop
        self._sessions: dict[str, Session] = {}

    def create(self, conversation_id: int | None = None) -> Session:
        """Create a new session and return it."""
        sid = new_id()
        session = Session(
            sid,
            db=self._db,
            config=self._config,
            services=self._services,
            tool_registry=self._tool_registry,
            orchestrator=self._orchestrator,
            ctrl=self._ctrl,
            root_dir=self._root_dir,
            loop=self._loop,
            conversation_id=conversation_id,
        )
        self._sessions[sid] = session
        logger.info(f"Session created: {sid}")
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def destroy(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            session.disconnect_all()
            logger.info(f"Session destroyed: {session_id}")

    def remove_client(self, ws: "ServerConnection") -> None:
        """Remove a WebSocket from all sessions it's attached to.

        Cleans up sessions that have no remaining clients.
        """
        empty = []
        for sid, session in self._sessions.items():
            session.clients.discard(ws)
            if not session.clients:
                empty.append(sid)
        for sid in empty:
            self.destroy(sid)
