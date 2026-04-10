"""
Command handlers for the backend WebSocket server.

Each handler takes a parsed command dataclass + session manager +
websocket, performs the action, and sends response events.  This is
where ``route_input()`` logic from gui/dispatch.py lives now.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.protocol import (
    new_id, serialize,
    SessionCreated, SessionResumed, SessionError,
    CommandResult, AgentError,
    parse_command,
    SessionCreateCmd, SessionDestroyCmd,
    ChatSendCmd, ChatCancelCmd,
    CommandSendCmd, ApprovalResponseCmd, ToolCallCmd,
    AgentToolResult,
)

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection
    from backend.session import SessionManager

logger = logging.getLogger("backend.handlers")


async def dispatch(raw: dict, ws: "ServerConnection",
                   manager: "SessionManager") -> None:
    """Parse a raw JSON message and route to the correct handler."""
    try:
        cmd = parse_command(raw)
    except ValueError as e:
        await ws.send(serialize(SessionError(error=str(e))))
        return

    handler = _HANDLERS.get(type(cmd))
    if handler is None:
        await ws.send(serialize(SessionError(
            error=f"No handler for command type: {cmd.type}",
        )))
        return

    await handler(cmd, ws, manager)


# -------------------------------------------------------------------
# Individual handlers
# -------------------------------------------------------------------

async def _handle_session_create(cmd: SessionCreateCmd, ws: "ServerConnection",
                                 manager: "SessionManager") -> None:
    session = manager.create(conversation_id=cmd.conversation_id)
    session.clients.add(ws)

    # If resuming an existing conversation, load history from DB
    if cmd.conversation_id is not None:
        history = session.db.get_conversation_messages(cmd.conversation_id)
        if history:
            await ws.send(serialize(SessionResumed(
                session_id=session.id,
                conversation_id=cmd.conversation_id,
                history=history,
            )))
            return

    await ws.send(serialize(SessionCreated(
        session_id=session.id,
        conversation_id=session.conversation_id,
    )))


async def _handle_session_destroy(cmd: SessionDestroyCmd, ws: "ServerConnection",
                                  manager: "SessionManager") -> None:
    manager.destroy(cmd.session_id)


async def _handle_chat_send(cmd: ChatSendCmd, ws: "ServerConnection",
                            manager: "SessionManager") -> None:
    session = manager.get(cmd.session_id)
    if session is None:
        await ws.send(serialize(SessionError(
            session_id=cmd.session_id,
            error=f"Unknown session: {cmd.session_id}",
        )))
        return

    # Ensure this client is registered for events
    session.clients.add(ws)

    if session.chat_lock.locked():
        await ws.send(serialize(AgentError(
            session_id=session.id,
            turn_id="",
            error="A chat turn is already in progress for this session.",
        )))
        return

    turn_id = new_id()
    async with session.chat_lock:
        await session.run_chat(cmd.message, turn_id)


async def _handle_chat_cancel(cmd: ChatCancelCmd, ws: "ServerConnection",
                              manager: "SessionManager") -> None:
    session = manager.get(cmd.session_id)
    if session:
        session.cancel()


async def _handle_command_send(cmd: CommandSendCmd, ws: "ServerConnection",
                               manager: "SessionManager") -> None:
    session = manager.get(cmd.session_id)
    if session is None:
        await ws.send(serialize(SessionError(
            session_id=cmd.session_id,
            error=f"Unknown session: {cmd.session_id}",
        )))
        return

    session.clients.add(ws)
    registry = session.get_command_registry()
    output = registry.dispatch(cmd.command, cmd.arg)

    await ws.send(serialize(CommandResult(
        session_id=session.id,
        request_id=cmd.request_id,
        text=output or "",
    )))


async def _handle_approval_response(cmd: ApprovalResponseCmd,
                                    ws: "ServerConnection",
                                    manager: "SessionManager") -> None:
    # Approval IDs are globally unique — scan all sessions
    for session in manager._sessions.values():
        if cmd.approval_id in session.pending_approvals:
            session.resolve_approval(cmd.approval_id, cmd.approved)
            return
    logger.warning(f"Approval response for unknown ID: {cmd.approval_id}")


async def _handle_tool_call(cmd: ToolCallCmd, ws: "ServerConnection",
                            manager: "SessionManager") -> None:
    session = manager.get(cmd.session_id)
    if session is None:
        await ws.send(serialize(SessionError(
            session_id=cmd.session_id,
            error=f"Unknown session: {cmd.session_id}",
        )))
        return

    session.clients.add(ws)

    from backend.bridge import run_sync
    result = await run_sync(
        lambda: session.tool_registry.call(
            cmd.tool_name, approve_command=session._approve_command,
            **cmd.arguments,
        ),
    )

    await ws.send(serialize(AgentToolResult(
        session_id=session.id,
        turn_id="",
        tool_call_id=cmd.request_id,
        tool_name=cmd.tool_name,
        success=result.success,
        error=result.error or "",
        llm_summary=result.llm_summary or "",
        data=result.data,
        display_paths=result.gui_display_paths or [],
    )))


# -------------------------------------------------------------------
# Handler dispatch table
# -------------------------------------------------------------------

_HANDLERS = {
    SessionCreateCmd: _handle_session_create,
    SessionDestroyCmd: _handle_session_destroy,
    ChatSendCmd: _handle_chat_send,
    ChatCancelCmd: _handle_chat_cancel,
    CommandSendCmd: _handle_command_send,
    ApprovalResponseCmd: _handle_approval_response,
    ToolCallCmd: _handle_tool_call,
}
