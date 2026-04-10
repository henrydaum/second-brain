"""
Asyncio WebSocket server for the Second Brain backend.

Runs on its own daemon thread with a dedicated asyncio event loop.
Accepts WebSocket connections, reads JSON commands, and delegates
to ``handlers.dispatch()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection

from backend.protocol import deserialize, serialize, SessionError
from backend.handlers import dispatch
from backend.session import SessionManager

logger = logging.getLogger("backend.server")


class BackendServer:
    """Encapsulates the WebSocket server, its event loop, and the session manager."""

    def __init__(self, *, db, config, services, tool_registry,
                 orchestrator, ctrl, root_dir):
        self._db = db
        self._config = config
        self._services = services
        self._tool_registry = tool_registry
        self._orchestrator = orchestrator
        self._ctrl = ctrl
        self._root_dir = root_dir

        self._loop: asyncio.AbstractEventLoop | None = None
        self._manager: SessionManager | None = None
        self._thread: threading.Thread | None = None
        self._ws_server = None

    @property
    def port(self) -> int:
        return self._config.get("backend_port", 5150)

    # ------- lifecycle -------

    def start(self) -> None:
        """Start the WebSocket server on a daemon thread."""
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="backend-ws",
        )
        self._thread.start()

    def _run_loop(self) -> None:
        """Entry point for the daemon thread: create loop, server, and run forever."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._manager = SessionManager(
            db=self._db,
            config=self._config,
            services=self._services,
            tool_registry=self._tool_registry,
            orchestrator=self._orchestrator,
            ctrl=self._ctrl,
            root_dir=self._root_dir,
            loop=self._loop,
        )

        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        """Start the WebSocket server and run until stopped."""
        async with websockets.serve(
            self._handle_connection,
            "127.0.0.1",
            self.port,
            ping_interval=30,
            ping_timeout=10,
        ) as server:
            self._ws_server = server
            logger.info(f"Backend WebSocket server listening on ws://127.0.0.1:{self.port}")
            await asyncio.Future()  # run forever

    # ------- connection handler -------

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Handle a single WebSocket connection lifetime."""
        logger.info(f"Client connected: {ws.remote_address}")
        try:
            async for raw_msg in ws:
                try:
                    data = deserialize(raw_msg)
                except json.JSONDecodeError as e:
                    await ws.send(serialize(SessionError(
                        error=f"Invalid JSON: {e}",
                    )))
                    continue

                try:
                    await dispatch(data, ws, self._manager)
                except Exception as e:
                    logger.error(f"Handler error: {e}", exc_info=True)
                    await ws.send(serialize(SessionError(
                        error=f"Internal error: {e}",
                    )))
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
        finally:
            logger.info(f"Client disconnected: {ws.remote_address}")
            if self._manager:
                self._manager.remove_client(ws)
