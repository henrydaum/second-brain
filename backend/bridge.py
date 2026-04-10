"""
Sync / async bridge utilities.

The Agent.chat() loop is synchronous (runs in a ThreadPoolExecutor),
while the WebSocket server is asyncio.  This module provides the
glue to safely cross that boundary:

  - Push events from a sync thread onto the async event loop.
  - Block a sync thread until an async Future is resolved (approvals).
  - Run blocking callables in an executor without stalling the loop.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine

logger = logging.getLogger("backend.bridge")

# Shared executor for running Agent.chat() and other blocking work.
# Sized for one agent turn per session — sessions are serialised anyway.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent")


async def run_sync(fn: Callable[..., Any], *args: Any,
                   loop: asyncio.AbstractEventLoop | None = None) -> Any:
    """Run a synchronous *fn* in the thread-pool without blocking the loop.

    Usage (from an async handler)::

        result = await run_sync(agent.chat, message)
    """
    loop = loop or asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


def push_event(coro: Coroutine, loop: asyncio.AbstractEventLoop) -> None:
    """Schedule an async coroutine from a synchronous thread.

    Fire-and-forget: the caller does not wait for the result.
    Used by Agent callbacks (on_tool_result, on_message) to send
    WebSocket events back to connected clients.

    Usage (from a sync callback running in the executor)::

        push_event(session.broadcast(event), loop)
    """
    asyncio.run_coroutine_threadsafe(coro, loop)


def block_on_async(coro: Coroutine, loop: asyncio.AbstractEventLoop,
                   timeout: float = 60.0) -> Any:
    """Block the current (sync) thread until *coro* completes on *loop*.

    Returns the coroutine's result, or raises ``TimeoutError`` if
    *timeout* seconds elapse.

    Used by the approval flow: the tool thread blocks here while
    waiting for the frontend to respond to an approval.request event.
    """
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)
