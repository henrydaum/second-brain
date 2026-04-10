"""
Backend event-based service for Second Brain.

Exposes a single entry point: ``start_backend_server()`` which launches
a WebSocket server on a daemon thread.  Frontends connect as clients
and exchange JSON events/commands per the protocol defined in
``backend.protocol``.
"""

from backend.server import BackendServer


def start_backend_server(*, db, config, services, tool_registry,
                         orchestrator, ctrl, root_dir) -> BackendServer:
    """Create and start the backend WebSocket server.

    Returns the :class:`BackendServer` instance (for shutdown if needed).
    """
    server = BackendServer(
        db=db,
        config=config,
        services=services,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        ctrl=ctrl,
        root_dir=root_dir,
    )
    server.start()
    return server
