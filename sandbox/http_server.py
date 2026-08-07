"""A listening socket, owned by the kernel and lent to one frontend.

``socket`` and ``http`` are refused to a guest, and ``sdk.net.http`` — the
route the validator points at instead — is an outbound client. So a frontend
could talk to the world and never be talked to, which is fine for a transport
that polls somebody else's servers (Telegram) and impossible for one a client
connects *to*. This is that missing half, built the way
:mod:`sandbox.console` builds its own: **the host listens, the guest drains.**

One thread accepts connections and parses requests into a bounded buffer;
``http.drain`` takes what has arrived and returns immediately, so a frontend
never blocks and renders keep landing between polls. The child process never
opens a socket at all, which is what lets a sandboxed HTTP frontend run *more*
isolated than a native one rather than less — and it is why the ERROR-level
refusal on ``socket`` can stay exactly as it is. This adds a mediated route to
a capability rather than opening an unmediated one.

**One claimant**, declared by ``serves_http = <port>``; a second is refused.
Two frontends on one port is not merely ambiguous, it is unbindable.

**Responses outlive the request**, which is the one place this is not a
console. A console write is over when it returns; an SSE response is held open
for the life of a conversation and written to a frame at a time. So the buffer
of arrived requests has a companion — ``_open``, the responses still being
written — and three of the four Requests speak about it rather than about the
buffer.

**Releasing does not close the listener.** The console's docstring explains at
length how releasing its reader put two readers on one stdin; the failure here
is different in mechanism and identical in shape. Closing the socket on release
means the next claim has to rebind, and a listener that has just closed cannot
always be rebound immediately — so a frontend *restart* would intermittently
come back with no port, which is exactly the "restart killed the terminal"
outcome that reasoning was written about. Release drops ownership, answers
what was pending, and leaves the socket bound. While unowned the server
answers 503 rather than buffering: nobody is going to drain it, and a client
deserves to be told that now rather than to time out.

**The source is injectable.** ``start(port, source=...)`` takes any iterator of
``(request, responder)`` pairs, so tests drive a server without a socket — the
same reason ``Console.start`` takes an iterator of lines, and most of why this
is a Request rather than something a guest could have done for itself.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import uuid
from collections import deque

logger = logging.getLogger("Sandbox")

# How many unanswered requests to hold before dropping the oldest. A frontend
# that has stopped draining is already broken, and a client that can queue
# without bound is a way to spend the kernel's memory from outside it.
MAX_PENDING = 200

# The largest request body accepted. Attachments arrive as file paths through
# ``frontend.submit``, not as uploads, so nothing legitimate is near this.
MAX_BODY = 4 * 1024 * 1024

# How long a connection may be silent before the reader gives up on it. Bounds
# a connection opened and never written to, which would otherwise hold a
# thread for the life of the process.
READ_TIMEOUT = 30.0


class _BadRequest(ValueError):
    """A request that will not be served, and the status saying why.

    Carrying the status here rather than mapping messages at the catch site
    keeps the reason and its code together — a client told 400 for something
    that was actually too large has been told the wrong thing.
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class _Response:
    """One reply being written, which may outlive the call that opened it.

    A plain response is written once and closed. An SSE response stays open and
    takes frames until somebody closes it — that is what ``text/event-stream``
    is, and it is the only way an agent's turn reaches a client that asked
    before the turn started.
    """

    def __init__(self, writer, closer):
        self._writer = writer
        self._closer = closer
        self.streaming = False
        self.closed = False

    def head(self, status: int, headers: dict, streaming: bool) -> None:
        """Write the status line and headers."""
        lines = [f"HTTP/1.1 {int(status)} {_REASONS.get(int(status), 'OK')}"]
        for key, value in (headers or {}).items():
            lines.append(f"{key}: {value}")
        if streaming:
            lines.append("Content-Type: text/event-stream")
            lines.append("Cache-Control: no-cache")
            lines.append("Connection: keep-alive")
        lines.append("")
        lines.append("")
        self._writer("\r\n".join(lines))
        self.streaming = streaming

    def body(self, text: str) -> None:
        """Write a whole body."""
        self._writer(text)

    def frame(self, data: str, event: str = "") -> None:
        """Write one SSE frame.

        The guest supplies the payload and the kernel supplies the framing,
        because which lines carry a ``data:`` prefix is transport mechanics and
        getting it wrong is invisible until a client silently sees nothing.
        """
        out = []
        if event:
            out.append(f"event: {event}")
        for line in str(data).split("\n"):
            out.append(f"data: {line}")
        out.append("")
        out.append("")
        self._writer("\n".join(out))

    def close(self) -> None:
        """Finish the reply. Idempotent."""
        if self.closed:
            return
        self.closed = True
        self._closer()


_REASONS = {200: "OK", 202: "Accepted", 400: "Bad Request",
            401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
            413: "Payload Too Large", 500: "Internal Server Error",
            503: "Service Unavailable"}


class HttpServer:
    """One listener: an accept thread, a buffer, and at most one claimant."""

    def __init__(self):
        self._pending: deque = deque()
        self._open: dict[str, _Response] = {}
        self._lock = threading.RLock()
        self._listener: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self._stopping = threading.Event()
        self._owner: str = ""
        self._port: int = 0
        # Which listener is the live one, for the reason the console keeps a
        # generation: a thread blocked in ``accept`` cannot be woken by a flag,
        # so a superseded one checks on the way back and stands down.
        self._generation = 0
        self._source = None

    # ── claiming ───────────────────────────────────────────────────

    @property
    def owner(self) -> str:
        """The token holding the port, or "" if nobody does."""
        with self._lock:
            return self._owner

    @property
    def port(self) -> int:
        """The port actually bound, or 0. Reads back an ephemeral bind."""
        with self._lock:
            return self._port

    def claim(self, token: str, port: int, source=None) -> bool:
        """Take the port for one frontend. False if somebody else has it.

        Re-claiming with the same token succeeds, so a frontend that restarts
        is not locked out by its own previous claim.
        """
        if not token:
            return False
        with self._lock:
            if self._owner and self._owner != token:
                return False
            self._owner = token
        return self.start(port, source)

    def release(self, token: str) -> None:
        """Give the port back. Only the holder can, so a stale token from a
        frontend that already stopped cannot revoke its successor's claim.

        The listener is left bound — see the module docstring. What *is* given
        up is the work in flight: a request nobody will now answer is answered
        503 here, because leaving a client hanging on a frontend that has gone
        away is the one outcome worse than refusing it.
        """
        with self._lock:
            if not token or self._owner != token:
                return
            self._owner = ""
            pending, self._pending = list(self._pending), deque()
            open_responses, self._open = dict(self._open), {}
        for item in pending:
            _fail(item.get("_response"), 503, "frontend released the port")
        for response in open_responses.values():
            response.close()

    # ── the listener ───────────────────────────────────────────────

    def start(self, port: int, source=None) -> bool:
        """Begin listening. Idempotent for the same port and source.

        A second claim reuses the running listener rather than binding another.
        A *different* port or source supersedes: the old listener is retired by
        generation and its socket closed, which is what wakes it out of accept.
        """
        with self._lock:
            live = self._listener is not None and self._listener.is_alive()
            same = (source is None or source is self._source) and (
                not port or port == self._port)
            if live and same:
                return True
            self._retire()
            self._generation += 1
            generation = self._generation
            self._stopping.clear()
            self._source = source
            if source is None:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    # Loopback only, and deliberately not configurable. Putting
                    # this on a public interface is a decision about exposure,
                    # and it belongs to whoever runs the tunnel — not to a
                    # plugin declaration the kernel reads.
                    sock.bind(("127.0.0.1", int(port)))
                    sock.listen(16)
                except OSError:
                    logger.exception("could not bind 127.0.0.1:%s", port)
                    return False
                self._sock = sock
                self._port = sock.getsockname()[1]
            else:
                self._port = int(port or 0)
            self._listener = threading.Thread(
                target=self._serve, args=(source, generation), daemon=True,
                name="http-listener")
            self._listener.start()
            return True

    def _retire(self) -> None:
        """Close whatever is listening now. Caller holds the lock."""
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _serve(self, source=None, generation: int = 0) -> None:
        """Accept until the socket closes. Runs on its own thread."""
        if source is not None:
            self._serve_source(source, generation)
            return
        while True:
            with self._lock:
                sock = self._sock
                if self._stopping.is_set() or generation != self._generation:
                    return
            if sock is None:
                return
            try:
                conn, _ = sock.accept()
            except OSError:
                # The socket was closed under us, which is how retirement
                # reaches a thread parked in accept. Not an error.
                return
            threading.Thread(target=self._handle, args=(conn, generation),
                             daemon=True, name="http-conn").start()

    def _serve_source(self, source, generation: int) -> None:
        """Take pre-built ``(request, responder)`` pairs. The test path."""
        try:
            for request, responder in source:
                with self._lock:
                    if self._stopping.is_set() or generation != self._generation:
                        return
                self._accept(dict(request), _Response(responder, lambda: None))
        except Exception as exc:
            logger.debug("http source ended: %s", exc)

    def _handle(self, conn, generation: int) -> None:
        """Parse one request off a connection and buffer it."""
        conn.settimeout(READ_TIMEOUT)
        closed = threading.Event()

        def write(text: str) -> None:
            if closed.is_set():
                return
            try:
                conn.sendall(text.encode("utf-8"))
            except OSError:
                closed.set()

        def close() -> None:
            closed.set()
            try:
                conn.close()
            except OSError:
                pass

        response = _Response(write, close)
        try:
            request = _read_request(conn)
        except _BadRequest as exc:
            _fail(response, exc.status, str(exc))
            return
        except OSError:
            close()
            return
        if request is None:
            close()
            return
        with self._lock:
            stale = self._stopping.is_set() or generation != self._generation
        if stale:
            _fail(response, 503, "the server is shutting down")
            return
        self._accept(request, response)

    def _accept(self, request: dict, response: _Response) -> None:
        """Buffer one parsed request, or refuse it if nobody will drain it."""
        dropped = None
        with self._lock:
            owned = bool(self._owner)
            if owned:
                request["id"] = request.get("id") or uuid.uuid4().hex
                request["_response"] = response
                self._pending.append(request)
                while len(self._pending) > MAX_PENDING:
                    dropped = self._pending.popleft()
        if not owned:
            _fail(response, 503, "no frontend is serving this port")
            return
        if dropped is not None:
            _fail(dropped.get("_response"), 503, "the server is overloaded")

    def stop(self) -> None:
        """Close the listener and forget everything. Teardown and tests."""
        self._stopping.set()
        with self._lock:
            self._generation += 1
            self._retire()
            pending, self._pending = list(self._pending), deque()
            open_responses, self._open = dict(self._open), {}
            self._listener = None
            self._source = None
            self._port = 0
        for item in pending:
            _fail(item.get("_response"), 503, "the server is shutting down")
        for response in open_responses.values():
            response.close()

    # ── what the Requests reach ────────────────────────────────────

    def drain(self, limit: int = 0) -> list[dict]:
        """Every request that has arrived. Never blocks.

        Non-blocking on purpose, the same as ``Console.read_line``: the
        listener thread is what waits. A guest that blocked here would hold its
        box and could not render until the next client connected.

        The ``_response`` each request was carrying stays host-side — the guest
        gets an id, and holding an id is enough to answer and *only* enough to
        answer. Same split ``project_approval`` makes.
        """
        out = []
        with self._lock:
            while self._pending and (not limit or len(out) < limit):
                item = self._pending.popleft()
                response = item.pop("_response", None)
                if response is not None:
                    self._open[item["id"]] = response
                out.append(item)
        return out

    def respond(self, request_id: str, status: int = 200,
                headers: dict | None = None, body: str = "",
                stream: bool = False) -> bool:
        """Answer a request, or open it as a stream. False if it is not open."""
        with self._lock:
            response = self._open.get(request_id or "")
        if response is None or response.closed:
            return False
        response.head(status, headers or {}, stream)
        if stream:
            if body:
                response.frame(body)
            return True
        response.body(body)
        self._finish(request_id)
        return True

    def push(self, request_id: str, data: str, event: str = "") -> bool:
        """Write one frame to an open stream. False if there is no stream."""
        with self._lock:
            response = self._open.get(request_id or "")
        if response is None or response.closed or not response.streaming:
            return False
        response.frame(data, event)
        return True

    def close(self, request_id: str) -> bool:
        """End a reply. False if it was not open."""
        with self._lock:
            response = self._open.get(request_id or "")
        if response is None:
            return False
        self._finish(request_id)
        return True

    def _finish(self, request_id: str) -> None:
        """Close a reply and forget it."""
        with self._lock:
            response = self._open.pop(request_id or "", None)
        if response is not None:
            response.close()


def _read_request(conn) -> dict | None:
    """Parse one HTTP/1.1 request off a connection.

    Deliberately minimal: a request line, headers, and a ``Content-Length``
    body. Keep-alive, chunked bodies and pipelining are not supported, because
    the client is an SSE consumer that opens a stream and posts small JSON —
    and a fuller parser is surface this does not need.
    """
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = conn.recv(8192)
        if not chunk:
            return None
        buffer += chunk
        if len(buffer) > MAX_BODY:
            raise _BadRequest(413, "request too large")
    head, _, rest = buffer.partition(b"\r\n\r\n")
    lines = head.decode("utf-8", "replace").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        raise _BadRequest(400, "malformed request line")
    method, target = parts[0], parts[1]
    path, _, query = target.partition("?")
    headers = {}
    for line in lines[1:]:
        key, sep, value = line.partition(":")
        if sep:
            headers[key.strip().lower()] = value.strip()
    try:
        length = int(headers.get("content-length") or 0)
    except ValueError:
        raise _BadRequest(400, "malformed content-length") from None
    # Refused, not truncated. Clamping would buffer four megabytes and then
    # hand the guest a *valid-looking* request with its body silently cut in
    # half — a parse error somewhere else, blamed on the client.
    if length > MAX_BODY:
        raise _BadRequest(413, "request too large")
    while len(rest) < length:
        chunk = conn.recv(8192)
        if not chunk:
            break
        rest += chunk
    return {"id": uuid.uuid4().hex, "method": method.upper(), "path": path,
            "query": query, "headers": headers,
            "body": rest[:length].decode("utf-8", "replace")}


def _fail(response, status: int, message: str) -> None:
    """Answer a request nobody is going to handle."""
    if response is None or getattr(response, "closed", True):
        return
    body = json.dumps({"error": message})
    response.head(status, {"Content-Type": "application/json",
                           "Content-Length": str(len(body))}, False)
    response.body(body)
    response.close()


# One server, because one process lends one port. Tests build their own rather
# than reaching for this.
SERVER = HttpServer()
