"""A listening socket, owned by the kernel and lent to one frontend.

``socket`` and ``http`` are refused to a guest, and ``sdk.net.http`` — the
route the validator points at instead — is an outbound client. So a frontend
could talk to the world and never be talked to, which is fine for a transport
that polls somebody else's servers (Telegram) and impossible for one a client
connects *to*. This is that missing half, built the way
:mod:`sandbox.console` builds its own: **the host listens, the guest drains.**

The kernel accepts and parses into a bounded buffer; ``http.drain`` takes what
has arrived and returns immediately, so a frontend never blocks and renders
keep landing between polls. The child process never opens a socket at all,
which is what lets a sandboxed HTTP frontend run *more* isolated than a native
one rather than less — and it is why the ERROR-level refusal on ``socket`` can
stay exactly as it is. This adds a mediated route to a capability rather than
opening an unmediated one.

**Parsing is ``http.server``'s**, not ours. Only guests are refused it, and a
hand-rolled request parser is a hundred lines of exactly the code that is worth
not owning: header folding, malformed lengths, the difference between a slow
client and a finished one. What is written by hand here is the *response*
side, because that is where the inversion lives — a reply outlives the call
that produced it, which no handler-returns-a-body framework expresses.

**One claimant**, declared by ``serves_http = <port>``; a second is refused.
Two frontends on one port is not merely ambiguous, it is unbindable. The
declaration is the *default*: ``<name>_port`` in config wins when set, so a
person can move a port without editing a plugin.

**Releasing does not close the listener.** The console's docstring explains at
length how releasing its reader put two readers on one stdin; the failure here
is different in mechanism and identical in shape. Closing the socket on release
means the next claim has to rebind, and a listener that has just closed cannot
always be rebound immediately — so a frontend *restart* would intermittently
come back with no port, which is exactly the "restart killed the terminal"
outcome that reasoning was written about. Release drops ownership, answers what
was pending, and leaves the socket bound. While unowned the server answers 503
rather than buffering: nobody is going to drain it, and a client deserves to be
told that now rather than to time out.

**The source is injectable.** ``start(port, source=...)`` takes any iterator of
``(request, writer)`` pairs, so tests drive a server without a socket — the
same reason ``Console.start`` takes an iterator of lines, and most of why this
is a Request rather than something a guest could have done for itself.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("Sandbox")

# How many unanswered requests to hold before dropping the oldest. A frontend
# that has stopped draining is already broken, and a client that can queue
# without bound is a way to spend the kernel's memory from outside it.
MAX_PENDING = 200

# The largest request body accepted. Attachments arrive as file paths through
# ``frontend.submit``, not as uploads, so nothing legitimate is near this.
MAX_BODY = 4 * 1024 * 1024

_REASONS = {200: "OK", 202: "Accepted", 400: "Bad Request",
            401: "Unauthorized", 403: "Forbidden", 404: "Not Found",
            413: "Payload Too Large", 500: "Internal Server Error",
            503: "Service Unavailable"}


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

    def send(self, status: int, headers: dict, body: str,
             streaming: bool) -> None:
        """Write the head, and the body if there is one to write now."""
        lines = [f"HTTP/1.1 {int(status)} {_REASONS.get(int(status), 'OK')}"]
        sent = {key.lower() for key in (headers or {})}
        for key, value in (headers or {}).items():
            lines.append(f"{key}: {value}")
        if streaming:
            lines.append("Content-Type: text/event-stream")
            lines.append("Cache-Control: no-cache")
        else:
            # Computed here unless the caller said otherwise. Without it a
            # client cannot know the body ended and waits for a close it has
            # no reason to expect — and a plugin author debugging that is
            # staring at a hung browser, not at a stack trace. Length in
            # *bytes*, because a multi-byte character would otherwise make the
            # header a lie.
            if "content-length" not in sent:
                lines.append(
                    f"Content-Length: {len(body.encode('utf-8'))}")
        # One request per connection: nothing here loops back for a second, so
        # promising keep-alive would leave a client waiting on a socket that is
        # about to close.
        if "connection" not in sent:
            lines.append("Connection: close")
        lines.append("")
        lines.append("")
        self._writer("\r\n".join(lines))
        self.streaming = streaming
        if body and not streaming:
            self._writer(body)
        elif body:
            self.frame(body)

    def frame(self, data: str, event: str = "") -> None:
        """Write one SSE frame.

        The guest supplies the payload and the kernel supplies the framing,
        because which lines carry a ``data:`` prefix is transport mechanics and
        getting it wrong is invisible until a client silently sees nothing.
        """
        out = []
        if event:
            out.append(f"event: {event}")
        out.extend(f"data: {line}" for line in str(data).split("\n"))
        out.append("")
        out.append("")
        self._writer("\n".join(out))

    def close(self) -> None:
        """Finish the reply. Idempotent."""
        if self.closed:
            return
        self.closed = True
        self._closer()


class _Handler(BaseHTTPRequestHandler):
    """Parse one request and hand it to the server, then hold the connection.

    Every verb lands in one place because routing is the guest's business —
    this knows nothing about paths. The thread then waits until the guest has
    finished answering, which is what keeps an SSE stream open: the connection
    lives exactly as long as this method has not returned.
    """

    protocol_version = "HTTP/1.1"
    server_version = "SecondBrain"
    sys_version = ""

    def log_message(self, fmt, *args):
        """Quiet. The ledger and the frontend are the record, not stderr."""
        logger.debug("http %s", fmt % args)

    def _take(self) -> None:
        """Buffer this request, then wait for the guest to finish with it."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._refuse(400, "malformed content-length")
            return
        if length > MAX_BODY:
            self._refuse(413, "request too large")
            return
        body = self.rfile.read(length) if length else b""
        path, _, query = self.path.partition("?")
        done = threading.Event()
        response = _Response(self._write, done.set)
        self.server.kernel._accept(
            {"id": "", "method": self.command, "path": path, "query": query,
             "headers": {k.lower(): v for k, v in self.headers.items()},
             "body": body.decode("utf-8", "replace")},
            response)
        # No deadline: a stream is *meant* to outlive its request, and the
        # frontend going away is what closes it — ``release`` and ``stop`` both
        # answer everything still open.
        done.wait()
        self.close_connection = True

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = (
        _take)

    def _write(self, text: str) -> None:
        """Put bytes on the wire, tolerating a client that has gone."""
        try:
            self.wfile.write(text.encode("utf-8"))
            self.wfile.flush()
        except OSError:
            self.close_connection = True

    def _refuse(self, status: int, message: str) -> None:
        """Answer a request that will not be served."""
        _fail(_Response(self._write, lambda: None), status, message)


class HttpServer:
    """One listener: a serving thread, a buffer, and at most one claimant."""

    def __init__(self):
        self._pending: deque = deque()
        self._open: dict[str, _Response] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._stopping = threading.Event()
        self._owner: str = ""
        self._port: int = 0
        # Which listener is the live one, for the reason the console keeps a
        # generation: a superseded one checks on the way back and stands down.
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
        self._abandon(pending, open_responses, "frontend released the port")

    def _abandon(self, pending, open_responses, why: str) -> None:
        """Answer everything in flight. Nothing may be left waiting."""
        for item in pending:
            _fail(item.get("_response"), 503, why)
        for response in open_responses.values():
            response.close()

    # ── the listener ───────────────────────────────────────────────

    def start(self, port: int, source=None) -> bool:
        """Begin listening. Idempotent for the same port and source.

        A second claim reuses the running listener rather than binding another.
        A *different* port or source supersedes: the old one is retired by
        generation and its socket closed, which is what wakes it out of accept.
        """
        with self._lock:
            live = self._thread is not None and self._thread.is_alive()
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
                    # Loopback only, and deliberately not configurable. Putting
                    # this on a public interface is a decision about exposure,
                    # and it belongs to whoever runs the tunnel — not to a
                    # plugin declaration the kernel reads.
                    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)),
                                                _Handler)
                except OSError:
                    logger.exception("could not bind 127.0.0.1:%s", port)
                    return False
                httpd.daemon_threads = True
                httpd.kernel = self
                self._httpd = httpd
                self._port = httpd.server_address[1]
                target, args = httpd.serve_forever, ()
            else:
                self._port = int(port or 0)
                target, args = self._serve_source, (source, generation)
            self._thread = threading.Thread(target=target, args=args,
                                            daemon=True, name="http-listener")
            self._thread.start()
            return True

    def _retire(self) -> None:
        """Close whatever is listening now. Caller holds the lock."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            threading.Thread(target=httpd.shutdown, daemon=True).start()
            try:
                httpd.server_close()
            except OSError:
                pass

    def _serve_source(self, source, generation: int) -> None:
        """Take pre-built ``(request, writer)`` pairs. The test path."""
        try:
            for request, writer in source:
                with self._lock:
                    if self._stopping.is_set() or generation != self._generation:
                        return
                self._accept(dict(request), _Response(writer, lambda: None))
        except Exception as exc:
            logger.debug("http source ended: %s", exc)

    def _accept(self, request: dict, response: _Response) -> None:
        """Buffer one parsed request, or refuse it if nobody will drain it."""
        dropped = None
        with self._lock:
            owned = bool(self._owner)
            if owned:
                # The one place an id is minted, so tracing where one came
                # from has a single answer.
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
            self._thread = None
            self._source = None
            self._port = 0
        self._abandon(pending, open_responses, "the server is shutting down")

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
        response.send(status, headers or {}, body, stream)
        if not stream:
            self.close(request_id)
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
            response = self._open.pop(request_id or "", None)
        if response is None:
            return False
        response.close()
        return True


def _fail(response, status: int, message: str) -> None:
    """Answer a request nobody is going to handle."""
    if response is None or getattr(response, "closed", True):
        return
    body = json.dumps({"error": message})
    response.send(status, {"Content-Type": "application/json"}, body, False)
    response.close()


# One server, because one process lends one port. Tests build their own rather
# than reaching for this.
SERVER = HttpServer()
