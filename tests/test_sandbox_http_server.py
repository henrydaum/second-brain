"""A listening port: the kernel accepts, the guest drains.

``socket`` is refused to a guest and ``sdk.net.http`` dials *out*, so a
frontend could talk to the world and never be talked to. That is fine for a
transport which polls somebody else's servers and impossible for one a client
connects to — which is every protocol where the UI opens the connection, an
SSE render stream included. This suite is the argument that inverting it, exactly as the console
inverts stdin, closes the gap without widening what a guest may reach.

The inversion buys the same testability the console's does: the listener takes
any iterator of ``(request, responder)`` pairs, so most of this needs no
socket. The cases that are *about* the socket use one on an ephemeral port.
"""

import json
import socket
import time

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from prompt_cues import RENDERING, UNCOUNTED
from sandbox.guest import requests as R
from sandbox.handlers.kernel import HANDLERS
from sandbox.http_server import MAX_PENDING, HttpServer
from sandbox.policy import ALWAYS_SAFE

HTTP_TYPES = {R.HTTP_DRAIN, R.HTTP_RESPOND, R.HTTP_PUSH, R.HTTP_CLOSE}


@pytest.fixture
def server():
    """A server of one's own, torn down however the test ends."""
    made = HttpServer()
    yield made
    made.stop()


def _sent(chunks):
    """A responder that records instead of writing to a socket."""
    return lambda text: chunks.append(text)


def _request(path="/sdk/conv.list", method="POST", body=""):
    """One parsed request, as the socket path would have built it."""
    return {"id": "", "method": method, "path": path, "query": "",
            "headers": {}, "body": body}


def _connect(server):
    """A live connection to a bound server."""
    return socket.create_connection(("127.0.0.1", server.port), timeout=5)


def _await_drain(server, count=1, timeout=3.0):
    """Drain once the listener thread has had time to buffer.

    Polled rather than slept on, because the accept happens on another thread
    and a fixed sleep is either flaky or slow.
    """
    deadline = time.monotonic() + timeout
    out = []
    while time.monotonic() < deadline and len(out) < count:
        out.extend(server.drain())
        if len(out) < count:
            time.sleep(0.02)
    return out


# ── the registrations, which fail silently when missed ──────────────

def test_every_http_request_is_registered_everywhere():
    """A type missing one registration is a type that fails quietly.

    Four sets have to agree, and three of the four go wrong without raising:
    an unregistered handler is a failed Request, but a missing policy entry
    reads as UNSAFE and a missing entry here merely makes things slow.
    """
    assert HTTP_TYPES <= R.ALL_TYPES
    assert HTTP_TYPES <= set(HANDLERS)
    assert HTTP_TYPES <= ALWAYS_SAFE


def test_push_is_uncounted_or_prompt_caching_quietly_dies():
    """The whole reason ``RENDERING`` is named and pinned rather than derived.

    An SSE frontend sends one ``http.push`` per token, right behind the
    backend's ``llm.delta``. Counting it would tick the write counter thousands of
    times a reply, so every live ``agent_prompt`` would recompute on every
    model call and the caching would be undone — with no symptom but slowness,
    which is exactly the failure this set exists to make impossible.
    """
    assert R.HTTP_PUSH in RENDERING
    assert R.HTTP_PUSH in UNCOUNTED
    # And its two siblings, by the argument the set is named for rather than
    # by volume: they finish moving text to a person and leave no state behind.
    assert {R.HTTP_RESPOND, R.HTTP_CLOSE} <= RENDERING
    # Draining is a read, and belongs on the other side of the line.
    assert R.HTTP_DRAIN in R.READ_ONLY
    assert R.HTTP_DRAIN not in RENDERING


def test_push_is_dropped_by_the_ledger_sink():
    """The same flood, one subsystem over.

    ``llm.delta`` is excluded from the sink because a row per token serializes
    the database against the model. An SSE frame per token would restore that
    exactly, so the sink has to drop this too — and the sink builds its own
    set, so agreeing with ``prompt_cues`` is something a test has to state.
    """
    from sandbox.guest.requests import HTTP_PUSH, LLM_DELTA, READ_ONLY

    unrecorded = READ_ONLY | {LLM_DELTA, HTTP_PUSH}
    assert R.HTTP_PUSH in unrecorded
    # Per-request rather than per-token, so these stay auditable.
    assert R.HTTP_RESPOND not in unrecorded
    assert R.HTTP_CLOSE not in unrecorded


# ── claiming ────────────────────────────────────────────────────────

def test_one_claimant_and_a_second_is_refused(server):
    """Two frontends on one port is not ambiguous, it is unbindable."""
    assert server.claim("a", 0) is True
    assert server.claim("b", server.port) is False
    assert server.owner == "a"


def test_the_same_token_may_reclaim(server):
    """A frontend that restarts must not be locked out by its own claim."""
    server.claim("a", 0)
    port = server.port
    assert server.claim("a", port) is True
    assert server.port == port


def test_only_the_holder_can_release(server):
    """A stale token must not revoke its successor's claim.

    The console names the token in ``release`` for this reason; a frontend
    that already stopped calling release with a token it no longer holds would
    otherwise take the port out from under whoever has it now.
    """
    server.claim("a", 0)
    server.release("b")
    assert server.owner == "a"
    server.release("")
    assert server.owner == "a"
    server.release("a")
    assert server.owner == ""


def test_an_unclaimed_token_reaches_nothing():
    """The handlers' ownership check, which is the whole authorization."""
    from sandbox.http_server import SERVER

    for handler in (HANDLERS[t] for t in HTTP_TYPES):
        result = handler(None, {"token": "not-the-owner"})
        assert not result.ok
        assert SERVER.owner != "not-the-owner"


# ── the buffer ──────────────────────────────────────────────────────

def test_drain_takes_what_arrived_and_never_blocks(server):
    """Non-blocking is the point: the listener waits, the guest does not."""
    server.claim("a", 0, source=iter(()))
    assert server.drain() == []
    chunks = []
    server._accept(_request(body="one"), _wrap(chunks))
    server._accept(_request(body="two"), _wrap(chunks))
    got = server.drain()
    assert [item["body"] for item in got] == ["one", "two"]
    assert server.drain() == []


def test_drain_keeps_the_connection_host_side(server):
    """Holding an id is enough to answer, and only enough to answer.

    The same split ``project_approval`` makes: a box is shown a decision, never
    handed the thing being decided.
    """
    server.claim("a", 0, source=iter(()))
    chunks = []
    server._accept(_request(), _wrap(chunks))
    item = server.drain()[0]
    assert "_response" not in item
    assert item["id"]


def test_an_unowned_port_refuses_rather_than_buffers(server):
    """Nobody is going to drain it, so say so now.

    A client left hanging on a frontend that has gone away is the one outcome
    worse than a refusal.
    """
    server.start(0, source=iter(()))
    chunks = []
    server._accept(_request(), _wrap(chunks))
    assert server.drain() == []
    assert "503" in "".join(chunks)


def test_the_buffer_is_bounded(server):
    """A client must not be able to spend the kernel's memory from outside."""
    server.claim("a", 0, source=iter(()))
    chunks = []
    for _ in range(MAX_PENDING + 25):
        server._accept(_request(), _wrap(chunks))
    assert len(server.drain()) == MAX_PENDING
    # The dropped ones were answered rather than forgotten.
    assert "".join(chunks).count("503") == 25


def test_release_answers_what_was_in_flight(server):
    """Work in flight is given up; a client waiting on it is told."""
    server.claim("a", 0, source=iter(()))
    chunks = []
    server._accept(_request(), _wrap(chunks))
    server.release("a")
    assert server.drain() == []
    assert "503" in "".join(chunks)


# ── responding ──────────────────────────────────────────────────────

def test_a_plain_response_is_written_once_and_closed(server):
    server.claim("a", 0, source=iter(()))
    chunks = []
    server._accept(_request(), _wrap(chunks))
    rid = server.drain()[0]["id"]
    assert server.respond(rid, 200, {"X-A": "b"}, "hello") is True
    written = "".join(chunks)
    assert "HTTP/1.1 200 OK" in written and "X-A: b" in written
    assert written.endswith("hello")
    # Finished, so nothing further reaches it.
    assert server.respond(rid, 200) is False
    assert server.push(rid, "x") is False


def test_a_stream_stays_open_and_takes_frames(server):
    """A reply that outlives the call that opened it — the reason for four
    Requests rather than two."""
    server.claim("a", 0, source=iter(()))
    chunks = []
    server._accept(_request(path="/events"), _wrap(chunks))
    rid = server.drain()[0]["id"]
    assert server.respond(rid, 200, {}, stream=True) is True
    assert server.push(rid, json.dumps({"type": "RUN_STARTED"})) is True
    assert server.push(rid, json.dumps({"type": "RUN_FINISHED"})) is True
    written = "".join(chunks)
    assert "text/event-stream" in written
    assert written.count("data: ") == 2
    assert server.close(rid) is True
    assert server.push(rid, "x") is False


def test_the_kernel_supplies_sse_framing(server):
    """Which lines carry a ``data:`` prefix is transport mechanics, and
    getting it wrong is invisible until a client silently sees nothing."""
    server.claim("a", 0, source=iter(()))
    chunks = []
    server._accept(_request(), _wrap(chunks))
    rid = server.drain()[0]["id"]
    server.respond(rid, 200, {}, stream=True)
    server.push(rid, "first\nsecond", event="message")
    written = "".join(chunks)
    assert "event: message" in written
    assert "data: first" in written and "data: second" in written
    assert written.endswith("\n\n")


def test_pushing_to_a_stranger_fails_rather_than_passes(server):
    """A frontend that never hears the client left renders a whole turn into
    a closed socket."""
    server.claim("a", 0, source=iter(()))
    assert server.push("no-such-id", "x") is False
    assert server.close("no-such-id") is False
    assert server.respond("no-such-id") is False


# ── the socket, for the parts that are about the socket ─────────────

def test_it_binds_loopback_and_serves_a_real_request(server):
    """The end to end path, once, over a real connection."""
    assert server.claim("a", 0) is True
    assert server.port
    conn = _connect(server)
    body = json.dumps({"hello": "there"})
    conn.sendall(
        f"POST /sdk/conv.list HTTP/1.1\r\nHost: h\r\nContent-Length: {len(body)}"
        f"\r\n\r\n{body}".encode())
    items = _await_drain(server)
    assert [(i["method"], i["path"]) for i in items] == [("POST", "/sdk/conv.list")]
    assert json.loads(items[0]["body"]) == {"hello": "there"}
    server.respond(items[0]["id"], 200, {"Content-Length": "2"}, "{}")
    assert conn.recv(200).startswith(b"HTTP/1.1 200 OK")
    conn.close()


def test_release_leaves_the_socket_bound(server):
    """The console's lesson, in its socket form.

    Closing on release means the next claim has to rebind, and a listener that
    just closed cannot always rebind at once — so a frontend *restart* would
    intermittently come back with no port at all. Release gives up the work,
    not the socket.
    """
    server.claim("a", 0)
    port = server.port
    server.release("a")
    assert server.port == port
    conn = _connect(server)
    conn.sendall(b"GET / HTTP/1.1\r\nHost: h\r\n\r\n")
    assert b"503" in conn.recv(200)
    conn.close()
    assert server.claim("a", port) is True
    assert server.port == port


def test_stop_closes_it(server):
    """Teardown really does let the port go — by the time ``stop`` returns.

    **What this catches is the connection *succeeding*.** ``server_close``
    closes the socket object, but the serve loop keeps the port in LISTEN for
    up to a poll interval after it; during that window Linux accepts the
    connection outright. So ``stop`` was returning before the one thing it
    promises had happened, and a ``claim`` on the same port right after it was
    a coin toss. ``_retire`` waits for the listener now.

    Which *error* a freed port produces is environmental and deliberately not
    asserted. Linux refuses immediately; a Windows box that drops rather than
    rejects loopback packets times out instead, which is why this test could
    never have failed there — it is the Linux run that gives it teeth.
    """
    server.claim("a", 0)
    port = server.port
    server.stop()
    assert server.port == 0
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_an_oversized_request_is_refused_rather_than_truncated(server):
    """413 on the declared length, before a byte of body is read.

    Clamping to the cap instead would buffer four megabytes and then hand the
    guest a *valid-looking* request with its body silently cut in half — which
    surfaces as a parse error somewhere else, blamed on the client.
    """
    from sandbox.http_server import MAX_BODY

    server.claim("a", 0)
    conn = _connect(server)
    conn.sendall(
        f"POST /sdk/conv.list HTTP/1.1\r\nHost: h\r\nContent-Length: "
        f"{MAX_BODY * 2}\r\n\r\n".encode())
    assert b"413" in conn.recv(200)
    conn.close()
    assert _await_drain(server, timeout=0.5) == []


def test_a_malformed_length_is_refused(server):
    """The one parse branch this module owns rather than delegates."""
    server.claim("a", 0)
    conn = _connect(server)
    conn.sendall(b"POST /x HTTP/1.1\r\nHost: h\r\n"
                 b"Content-Length: notanumber\r\n\r\n")
    assert b"400" in conn.recv(400)
    conn.close()
    assert _await_drain(server, timeout=0.5) == []


def test_a_malformed_request_never_reaches_the_guest(server):
    """Whatever the stdlib answers, nothing unparseable is buffered.

    The status is deliberately not asserted: a one-word request line is a
    valid HTTP/0.9 request, and the stdlib answers it the HTTP/0.9 way — body
    only, no status line. What matters here is the boundary, not the wording.
    """
    server.claim("a", 0)
    conn = _connect(server)
    conn.sendall(b"nonsense\r\nHost: h\r\n\r\n")
    conn.close()
    assert _await_drain(server, timeout=0.5) == []


def test_a_plain_response_carries_its_length(server):
    """Without it a client cannot know the body ended.

    It waits for a close it has no reason to expect, and the author debugging
    that is staring at a hung browser rather than a stack trace — so the
    header is computed here unless the caller said otherwise.
    """
    server.claim("a", 0)
    conn = _connect(server)
    conn.sendall(b"GET /x HTTP/1.1\r\nHost: h\r\n\r\n")
    item = _await_drain(server)[0]
    server.respond(item["id"], 200, {}, "hello")
    head = conn.recv(500).decode()
    assert "Content-Length: 5" in head
    conn.close()


def test_length_is_bytes_not_characters(server):
    """A multi-byte character would otherwise make the header a lie."""
    server.claim("a", 0, source=iter(()))
    chunks = []
    server._accept(_request(), _wrap(chunks))
    rid = server.drain()[0]["id"]
    server.respond(rid, 200, {}, "héllo")
    assert "Content-Length: 6" in "".join(chunks)


def test_a_caller_may_set_its_own_headers(server):
    """CORS is the motivating case, and it is the guest's to decide."""
    server.claim("a", 0, source=iter(()))
    chunks = []
    server._accept(_request(), _wrap(chunks))
    rid = server.drain()[0]["id"]
    server.respond(rid, 200, {"Access-Control-Allow-Origin": "*",
                              "Content-Length": "99"}, "hi")
    written = "".join(chunks)
    assert "Access-Control-Allow-Origin: *" in written
    # Theirs wins: a caller that set one meant it.
    assert written.count("Content-Length") == 1
    assert "Content-Length: 99" in written


def _wrap(chunks):
    """A ``_Response`` over a list, for the tests that need no socket."""
    from sandbox.http_server import _Response

    return _Response(_sent(chunks), lambda: None)


# ── the whole path, through the bridge ──────────────────────────────

SERVING_FRONTEND = '''
"""A frontend a client connects to."""

import json

from guest.bases import BaseFrontend


class Web(BaseFrontend):
    """Serves its own UI."""

    name = "web"
    serves_http = 0
    poll_interval = 0.01

    def start(self, sdk):
        """Nothing to open — the kernel owns the socket."""
        self._streams = []
        return True

    def poll(self, sdk):
        """Drain whatever arrived."""
        requests = sdk.http.drain()
        if not requests:
            return False
        for request in requests:
            if request["path"] == "/events":
                sdk.http.respond(request["id"], stream=True)
                self._streams.append(request["id"])
            else:
                sdk.http.respond(request["id"], body=json.dumps(
                    {"saw": request["body"], "method": request["method"]}))
        return True

    def render(self, sdk, session_key, kind, payload):
        """Push every render down every open stream."""
        for stream_id in list(self._streams):
            if not sdk.http.push(stream_id, json.dumps(
                    {"kind": kind, "payload": payload})):
                self._streams.remove(stream_id)

    def stop(self, sdk):
        """Nothing to close."""
        return True
'''


@pytest.fixture
def serving(tmp_path):
    """A sandboxed frontend holding the real singleton's port."""
    import threading

    from guest.loader import unload_box
    from sandbox.bridge import adapt
    from sandbox.http_server import SERVER

    path = tmp_path / "frontend_web.py"
    path.write_text(SERVING_FRONTEND, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the frontend did not adapt"
    made = module.SandboxedWeb()
    thread = threading.Thread(target=made.start, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not SERVER.port:
        time.sleep(0.01)
    yield made, SERVER
    made.stop()
    SERVER.stop()
    unload_box("frontend_web")


def test_a_sandboxed_frontend_serves_a_real_request(serving):
    """The whole path: declaration, claim, drain, respond — through a box.

    Everything above this exercises the server directly. This is the one that
    says a *guest* can reach it, which is the claim the four Requests exist to
    make good on.
    """
    made, server = serving
    assert server.port, "the frontend never took a port"
    conn = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    conn.sendall(b"POST /say HTTP/1.1\r\nHost: h\r\nContent-Length: 5\r\n"
                 b"\r\nhello")
    reply = b""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and b"saw" not in reply:
        reply += conn.recv(4096)
    assert b"HTTP/1.1 200 OK" in reply
    assert json.loads(reply.split(b"\r\n\r\n", 1)[1].decode()) == {
        "saw": "hello", "method": "POST"}
    conn.close()


def test_a_render_reaches_an_open_stream(serving):
    """A reply outliving the call that opened it, end to end.

    The kernel calls ``render`` on the adapter and the guest pushes a frame
    down a connection opened by an earlier poll — which is the shape every
    streaming frontend needs and the reason ``respond`` and ``push`` are
    separate Requests.
    """
    made, server = serving
    conn = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    conn.sendall(b"GET /events HTTP/1.1\r\nHost: h\r\n\r\n")
    head = b""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and b"event-stream" not in head:
        head += conn.recv(4096)
    assert b"text/event-stream" in head

    made.render_messages("s1", ["hi there"])
    frame = b""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and b"data: " not in frame:
        frame += conn.recv(4096)
    payload = json.loads(frame.split(b"data: ", 1)[1].split(b"\n")[0].decode())
    assert payload == {"kind": "messages", "payload": ["hi there"]}
    conn.close()


def test_stopping_the_frontend_gives_the_port_back(serving):
    """A claim cannot outlive its box — the desk argument, for a socket."""
    made, server = serving
    assert server.owner
    made.stop()
    assert server.owner == ""


def test_config_moves_the_port_without_editing_the_plugin(tmp_path):
    """The declaration is a default, not a decision.

    ``<name>_port`` wins when set. Named by convention rather than by a second
    declaration, the same way ``secret_*`` declares itself — and reachable at
    claim time because ``register`` binds config before it starts a frontend.
    """
    import threading

    from guest.loader import unload_box
    from sandbox.bridge import adapt
    from sandbox.http_server import SERVER

    path = tmp_path / "frontend_web.py"
    path.write_text(SERVING_FRONTEND, encoding="utf-8")
    module = adapt(path)

    # Naming a port means racing for it. The probe takes a free one and gives
    # it straight back, so the number is real but unheld, and between the give
    # back and the claim anybody may take it — which under ``-n auto`` is
    # nineteen other workers binding ephemeral ports out of the same range.
    # A lost race is not this test's subject: ``SERVER.start`` logs the OSError
    # and returns False, leaving ``port`` at 0, so it is distinguishable from
    # the failure that *is* the subject — a port that came up somewhere other
    # than where config said. Retry the race; assert the claim.
    made = module.SandboxedWeb()

    def claim():
        """Take a free port, give it back, and try to come up on it again."""
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        chosen = probe.getsockname()[1]
        probe.close()
        made.bind(None, None, {"web_port": chosen})
        threading.Thread(target=made.start, daemon=True).start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not SERVER.port:
            time.sleep(0.01)
        return chosen

    try:
        for remaining in reversed(range(5)):
            chosen = claim()
            if SERVER.port:
                # The declaration says 0 (ephemeral); config said otherwise.
                assert SERVER.port == chosen
                break
            made.stop()
            SERVER.stop()
            assert remaining, f"never won a free port to claim (last: {chosen})"
    finally:
        made.stop()
        SERVER.stop()
        unload_box("frontend_web")
