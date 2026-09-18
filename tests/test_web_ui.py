"""Starting the web app, and the notification that says where it ended up.

Everything here is about the one failure this module keeps having: the UI works
and nobody is told, or the UI is announced and does not work. Both are silent —
a notification that is never raised looks exactly like one that was raised and
dropped, and an address in a notification looks equally correct whether or not
anything is listening behind it.

No real dev server: ``npm`` is minutes and a bound port, and what needs pinning
is the decision-making around it. The two things that are real are an HTTP
server standing in for the UI and the bus the notification actually travels on.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from events.event_bus import bus
from events.event_channels import NOTIFICATION_PUSHED
from runtime import web_ui


class _Runtime:
    """Just the one attribute ``web_ui`` reads."""

    def __init__(self, sessions=None):
        self.sessions = sessions if sessions is not None else {}


def _server(bridge_status: int | None):
    """A stand-in UI. ``bridge_status`` is what ``POST /sdk/...`` answers, or
    ``None`` to refuse the route the way a proxy with no upstream would."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if bridge_status is None:
                self.send_response(502)
            else:
                self.send_response(bridge_status)
            body = b'{"ok": true, "data": []}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def raised(monkeypatch):
    """Every notification ``web_ui`` puts on the bus, in order."""
    seen = []
    bus.subscribe(NOTIFICATION_PUSHED, seen.append)
    # The waits are real and the test should not be. Each test sets its own.
    monkeypatch.setattr(web_ui, "AUDIENCE_TIMEOUT", 3.0)
    # The bridge is asked repeatedly, because the HTTP frontend binds its port
    # a moment after the page starts answering and a single ask routinely got
    # there first — reporting "nothing answers behind it" about a frontend that
    # was two seconds away. Here the stand-in's answer is final on the first
    # ask, so the retry only costs the test its own timeout.
    monkeypatch.setattr(web_ui, "BRIDGE_TIMEOUT", 0.0)
    yield seen


def _settle(seen, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline and not seen:
        time.sleep(0.05)
    return seen


# ── The happy path, and what "works" means ───────────────────────────

def test_a_working_ui_is_announced_with_its_address(raised):
    """The line this module exists to produce, worded as the user asked."""
    server, url = _server(bridge_status=200)
    try:
        web_ui.serve({"ui_url": url, "ui_autostart": False},
                     _Runtime({"repl:main": object()}))
        assert _settle(raised), "nothing was announced"
    finally:
        server.shutdown()

    assert raised[0]["title"] == f"UI is reachable at: {url}"
    assert raised[0]["level"] == "success"


def test_a_page_that_loads_but_cannot_reach_the_kernel_says_so(raised):
    """The failure that shipped twice, and looked like success both times.

    A dev server started before the token existed proxies with no credential:
    the page loads perfectly and every Request comes back ``unauthorized``. A
    probe that only asked whether the port answered announced that address as
    working, so the user was told to open something that could not talk to
    Second Brain.
    """
    server, url = _server(bridge_status=401)
    try:
        web_ui.serve({"ui_url": url, "ui_autostart": False},
                     _Runtime({"repl:main": object()}))
        assert _settle(raised), "a broken bridge was not reported at all"
    finally:
        server.shutdown()

    assert "not talking to Second Brain" in raised[0]["title"]
    assert raised[0]["level"] == "warning"
    assert "token" in raised[0]["body"]


def test_a_page_with_nothing_behind_it_names_the_frontend(raised):
    """The other direction: the UI is fine and the HTTP frontend is off.

    A proxy with nothing upstream answers 502, which is why this is not
    "any error means refused": that reading told somebody whose frontend was
    switched off to go and fix their token. Different advice, so a different
    message — which is why ``bridge_ok`` answers three ways rather than two.
    """
    server, url = _server(bridge_status=None)
    try:
        web_ui.serve({"ui_url": url, "ui_autostart": False},
                     _Runtime({"repl:main": object()}))
        assert _settle(raised)
    finally:
        server.shutdown()

    assert "frontends enable http" in raised[0]["body"]


# ── Delivery ─────────────────────────────────────────────────────────

def test_the_notification_waits_for_a_frontend_to_exist(raised):
    """Raised into an empty runtime, a notification is delivered to nobody.

    ``on_bus_notification_pushed`` renders to live sessions and silently drops
    the rest, so announcing before a frontend has opened its session is exactly
    as useful as not announcing. That is not a race in the adopt path, it is
    the *normal* case: an already-running dev server answers the first probe in
    milliseconds, well before the REPL has a session.
    """
    server, url = _server(bridge_status=200)
    runtime = _Runtime()  # nobody home yet
    try:
        web_ui.serve({"ui_url": url, "ui_autostart": False}, runtime)

        time.sleep(1.0)
        assert not raised, "announced before any frontend could receive it"

        runtime.sessions["repl:main"] = object()
        assert _settle(raised), "never announced once a session existed"
    finally:
        server.shutdown()

    assert raised[0]["title"] == f"UI is reachable at: {url}"


def test_it_announces_anyway_rather_than_waiting_forever(raised):
    """A headless run has no sessions and still belongs in the log and panel.

    Waiting is a courtesy to whoever is watching, not a precondition.
    """
    server, url = _server(bridge_status=200)
    try:
        web_ui.serve({"ui_url": url, "ui_autostart": False}, _Runtime())
        assert _settle(raised, timeout=10.0)
    finally:
        server.shutdown()


# ── Not starting things ──────────────────────────────────────────────

def test_an_already_served_address_is_adopted_rather_than_restarted(raised):
    """Killing a server this process did not start is the worse failure.

    It is also what makes a ``/restart`` cheap and what lets a real deployment
    — the macOS gateway serves a build and has no dev server at all — be used
    rather than fought with.
    """
    server, url = _server(bridge_status=200)
    try:
        web_ui.serve({"ui_url": url, "ui_autostart": True},
                     _Runtime({"repl:main": object()}))
        assert _settle(raised)
        assert web_ui._process is None, "spawned a second server over a live one"
    finally:
        server.shutdown()


def test_an_empty_url_does_nothing_at_all(raised):
    """The off switch, and it must not be a crash."""
    web_ui.serve({"ui_url": "", "ui_autostart": True}, _Runtime())
    time.sleep(0.5)
    assert not raised
    assert web_ui._process is None


# ── Whose dev server is that? ────────────────────────────────────────
#
# The rule is "only stop what this *app* started", and the word that had to
# widen is *app*: a `/restart` re-execs, so the server still serving afterwards
# was started by a process that no longer exists. Reading it as "started by me"
# meant adopting it and then declining to ever stop it — so one ungraceful exit
# left a dev server outliving every boot for the rest of the machine's uptime.


@pytest.fixture
def owned(monkeypatch, tmp_path):
    """A pid file under tmp_path, and a kill that is recorded rather than done."""
    monkeypatch.setattr(web_ui, "PID_FILE", tmp_path / "web_ui.pid")
    monkeypatch.setattr(web_ui, "_process", None)
    monkeypatch.setattr(web_ui, "_adopted_pid", None)
    killed = []
    monkeypatch.setattr(web_ui, "_end", killed.append)
    return killed


def test_a_server_from_a_previous_run_is_reclaimed_and_stopped(owned, monkeypatch):
    """The orphan case, which is the one that made this permanent.

    Nothing in memory says the server is ours — there is no handle, because the
    process that held it is gone. What establishes it is the pid on the port
    matching the pid written down, which is a fact this process can check
    against reality rather than a promise it has to take on trust.
    """
    web_ui._remember(4321)
    monkeypatch.setattr(web_ui, "_owner_of_port", lambda port: 4321)

    web_ui._adopted_pid = web_ui._reclaim("http://localhost:5174")
    assert web_ui._adopted_pid == 4321
    web_ui.stop()

    assert owned == [4321]
    assert not web_ui.PID_FILE.exists(), "the record outlived what it named"


def test_a_server_somebody_else_is_running_is_left_alone(owned, monkeypatch):
    """Two ways to not be ours, and both must end in doing nothing.

    A dev server in somebody's own terminal has no record at all; a *stale*
    record names a pid that has since been reused, and trusting it would kill
    whatever inherited that number. Neither is worth a running process, so the
    check is against the port's current owner in both cases.
    """
    monkeypatch.setattr(web_ui, "_owner_of_port", lambda port: 9999)

    assert web_ui._reclaim("http://localhost:5174") is None  # no record

    web_ui._remember(4321)                                   # stale record
    assert web_ui._reclaim("http://localhost:5174") is None
    assert not web_ui.PID_FILE.exists(), "a stale record was kept"

    web_ui.stop()
    assert owned == [], "something nobody owns was killed"


def test_the_probe_carries_an_origin_so_a_real_gateway_answers_it():
    """A deployed UI refuses an origin-less POST, and that is not a broken token.

    ``/sdk`` is the one route the gateway credentials itself, so it checks the
    ``Origin`` header and refuses anything that does not carry the app's own —
    a cross-origin POST of that shape needs no preflight and would otherwise
    arrive pre-authenticated. A Vite dev server proxies with no such check, so
    a probe sending no ``Origin`` worked for as long as the only thing ever
    probed was a dev server, and then reported a perfectly healthy deployment
    as *"serving but its bridge is refused"* — pointing at a token that was
    never the problem.

    The failure is silent in the direction that matters: the advice is
    plausible, specific, and about the wrong thing entirely.
    """
    seen_origins = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            origin = self.headers.get("Origin")
            seen_origins.append(origin)
            # Exactly the Caddyfile's rule: our own origin, or forbidden.
            expected = f"http://127.0.0.1:{self.server.server_address[1]}"
            self.send_response(200 if origin == expected else 403)
            body = b'{"ok": true, "data": []}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        # A trailing slash, because that is how a person pastes an address in —
        # and `https://host/` is not equal to `https://host` to a string match.
        assert web_ui.bridge_ok(url + "/") is True
        assert seen_origins == [url]
    finally:
        server.shutdown()


def test_a_gateway_with_nothing_behind_it_is_not_a_running_ui():
    """502 means the proxy is up and the app is not — so autostart must fire.

    ``ui_url`` may be an address a gateway answers: a Tailscale name, a reverse
    proxy, anything fronting the app. That gateway is up whether or not the
    dev server is, and it says so with a 502.

    Reading "any HTTP answer" as *reachable* therefore made the kernel decide a
    server was already serving, skip the spawn, and then report the bridge as
    unanswered — advice about an HTTP frontend that was running perfectly. The
    arrangement autostart is most useful for was the one arrangement where it
    could never fire, and nothing in the message pointed at the real cause.
    """
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert web_ui.reachable(f"http://127.0.0.1:{server.server_address[1]}") is False
    finally:
        server.shutdown()


def test_a_page_that_merely_dislikes_the_request_is_still_a_running_ui():
    """The other side of the line: 404 and 500 are answers about *this* server.

    Vite answers before its own routes are ready, so a status code is not a
    verdict this probe is entitled to have an opinion about. Only the gateway
    family speaks about something upstream.
    """
    for status in (404, 500):
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            code = status

            def do_GET(self):
                self.send_response(self.code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            assert web_ui.reachable(f"http://127.0.0.1:{server.server_address[1]}") is True
        finally:
            server.shutdown()
