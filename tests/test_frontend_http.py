"""The HTTP frontend: the whole connection between an app and the kernel.

This is the store plugin's conformance suite, and it matters more than a
frontend's usually would — nothing else is reachable from the client, so what
is not covered here is what an app silently cannot do.

The design it pins is deliberately thin. There are two surfaces:

* ``GET /events`` streams every ``render`` the kernel makes, **verbatim**. No
  mapping table, no protocol vocabulary, no message-id bookkeeping. A client
  that can read the eleven kinds can do what the REPL can — and more, since
  ``notification`` is a kind the REPL flattens back into chat.
* ``POST /sdk/<type>`` runs any Request, through ``frontend.act``, rooted at
  the session named by ``?thread=``. Nothing is allowlisted, because policy
  already decides — and an unsafe one raises a dialog that arrives on the very
  stream the client is holding.

Three things fail *quietly* when broken, so each has a test that says so out
loud: a render that is dropped rather than buffered, a route that skips the
bearer token, and a body that is allowed to name its own session.

Skips cleanly when no store ref is reachable.
"""

import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source

PLUGIN = "frontends/frontend_http.py"
TOKEN = "test-token-abc"


def _source_or_skip() -> str:
    text = store_source(PLUGIN)
    if text is None:
        pytest.skip(f"{PLUGIN} is not present on a local store ref")
    return text


@pytest.fixture(scope="module")
def source() -> str:
    return _source_or_skip()


# ──────────────────────────────────────────────────────────────────────
# What the kernel reads off the file. These run by default: the subject is
# kernel behaviour and the store file is the input.
# ──────────────────────────────────────────────────────────────────────

def test_it_conforms(source):
    """``conforms.`` is what says it will load in a box at all."""
    from sandbox.validator import validate

    report = validate(source, filename=Path(PLUGIN).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


def test_the_declarations_the_bridge_reads(source):
    """Four declarations decide whether this frontend works at all."""
    from sandbox.validator import validate

    declared = validate(source, filename=Path(PLUGIN).name).declarations
    assert declared.get("serves_http") == 8787
    # Without this, a submitted turn runs inline and holds the box, so nothing
    # could render while the agent was thinking — and the stream this frontend
    # is serving would carry nothing until the turn was already over.
    assert declared.get("background_submit") is True
    assert declared.get("name") == "http"
    caps = declared.get("capabilities") or {}
    assert caps.get("supports_streaming") is True
    assert caps.get("supports_typing") is True


def test_it_declares_every_request_it_makes(source):
    """``requests`` is documentation plus a *load-time* name check.

    Worth being exact, because the obvious reading is wrong and this file
    inherited that wrong reading from its predecessor. Nothing enforces the
    list at runtime for a frontend: ``bridge`` reads it into ``granted`` and
    only ever spends it as ``Chain.approved``, which is set solely for a
    command the state machine approved. What the declaration does buy is
    ``validator._check_requests`` refusing a name that is not a real Request
    type — so it catches a typo at load rather than a denial in production —
    and it tells a reader what this plugin reaches for.
    """
    declared = set(_declared_requests(source))
    for needed in ("http.drain", "http.respond", "http.push", "http.close",
                   "frontend.act", "frontend.collect", "frontend.attend",
                   "secret.reveal", "config.read", "fs.read_bytes"):
        assert needed in declared, f"{needed} is used but not declared"


def test_it_needs_no_standing_authority_of_its_own(source):
    """Every Request it makes directly is unconditionally safe.

    The interesting capability arrives through ``frontend.act``, one Request at
    a time, classified against the *session* rather than against this plugin.
    So the plugin's own declaration stays small — and it should keep staying
    small, because anything added here is authority the frontend holds all the
    time rather than authority a person is present for.
    """
    from sandbox.guest import requests as R
    from sandbox.policy import ALWAYS_SAFE

    for name in _declared_requests(source):
        assert name in ALWAYS_SAFE or name in (R.CONFIG_READ, R.SECRET_REVEAL), (
            f"{name} is not unconditionally safe, so this frontend would hold "
            f"standing authority the session model is meant to scope")


def test_it_holds_nothing_that_needs_a_subprocess(source):
    """The opposite claim to Telegram's, and worth stating for the same reason.

    Telegram owns an asyncio loop, so it *structurally* requires isolation: an
    in-process resident box runs each call on a fresh worker thread, and a loop
    bound in ``start`` belongs to somebody else by the time ``poll`` returns.
    This frontend owns no loop, no socket and no thread-affine anything — the
    kernel holds the port — so an installed copy resolving IN_PROCESS is
    correct rather than a gap.

    Pinned as the absence of foreign imports because that is what
    ``required_isolation`` actually reads. If this ever gains one, the answer
    changes on its own, which is the design working.
    """
    from sandbox.validator import validate

    report = validate(source, filename=Path(PLUGIN).name)
    assert not report.unmediated, (
        f"{sorted(report.unmediated)} would force a subprocess; if that is "
        f"intended, say why here")


def _declared_requests(source: str):
    """The ``requests`` list, read the way the validator reads it."""
    from sandbox.validator import validate

    return list(validate(source, filename=Path(PLUGIN).name)
                .declarations.get("requests") or [])


# ──────────────────────────────────────────────────────────────────────
# Behaviour, in a real box against a real socket. Marked ``store``: this is
# the plugin's own code, and a kernel change cannot break it.
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def owns_its_token():
    """Register the plugin as the owner of its own secret.

    ``policy._owns_setting`` asks the *setting registry* — deliberately, so
    ownership is a fact about what is installed rather than something a guest
    can assert about itself. Discovery populates it at install; a test that
    never installs has to say so, or ``secrets.reveal`` is refused and the
    frontend starts with no token and answers 401 to everything.
    """
    from plugins import plugin_discovery

    owners = plugin_discovery._setting_to_plugins
    before = set(owners.get("secret_http_token") or set())
    owners.setdefault("secret_http_token", set()).add("http")
    yield
    owners["secret_http_token"] = before


class _Session:
    """One live session, as the runtime holds it."""

    def __init__(self, frontend_name="http"):
        self.frontend_name = frontend_name
        self.attended = None
        self.user_id = 1
        self.conversation_id = None


class _Runtime:
    """Just enough runtime for ownership and attendance to be answerable."""

    def __init__(self):
        self.sessions = {}
        self.active_session_key = "somebody-else"
        #: Every ``session.state_set`` that landed, with the session it named.
        self.state = []

    def update_session_plugin_state(self, session_key, namespace, value):
        """Record which session a state write actually reached."""
        self.state.append((session_key, namespace, value))

    def is_attended(self, session_key):
        """The real rule: the frontend's opinion wins, else the active one."""
        session = self.sessions.get(session_key)
        if session is not None and session.attended is not None:
            return session.attended
        return session_key == self.active_session_key


class _Desk:
    """What ``sdk.frontend.*`` reaches: the native adapter, recorded.

    ``background_submit`` is False so anything that would be detached is not,
    which keeps assertions deterministic. The real adapter sets True.
    """

    background_submit = False
    name = "http"

    def __init__(self, runtime):
        self.runtime = runtime
        self.attendance = []

    def mark_attended(self, session_key):
        """Somebody opened a stream for this session."""
        # The real runtime deliberately does not create sessions as a side
        # effect of an attendance signal. EventSource opens before boot creates
        # the conversation, so the frontend must refresh attendance later.
        session = self.runtime.sessions.get(session_key)
        if session is not None:
            session.attended = True
        self.attendance.append((session_key, True))

    def mark_unattended(self, session_key):
        """Their stream went away."""
        session = self.runtime.sessions.get(session_key)
        if session is not None:
            session.attended = False
        self.attendance.append((session_key, False))

    def has_pending_approval(self, session_key):
        """Nothing is waiting in these tests."""
        return False


class _Frontend:
    """A loaded box, driven the way residency drives one.

    The adapter is deliberately not used. ``_adapt_frontend`` is *kernel* code
    and ``tests/test_sandbox_http_server.py`` covers it end to end; what these
    tests are about is the plugin's own behaviour, and calling the box directly
    is both the most direct route and the clearest statement of what residency
    does — park a desk, claim the port, bind the token, drive poll and render.
    """

    def __init__(self, box, server, token, desk, runtime):
        self._box = box
        self.server = server
        self.token = token
        self.desk = desk
        self.runtime = runtime

    @property
    def state(self):
        """Every ``session.state_set`` that landed, and where."""
        return self.runtime.state

    def poll(self):
        """One turn of the loop, as the kernel's poll thread would."""
        return self._box.call("poll")

    def render(self, session_key: str, kind: str, payload=None):
        """One render, as ``_adapt_frontend._render`` would forward it."""
        return self._box.call("render", session_key=session_key, kind=kind,
                              payload=payload)

    def settle(self, tries=60):
        """Poll until nothing more happens, for detached work to land."""
        for _ in range(tries):
            if not self.poll().data:
                return
            time.sleep(0.01)


@pytest.fixture
def running(tmp_path, source, owns_its_token):
    """A frontend box holding a real port, with a real token.

    A fresh ``Sandbox``, installed as *the* sandbox for the duration: the
    context factory is what answers config and secret Requests from inside the
    box, and ``frontend.act`` reaches the sandbox through ``bridge.get_sandbox``
    — a detached Request has to land in the same interpreter that is already
    answering for this frontend.

    The chain root is ``frontend:http`` because that is what residency assigns,
    and ``PersistentBox._identity`` reads the registered name off it — which is
    what makes ``policy._owns_setting`` recognise the plugin as the owner of
    its own ``secret_http_token``.
    """
    import sandbox.http_server as module
    from sandbox import Chain, Sandbox
    from sandbox.bridge import _SANDBOX, configure
    from sandbox.frontends import park, unpark
    from sandbox.http_server import HttpServer

    www = tmp_path / "www"
    www.mkdir()
    (www / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (www / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00binary")

    settings = {"secret_http_token": TOKEN, "http_allowed_origins": "",
                "http_static_dir": str(www)}
    runtime = _Runtime()

    path = tmp_path / "frontend_http.py"
    path.write_text(source, encoding="utf-8")

    box_sandbox = Sandbox()
    box_sandbox.bind_context(
        lambda session_key=None: SimpleNamespace(config=settings,
                                                 session_key=session_key,
                                                 runtime=runtime))
    server = HttpServer()
    desk = _Desk(runtime)
    token = park(desk)
    assert server.claim(token, 0), "the test server did not bind"

    # The plugin reaches ``sdk.http.*`` through the process-wide singleton, so
    # for the duration of one test that singleton *is* this server.
    previous_server, module.SERVER = module.SERVER, server
    previous_sandbox = _SANDBOX
    configure(box_sandbox)
    box = box_sandbox.open(path, "HTTP", name="frontend_http",
                           chain=Chain(root="frontend:http"))
    assert box.call("__bind__", token=token).ok
    assert box.call("start").ok, "the frontend refused to start"
    try:
        yield _Frontend(box, server, token, desk, runtime)
    finally:
        module.SERVER = previous_server
        configure(previous_sandbox)
        box_sandbox.shutdown()
        server.stop()
        unpark(token)


def _request(method: str, path: str, token=TOKEN, body: str = "",
             extra=()) -> bytes:
    """A raw HTTP request, with the bearer header unless told otherwise."""
    head = [f"{method} {path} HTTP/1.1", "Host: h"]
    if token is not None:
        head.append(f"Authorization: Bearer {token}")
    head.extend(extra)
    if body:
        head.append("Content-Type: application/json")
        head.append(f"Content-Length: {len(body.encode())}")
    return ("\r\n".join(head) + "\r\n\r\n" + body).encode()


def _open(frontend, raw: bytes):
    """Send a request and let the frontend pick it up.

    ``poll`` is driven here because in production the kernel's poll thread does
    it; a test that forgot would hang on a request nobody collected.
    """
    conn = socket.create_connection(("127.0.0.1", frontend.server.port),
                                    timeout=5)
    conn.sendall(raw)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if frontend.poll().data:
            break
        time.sleep(0.01)
    return conn


def _read(conn, until=b"\r\n\r\n", timeout=3.0) -> bytes:
    """Whatever has come back so far."""
    got = b""
    conn.settimeout(0.3)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and until not in got:
        try:
            chunk = conn.recv(65536)
        except OSError:
            break
        if not chunk:
            break
        got += chunk
    return got


def _status(raw: bytes) -> int:
    """The status line's code."""
    return int(raw.split(b" ")[1])


def _json_body(raw: bytes):
    """The JSON body of a complete one-shot response."""
    _, _, body = raw.partition(b"\r\n\r\n")
    return json.loads(body.decode("utf-8"))


def _frames(raw: bytes) -> list:
    """Every SSE ``data:`` frame in what has arrived so far."""
    out = []
    for line in raw.decode("utf-8", "replace").split("\n"):
        if line.startswith("data: "):
            try:
                out.append(json.loads(line[6:]))
            except ValueError:
                pass
    return out


# ──────────────────────────────────────────────────────────────────────
# The stream: renders, verbatim.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.store
def test_every_render_kind_crosses_unchanged(running):
    """The whole outbound contract, stated once.

    No translation is the design. If a kind ever needs reshaping on the way
    out, it belongs in the client — the moment this file starts mapping is the
    moment it stops being a general-purpose frontend.
    """
    conn = _open(running, _request("GET", "/events?thread=t1"))
    _read(conn, until=b"\r\n\r\n")

    sent = [
        ("messages", ["hello **world**"]),
        ("attachments", ["/tmp/a.png"]),
        ("form_field", {"name": "category", "display": {"prompt": "Which?"}}),
        ("approval", {"id": "r1", "title": "Run it?", "type": "boolean"}),
        ("approval_settled", {"request_id": "r1", "reason": "answered"}),
        ("buttons", [{"label": "Yes", "value": "yes"}]),
        ("error", {"message": "it broke"}),
        ("typing", True),
        ("tool_status", {"call_id": "c1", "tool_name": "search"}),
        ("stream_delta", {"stream_id": "s1", "seq": 0, "delta": "hi"}),
        ("notification", {"title": "Plugin registered", "body": "tool_x",
                          "source": "plugin_watcher", "level": "success"}),
    ]
    for kind, payload in sent:
        assert running.render("http:t1", kind, payload).ok

    frames = _frames(_read(conn, until=b"notification", timeout=3.0))
    conn.close()

    assert [f["kind"] for f in frames] == [kind for kind, _ in sent]
    for frame, (kind, payload) in zip(frames, sent):
        assert frame["payload"] == payload, f"{kind} was reshaped on the way out"
        assert frame["session_key"] == "http:t1"


@pytest.mark.store
def test_a_render_with_no_stream_is_kept_for_the_next_one(running):
    """A background turn nobody was watching still produced something.

    Losing it is invisible, which is what makes it the worst available failure:
    the agent looks like it said nothing rather than like it failed.
    """
    assert running.render("http:t2", "messages", ["a background result"]).ok

    conn = _open(running, _request(
        "GET", "/events?thread=t2", extra=["Last-Event-ID: 0"]))
    frames = _frames(_read(conn, until=b"background result", timeout=3.0))
    conn.close()

    assert [f["payload"] for f in frames] == [["a background result"]]


@pytest.mark.store
def test_a_reconnecting_client_resumes_where_it_left_off(running):
    """``Last-Event-ID`` is free with ``EventSource``, so the frames are
    numbered and a page refresh does not lose the turn that ran across it."""
    for text in ("first", "second", "third"):
        running.render("http:t3", "messages", [text])

    conn = _open(running, _request(
        "GET", "/events?thread=t3", extra=["Last-Event-ID: 2"]))
    got = _read(conn, until=b"third", timeout=3.0)
    conn.close()

    assert [f["payload"] for f in _frames(got)] == [["third"]]
    # And the id is on the wire, or the browser has nothing to send back.
    assert b"id: 3" in got


@pytest.mark.store
def test_opening_a_stream_says_somebody_is_watching(running):
    """Attendance is the whole reason an unsafe Request can be asked about
    rather than silently refused, and the stream is the honest signal for it."""
    running.runtime.sessions["http:t4"] = _Session()
    conn = _open(running, _request("GET", "/events?thread=t4"))
    _read(conn, until=b"\r\n\r\n")

    assert ("http:t4", True) in running.desk.attendance
    assert running.runtime.is_attended("http:t4")
    conn.close()


@pytest.mark.store
def test_a_client_that_leaves_stops_being_attended(running):
    """Learned on the write after they went, which is how SSE works.

    What must not happen is going on believing somebody is there — that would
    leave a dialog being raised for a session nobody can answer from.
    """
    conn = _open(running, _request("GET", "/events?thread=t5"))
    _read(conn, until=b"\r\n\r\n")
    conn.close()

    # Two renders: the first meets a socket the OS has not finished tearing
    # down, the second is the one that fails. Both are ordinary.
    for _ in range(4):
        running.render("http:t5", "messages", ["anyone there?"])
        if ("http:t5", False) in running.desk.attendance:
            break
        time.sleep(0.05)

    assert ("http:t5", False) in running.desk.attendance
    assert not running.runtime.is_attended("http:t5")


@pytest.mark.store
def test_first_request_after_session_creation_refreshes_attendance(running):
    """A browser opens EventSource before boot creates its conversation.

    The runtime correctly ignores attendance for a session that does not yet
    exist. Once boot has created it, the next Request must reassert that the
    still-open stream has a person behind it; otherwise an immediate
    ``conv.delete`` is refused instead of raising its approval dialog.
    """
    key = "http:new-session"
    conn = _open(running, _request("GET", "/events?thread=new-session"))
    _read(conn, until=b"\r\n\r\n")
    assert key not in running.runtime.sessions

    # ``conv.create`` has completed between HTTP Requests.
    running.runtime.sessions[key] = _Session()
    assert not running.runtime.is_attended(key)

    request = _open(running, _request(
        "POST", "/sdk/config.read?thread=new-session",
        body=json.dumps({"key": "http_static_dir"})))
    running.settle()
    raw = _read(request, until=b"}", timeout=3.0)
    request.close()

    assert _status(raw) == 200
    assert running.runtime.is_attended(key)
    conn.close()


# ──────────────────────────────────────────────────────────────────────
# The SDK route.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.store
def test_a_safe_request_round_trips(running):
    """One POST, one Result. The client never learns it was detached."""
    conn = _open(running, _request("POST", "/sdk/config.read?thread=t1",
                                   body=json.dumps({"key": "http_static_dir"})))
    running.settle()
    raw = _read(conn, until=b"}", timeout=3.0)
    conn.close()

    assert _status(raw) == 200
    assert _json_body(raw)["data"]


@pytest.mark.store
def test_a_refused_request_answers_with_its_code(running):
    """A refusal is an answer to forward, not a failure of this frontend.

    Nothing here decides it: ``session.add_tool`` is ALWAYS_UNSAFE, the chain
    is rooted at a session nobody has declared attended, so the approver has
    nobody to ask. The client gets 403 and the reason.
    """
    conn = _open(running, _request("POST", "/sdk/session.add_tool?thread=t6",
                                   body=json.dumps({"tool": "anything"})))
    running.settle()
    raw = _read(conn, until=b"}", timeout=3.0)
    conn.close()

    assert _status(raw) == 403
    assert _json_body(raw)["code"] == "approval_declined"


@pytest.mark.store
def test_an_unknown_request_type_is_named(running):
    """The client made a typo, and should be told which one."""
    conn = _open(running, _request("POST", "/sdk/conv.summon?thread=t1",
                                   body="{}"))
    raw = _read(conn, until=b"}", timeout=3.0)
    conn.close()

    assert _status(raw) == 400
    assert "conv.summon" in _json_body(raw)["error"]


@pytest.mark.store
def test_the_transport_is_not_reachable_through_the_sdk_route(running):
    """A client closing the stream it is being served on is the shape to keep
    impossible. Refused by the kernel rather than by a list here."""
    conn = _open(running, _request("POST", "/sdk/http.close?thread=t1",
                                   body=json.dumps({"request_id": "x"})))
    raw = _read(conn, until=b"}", timeout=3.0)
    conn.close()

    assert _status(raw) == 400


@pytest.mark.store
def test_a_client_cannot_name_its_own_session_or_token(running):
    """Identity is ours to state. A body claiming either is claiming to be
    somebody it is not, so both are dropped before the Request is built."""
    body = json.dumps({"session_key": "http:somebody-else",
                       "token": "stolen"})
    conn = _open(running, _request("POST", "/sdk/frontend.pending?thread=t7",
                                   body=body))
    running.settle()
    raw = _read(conn, until=b"}", timeout=3.0)
    conn.close()

    # It resolved *our* adapter with *our* thread, which a spoofed token could
    # not have done and a spoofed session_key would have redirected.
    assert _status(raw) == 200


@pytest.mark.store
def test_a_client_cannot_reach_another_frontends_session_by_key(running):
    """The other spelling, and the one that nearly got away.

    ``session.*`` names a session with ``key``, not ``session_key``, and of the
    eleven that accept one only ``add_prompt_extra`` checks it against the
    caller. So ``session.cancel {"key": "telegram:12345"}`` would have stopped
    a Telegram user's turn from a browser — unconditionally safe, no dialog,
    nothing in the way. The kernel's asymmetry predates this file; a bearer
    token reaching it does not.
    """
    running.runtime.sessions["telegram:12345"] = _Session("telegram")

    conn = _open(running, _request(
        "POST", "/sdk/session.state_set?thread=t9",
        body=json.dumps({"key": "telegram:12345", "namespace": "sandbox",
                         "value": "reached"})))
    running.settle()
    _read(conn, until=b"}", timeout=3.0)
    conn.close()

    # Rewritten to our own thread on the way through, so the other session was
    # never named at all.
    assert running.state == [("http:t9", "sandbox", "reached")]


@pytest.mark.store
def test_a_key_that_is_not_a_session_is_left_alone(running):
    """``key`` means a *setting name* to ``config.read``, so the rule has to be
    per-family. Stripping it everywhere would break ordinary reads."""
    conn = _open(running, _request("POST", "/sdk/config.read?thread=t1",
                                   body=json.dumps({"key": "http_static_dir"})))
    running.settle()
    raw = _read(conn, until=b"}", timeout=3.0)
    conn.close()

    assert _status(raw) == 200
    assert _json_body(raw)["data"], "the setting name was stripped"


# ──────────────────────────────────────────────────────────────────────
# The perimeter.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.store
@pytest.mark.parametrize("method,path", [
    ("GET", "/events?thread=t1"),
    ("POST", "/sdk/config.read?thread=t1"),
    ("GET", "/index.html"),
    ("GET", "/logo.png"),
])
def test_every_route_is_behind_the_token(running, method, path):
    """Including the static one. A token checked on some paths is not a
    perimeter, and the app's own HTML is as much worth withholding."""
    conn = _open(running, _request(method, path, token=None))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 401


@pytest.mark.store
def test_a_wrong_token_is_refused(running):
    """The other half of the same check."""
    conn = _open(running, _request("GET", "/events?thread=t1", token="nope"))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 401


@pytest.mark.store
def test_the_stream_accepts_a_query_token(running):
    """Because ``EventSource`` cannot send headers. That is the browser API,
    not an oversight, and it is the only client that reconnects on its own."""
    conn = _open(running, _request(
        "GET", f"/events?thread=t8&token={TOKEN}", token=None))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 200


@pytest.mark.store
@pytest.mark.parametrize("method,path", [
    ("POST", "/sdk/config.read"),
    ("GET", "/index.html"),
])
def test_no_other_route_accepts_one(running, method, path):
    """Narrow on purpose: a token in a URL reaches logs and browser history,
    so it buys exactly the one thing that cannot be done without it."""
    conn = _open(running, _request(
        f"{method}", f"{path}?token={TOKEN}", token=None,
        body="{}" if method == "POST" else ""))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 401


@pytest.mark.store
def test_preflight_is_answered_before_the_token(running):
    """A browser sends no Authorization header on OPTIONS, so checking it here
    would refuse every cross-origin request before it was ever made."""
    conn = _open(running, _request("OPTIONS", "/sdk/conv.list", token=None))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 204


@pytest.mark.store
def test_a_path_that_escapes_the_static_root_is_refused(running):
    """``fs.read_bytes`` is SAFE, so policy will not catch a careless join —
    this check is the only thing between a URL and the rest of the disk."""
    conn = _open(running, _request("GET", "/../../secrets.txt"))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) in (403, 404)


@pytest.mark.store
def test_a_binary_asset_is_served_unmangled(running):
    """Encoding a font or a PNG as text corrupts it, and nothing downstream
    would tell you."""
    conn = _open(running, _request("GET", "/logo.png"))
    raw = _read(conn, until=b"binary", timeout=3.0)
    conn.close()

    assert _status(raw) == 200
    assert b"\x89PNG\r\n\x1a\n\x00binary" in raw


@pytest.mark.store
def test_an_unknown_route_says_so(running):
    """With the static root configured, a bare 404 is still the answer for a
    path that looks like a file and is not one."""
    conn = _open(running, _request("GET", "/nope.css"))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 404


# ──────────────────────────────────────────────────────────────────────
# GET /files — host files as a URL.
#
# A client can already read any file through ``POST /sdk/fs.read_bytes``, so
# this grants nothing new. What it adds is a *transport*: a Request answers
# base64 inside JSON, and an ``<img>``, ``<video>`` or ``<audio>`` wants a URL.
# Assembling a Blob works for a picture and is hopeless for media — it buffers
# the whole file before the first frame and cannot seek.
# ──────────────────────────────────────────────────────────────────────

def _body(raw: bytes) -> bytes:
    return raw.partition(b"\r\n\r\n")[2]


def _header(raw: bytes, name: str) -> str:
    head = raw.partition(b"\r\n\r\n")[0].decode("utf-8", "replace")
    for line in head.split("\r\n")[1:]:
        key, _, value = line.partition(":")
        if key.strip().lower() == name.lower():
            return value.strip()
    return ""


def _files_url(path) -> str:
    from urllib.parse import quote

    return f"/files?path={quote(str(path), safe='')}"


@pytest.mark.store
def test_a_host_file_is_served_with_its_type(running, tmp_path):
    """The whole point: a real body with a real ``Content-Type``, so the
    browser decodes it natively instead of the client rebuilding a Blob."""
    target = tmp_path / "chart.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"pixels" * 100)

    conn = _open(running, _request("GET", _files_url(target)))
    raw = _read(conn, until=b"pixels", timeout=3.0)
    conn.close()

    assert _status(raw) == 200
    assert _header(raw, "Content-Type") == "image/png"
    assert _body(raw) == target.read_bytes()
    # Advertised unconditionally: a media element looks for this before it
    # decides whether it may seek.
    assert _header(raw, "Accept-Ranges") == "bytes"


@pytest.mark.store
def test_a_range_request_gets_only_that_span(running, tmp_path):
    """What a Blob can never do. ``<video>`` seeks by asking for the bytes it
    landed on rather than everything before them."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(bytes(range(256)) * 8)

    conn = _open(running, _request("GET", _files_url(target),
                                   extra=["Range: bytes=10-19"]))
    raw = _read(conn, until=b"\r\n\r\n", timeout=3.0)
    time.sleep(0.2)
    raw += _read(conn, until=b"~never~", timeout=0.5)
    conn.close()

    assert _status(raw) == 206
    assert _header(raw, "Content-Range") == f"bytes 10-19/{target.stat().st_size}"
    assert _body(raw) == bytes(range(10, 20))


@pytest.mark.store
def test_a_suffix_range_reads_from_the_end(running, tmp_path):
    """``bytes=-N`` is how a player reads a trailing index (an MP4 ``moov``
    atom) without downloading the file to find it."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"HEAD" + b"x" * 100 + b"TAIL")

    conn = _open(running, _request("GET", _files_url(target),
                                   extra=["Range: bytes=-4"]))
    raw = _read(conn, until=b"TAIL", timeout=3.0)
    conn.close()

    assert _status(raw) == 206 and _body(raw) == b"TAIL"


@pytest.mark.store
def test_a_range_past_the_end_is_refused_with_the_real_size(running, tmp_path):
    """416 carries ``Content-Range: bytes *–/size``, which is how a player
    learns the length it guessed wrong about."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"12345")

    conn = _open(running, _request("GET", _files_url(target),
                                   extra=["Range: bytes=900-999"]))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 416 and _header(raw, "Content-Range") == "bytes */5"


@pytest.mark.store
def test_a_file_too_big_for_one_message_comes_back_in_windows(running, tmp_path):
    """One response body crosses in one wire message, and that message is
    capped — so a large file is answered as 206 whether or not a Range was
    asked for, and the client comes back for the rest. A media element does
    that by itself; serving a truncated 200 instead would look like a file
    that simply ends early, which nothing downstream would report."""
    window = (16 * 1024 * 1024 - 1024 * 1024) * 3 // 4
    size = window + 4096
    target = tmp_path / "big.bin"
    target.write_bytes(b"\xa5" * size)

    conn = _open(running, _request("HEAD", _files_url(target)))
    head = _read(conn, timeout=3.0)
    conn.close()
    assert _status(head) == 200
    # HEAD answers the *whole* length, which is what a client sizes against.
    assert _header(head, "Content-Length") == str(size)

    conn = _open(running, _request("GET", _files_url(target)))
    raw = _read(conn, until=b"~never~", timeout=15.0)
    conn.close()
    assert _status(raw) == 206
    assert _header(raw, "Content-Range") == f"bytes 0-{window - 1}/{size}"
    assert len(_body(raw)) == window

    # And the tail is one more request, which is the whole point of 206.
    conn = _open(running, _request("GET", _files_url(target),
                                   extra=[f"Range: bytes={window}-"]))
    raw = _read(conn, until=b"~never~", timeout=15.0)
    conn.close()
    assert _status(raw) == 206 and len(_body(raw)) == 4096


@pytest.mark.store
def test_a_missing_file_is_a_404_not_a_crash(running, tmp_path):
    conn = _open(running, _request("GET", _files_url(tmp_path / "ghost.png")))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 404


@pytest.mark.store
def test_a_directory_is_not_a_file(running, tmp_path):
    conn = _open(running, _request("GET", _files_url(tmp_path)))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 400


@pytest.mark.store
def test_it_needs_the_bearer_token_like_everything_else(running, tmp_path):
    """The route sits *after* the auth check. A byte transport that skipped it
    would be the one way to read a file without the token."""
    target = tmp_path / "secret.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")

    conn = _open(running, _request("GET", _files_url(target), token=None))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 401


@pytest.mark.store
def test_asking_for_nothing_says_what_was_missing(running):
    conn = _open(running, _request("GET", "/files"))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 400 and "path" in _json_body(raw)["error"]


@pytest.mark.store
def test_a_media_element_can_authenticate_at_all(running, tmp_path):
    """``?token=`` is accepted here for the same reason ``/events`` accepts it:
    the browser issues the request itself and there is nowhere to put a header.

    An ``<img>`` or ``<video>`` fetches its own ``src``. Without this the route
    could only be reached by ``fetch`` — which means rebuilding a Blob, which
    is the thing it exists to avoid and which cannot seek. So the header-only
    version of this route is one that cannot do its job.
    """
    target = tmp_path / "chart.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\npixels")

    conn = _open(running, _request(
        "GET", f"{_files_url(target)}&token={TOKEN}", token=None))
    raw = _read(conn, until=b"pixels", timeout=3.0)
    conn.close()

    assert _status(raw) == 200 and _body(raw) == target.read_bytes()


@pytest.mark.store
def test_a_wrong_query_token_is_still_refused(running, tmp_path):
    """The concession is to where the token may travel, never to whether one
    is needed."""
    target = tmp_path / "chart.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n")

    conn = _open(running, _request(
        "GET", f"{_files_url(target)}&token=wrong", token=None))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 401


@pytest.mark.store
def test_no_other_route_takes_a_query_token(running):
    """A token in a URL reaches logs and history, so the list stays at two."""
    conn = _open(running, _request(
        "POST", f"/sdk/conv.list?thread=main&token={TOKEN}", token=None,
        body="{}"))
    raw = _read(conn, timeout=3.0)
    conn.close()

    assert _status(raw) == 401


@pytest.mark.store
def test_every_native_modality_gets_a_playable_type(running, tmp_path):
    """The two tables must agree, because a client uses both.

    Second Brain's ``parsing._NATIVE_DEFAULTS`` says what *kind* of file this
    is — the map a client categorises by, so it knows to reach for ``<video>``
    rather than ``<img>``. This route says how to *decode* it. Those are
    different questions with different answers, which is fine, right up until
    one recognises an extension the other does not: the client picks
    ``<video>``, the response says ``application/octet-stream``, and the
    element refuses to play a file that is perfectly good.

    So every extension the kernel can name a modality for must come back
    labelled with a matching top-level type. This found `.avi`, `.heic`,
    `.heif`, `.aac` and `.wma` missing.
    """
    from parsing.registry import _NATIVE_DEFAULTS

    mislabelled = []
    for extension, modality in sorted(_NATIVE_DEFAULTS.items()):
        target = tmp_path / f"probe{extension}"
        target.write_bytes(b"\x00\x01\x02\x03")
        conn = _open(running, _request("HEAD", _files_url(target)))
        raw = _read(conn, timeout=3.0)
        conn.close()
        served = _header(raw, "Content-Type")
        if served.partition("/")[0] != modality:
            mislabelled.append(f"{extension}: {modality} served as {served!r}")

    assert not mislabelled, (
        "a client categorising by modality would hand these to an element "
        "that will not play them:\n  " + "\n  ".join(mislabelled))


@pytest.mark.store
def test_an_unknown_extension_is_still_served(running, tmp_path):
    """Bytes for every extension; only the *label* falls back. A file the
    browser cannot render is a download, not an error."""
    target = tmp_path / "model.q4_k_m.gguf"
    target.write_bytes(b"GGUF\x00weights")

    conn = _open(running, _request("GET", _files_url(target)))
    raw = _read(conn, until=b"weights", timeout=3.0)
    conn.close()

    assert _status(raw) == 200
    assert _header(raw, "Content-Type") == "application/octet-stream"
    assert _body(raw) == target.read_bytes()


@pytest.mark.store
def test_a_file_with_no_extension_is_served_too(running, tmp_path):
    target = tmp_path / "LICENSE"
    target.write_bytes(b"MIT")

    conn = _open(running, _request("GET", _files_url(target)))
    raw = _read(conn, until=b"MIT", timeout=3.0)
    conn.close()

    assert _status(raw) == 200 and _body(raw) == b"MIT"
