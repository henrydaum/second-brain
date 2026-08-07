"""The AG-UI frontend: the whole connection between the app and the kernel.

This file is the store plugin's conformance suite, and it matters more than a
frontend's usually would — nothing else is reachable from the client, so what
is not tested here is what the app silently cannot do.

Three properties carry most of the weight, and all three fail *quietly* when
broken:

- ``supports_typing`` is what tells the frontend a run ended. Without it no SSE
  stream ever closes and every client hangs on a reply that finished.
- Events rendered with no stream open must be **buffered**, not dropped: a
  scheduled subagent's report has nowhere else to go and losing it is invisible.
- Every route is behind the bearer token, the static one included.

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

AGUI = "frontends/frontend_agui.py"
TOKEN = "test-token-abc"


def _source_or_skip() -> str:
    text = store_source(AGUI)
    if text is None:
        pytest.skip(f"{AGUI} is not present on a local store ref")
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

    report = validate(source, filename=Path(AGUI).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


def test_the_declarations_the_bridge_reads(source):
    """Four declarations decide whether this frontend works at all."""
    from sandbox.validator import validate

    declared = validate(source, filename=Path(AGUI).name).declarations
    assert declared.get("serves_http") == 8787
    # Without this ``submit_text`` runs the turn inline and holds the box, so
    # the stream just opened would carry nothing until the turn was over.
    assert declared.get("background_submit") is True
    assert declared.get("name") == "agui"
    caps = declared.get("capabilities") or {}
    # The one that fails silently. ``BaseFrontend._route_typing`` returns early
    # without it, so ``render("typing", False)`` never arrives, so no stream is
    # ever closed and every client hangs on a finished reply.
    assert caps.get("supports_typing") is True, \
        "without supports_typing no SSE stream is ever closed"
    assert caps.get("supports_streaming") is True


def test_it_declares_every_request_it_makes(source):
    """A Request left undeclared is refused at runtime, not at load."""
    declared = set(_declared_requests(source))
    for needed in ("http.drain", "http.respond", "http.push", "http.close",
                   "frontend.submit", "frontend.resolve", "frontend.pending",
                   "secret.reveal", "config.read", "fs.read_bytes"):
        assert needed in declared, f"{needed} is used but not declared"


def test_it_never_reaches_for_an_unattended_write(source):
    """The rule the whole API shape rests on.

    A frontend's chain roots at its session key, so it is unattended and an
    unsafe Request is *refused rather than asked* — silently. ``conv.delete``,
    ``config.write`` and ``command.call`` would therefore look like they worked
    and do nothing, which is why everything mutating goes back through
    ``submit_text("/command …")`` instead.
    """
    from sandbox.policy import ALWAYS_SAFE
    from sandbox.guest import requests as R

    for name in _declared_requests(source):
        assert name in ALWAYS_SAFE or name in (R.CONFIG_READ, R.SECRET_REVEAL), (
            f"{name} is not unconditionally safe; an unattended frontend chain "
            f"would have it refused rather than asked")


def _declared_requests(source: str):
    """The ``requests`` list, read the way the validator reads it."""
    from sandbox.validator import validate

    return list(validate(source, filename="frontend_agui.py")
                .declarations.get("requests") or [])


# ──────────────────────────────────────────────────────────────────────
# Behaviour, in a real box against a real socket. Marked ``store``: this is
# the plugin's own code, and a kernel change cannot break it.
# ──────────────────────────────────────────────────────────────────────

pytestmark_store = pytest.mark.store


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
    before = set(owners.get("secret_agui_token") or set())
    owners.setdefault("secret_agui_token", set()).add("agui")
    yield
    owners["secret_agui_token"] = before


class _Desk:
    """What ``sdk.frontend.submit_text`` reaches.

    In production this is the native adapter and submitting drives a whole
    agent turn. Here it only records, which is the right depth for these
    tests: what is under test is that the frontend hands the text over and
    keeps its stream open, not what the state machine then does with it.

    ``background_submit`` is False so the submit is inline and deterministic —
    the real adapter sets True and the kernel detaches, which would make every
    assertion a race.
    """

    background_submit = False

    def __init__(self):
        self.submitted = []
        self.pending = {}
        self.answered = None

    def submit_text(self, session_key, text):
        """Take a line, as the state machine's entry point would."""
        self.submitted.append((session_key, text))
        return SimpleNamespace(ok=True, messages=[], error=None)

    def has_pending_approval(self, session_key):
        """Whether an approval is waiting. Asked before every submit.

        The frontend asks rather than remembers, because an approval can be
        answered from another frontend or time out — so this has to answer for
        real, not just exist.
        """
        return session_key in self.pending

    def next_pending_approval_id(self, session_key):
        """The id of the approval that would be answered."""
        return self.pending.get(session_key)

    def is_approval_pending(self, session_key, request_id=None):
        """Whether *this* approval is still waiting.

        Asked by ``frontend.resolve`` before it does anything, which is what
        stops a stale answer landing on whatever is waiting now.
        """
        if request_id is None:
            return session_key in self.pending
        return self.pending.get(session_key) == request_id

    def resolve_approval(self, session_key, request_id, value, resolved_by=None):
        """Answer one by id."""
        if self.pending.get(session_key) != request_id:
            return False
        self.pending.pop(session_key, None)
        self.answered = (session_key, request_id, value)
        return True

    def resolve_next_approval(self, session_key, value, resolved_by=None):
        """Answer whichever is next."""
        request_id = self.pending.get(session_key)
        return bool(request_id) and self.resolve_approval(
            session_key, request_id, value, resolved_by)


class _Frontend:
    """A loaded AG-UI box, driven the way residency drives one.

    The adapter is deliberately not used here. ``_adapt_frontend`` is *kernel*
    code and `tests/test_sandbox_http_server.py` already covers it end to end;
    what these tests are about is the plugin's own behaviour, and calling the
    box directly is both the most direct way to reach it and the clearest
    statement of what residency actually does — park a desk, claim the port,
    bind the token, then drive ``poll`` and ``render``.
    """

    def __init__(self, box, server, token, desk):
        self._box = box
        self.server = server
        self.token = token
        self.desk = desk

    def poll(self):
        """One turn of the loop, as the kernel's poll thread would."""
        return self._box.call("poll")

    def render(self, session_key: str, kind: str, payload=None):
        """One render, as ``_adapt_frontend._render`` would forward it."""
        return self._box.call("render", session_key=session_key, kind=kind,
                              payload=payload)


@pytest.fixture
def running(tmp_path, source, owns_its_token):
    """A frontend box holding a real port, with a real token.

    A fresh ``Sandbox`` rather than the process-wide one: the context factory
    is what *answers* config and secret Requests from inside the box, and
    binding it on a sandbox other tests share is both fragile and rude.

    The chain root is ``frontend:agui`` because that is what residency
    assigns, and ``PersistentBox._identity`` reads the registered name off it —
    which is what makes ``policy._owns_setting`` recognise the plugin as the
    owner of its own ``secret_agui_token``. Opening without it would fall back
    to the *box* name and the reveal would be denied.
    """
    from sandbox import Chain, Sandbox
    from sandbox.frontends import park, unpark
    from sandbox.http_server import HttpServer

    www = tmp_path / "www"
    www.mkdir()
    (www / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    (www / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00binary")

    settings = {"secret_agui_token": TOKEN, "agui_allowed_origins": "",
                "agui_static_dir": str(www)}

    path = tmp_path / "frontend_agui.py"
    path.write_text(source, encoding="utf-8")

    box_sandbox = Sandbox()
    box_sandbox.bind_context(
        lambda session_key=None: SimpleNamespace(config=settings,
                                                 session_key=session_key))
    server = HttpServer()
    desk = _Desk()
    token = park(desk)
    assert server.claim(token, 0), "the test server did not bind"

    # The plugin reaches ``sdk.http.*`` through the process-wide singleton, so
    # for the duration of one test that singleton *is* this server.
    import sandbox.http_server as module

    previous, module.SERVER = module.SERVER, server
    box = box_sandbox.open(path, "AGUI", name="frontend_agui",
                           chain=Chain(root="frontend:agui"))
    assert box.call("__bind__", token=token).ok
    assert box.call("start").ok, "the frontend refused to start"
    try:
        yield _Frontend(box, server, token, desk)
    finally:
        module.SERVER = previous
        box_sandbox.shutdown()
        server.stop()
        unpark(token)


def _request(method: str, path: str, token=TOKEN, body: str = "") -> bytes:
    """A raw HTTP request, with the bearer header unless told otherwise."""
    head = [f"{method} {path} HTTP/1.1", "Host: h"]
    if token is not None:
        head.append(f"Authorization: Bearer {token}")
    if body:
        head.append("Content-Type: application/json")
        head.append(f"Content-Length: {len(body.encode())}")
    return ("\r\n".join(head) + "\r\n\r\n" + body).encode()


def _open(frontend, raw: bytes):
    """Send a request and let the frontend pick it up.

    ``poll`` is driven here because in production the kernel's poll thread
    does it; a test that forgot would hang on a request nobody collected.
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


def _events(conn, seed=b"", timeout=3.0):
    """Every SSE frame that has arrived, parsed.

    ``seed`` is whatever an earlier read already took off the socket. Waiting
    for ``RUN_STARTED`` can pull the frames right behind it in the same recv —
    the buffered ones flushed at the top of a run, exactly the case the
    buffering test is about — so they have to be carried forward rather than
    re-read from a socket that has already given them up.
    """
    got = seed + _read(conn, until=b"RUN_FINISHED", timeout=timeout)
    out = []
    for line in got.decode("utf-8", "replace").split("\n"):
        if line.startswith("data: "):
            try:
                out.append(json.loads(line[6:]))
            except ValueError:
                pass
    return out


def _run(frontend, thread: str):
    """Open an AG-UI run and return the live connection."""
    body = json.dumps({"threadId": thread, "runId": f"r-{thread}",
                       "messages": [{"id": "m1", "role": "user",
                                     "content": "hello there"}]})
    conn = _open(frontend, _request("POST", "/agui", body=body))
    opened = _read(conn, until=b"RUN_STARTED")
    assert b"text/event-stream" in opened
    return conn, opened


# ── the perimeter ───────────────────────────────────────────────────

@pytest.mark.store
def test_no_token_is_refused(running):
    """Everything past here assumes this holds."""
    conn = _open(running, _request("GET", "/commands", token=None))
    assert b"401" in _read(conn)
    conn.close()


@pytest.mark.store
def test_a_wrong_token_is_refused(running):
    conn = _open(running, _request("GET", "/commands", token="nope"))
    assert b"401" in _read(conn)
    conn.close()


@pytest.mark.store
def test_the_static_route_is_behind_the_token_too(running):
    """A token checked on some paths is not a perimeter.

    The app's own HTML is as much a thing worth not serving to strangers as
    the conversation is.
    """
    conn = _open(running, _request("GET", "/index.html", token=None))
    got = _read(conn)
    assert b"401" in got and b"<h1>" not in got
    conn.close()


@pytest.mark.store
def test_preflight_is_answered_without_a_token(running):
    """A browser sends no Authorization header on OPTIONS.

    Checking the token here would refuse every cross-origin request before it
    was ever made, and the answer reveals nothing the client did not send.
    """
    conn = _open(running, _request("OPTIONS", "/agui", token=None))
    assert b"204" in _read(conn)
    conn.close()


# ── serving the app ─────────────────────────────────────────────────

@pytest.mark.store
def test_a_binary_asset_survives_being_served(running):
    """The reason ``http.respond`` learned to take bytes.

    ``sdk.fs.read_bytes`` hands the guest real bytes; encoding them as UTF-8
    turns every PNG and woff2 into mojibake, and nothing downstream says so.
    """
    conn = _open(running, _request("GET", "/logo.png"))
    got = _read(conn, until=b"binary")
    assert b"image/png" in got
    assert b"\x89PNG\r\n\x1a\n\x00binary" in got
    conn.close()


@pytest.mark.store
def test_an_unknown_path_falls_back_to_the_shell(running):
    """A client-side router's path is not a missing file."""
    conn = _open(running, _request("GET", "/settings/models"))
    assert b"<h1>hi</h1>" in _read(conn, until=b"</h1>")
    conn.close()


@pytest.mark.store
def test_traversal_is_refused(running):
    """``fs.read_bytes`` is SAFE, so policy will not save a careless join."""
    conn = _open(running, _request("GET", "/../../etc/passwd"))
    assert b"403" in _read(conn)
    conn.close()


# ── the run lifecycle ───────────────────────────────────────────────

@pytest.mark.store
def test_a_run_streams_and_closes_on_typing_off(running):
    """The whole lifecycle, and the signal the design hangs on.

    ``render("typing", False)`` is the only thing that ends a run — AG-UI is
    request-scoped and ``background_submit`` means renders arrive later from
    another thread, so without it the stream would stay open forever.
    """
    conn, opened = _run(running, "t1")
    running.render("agui:t1", "stream_delta",
                   {"stream_id": "s1", "seq": 0, "delta": "hel"})
    running.render("agui:t1", "stream_delta",
                   {"stream_id": "s1", "seq": 1, "delta": "lo"})
    running.render("agui:t1", "stream_delta",
                   {"stream_id": "s1", "done": True, "final_text": "hello"})
    running.render("agui:t1", "typing", False)

    frames = _events(conn, opened)
    types = [e["type"] for e in frames]
    assert "TEXT_MESSAGE_START" in types
    assert types[-1] == "RUN_FINISHED"
    text = "".join(e.get("delta", "") for e in frames
                   if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text == "hello"
    conn.close()


@pytest.mark.store
def test_a_completed_stream_is_not_repeated_as_a_whole_message(running):
    """The kernel sends both; showing both would double every reply."""
    conn, opened = _run(running, "t2")
    running.render("agui:t2", "stream_delta",
                   {"stream_id": "s2", "delta": "hello"})
    running.render("agui:t2", "stream_delta",
                   {"stream_id": "s2", "done": True, "final_text": "hello"})
    running.render("agui:t2", "messages", ["hello"])
    running.render("agui:t2", "typing", False)
    frames = _events(conn, opened)
    starts = [e for e in frames if e["type"] == "TEXT_MESSAGE_START"]
    assert len(starts) == 1, "the streamed reply was shown twice"
    conn.close()


@pytest.mark.store
def test_a_message_that_never_streamed_is_sent_whole(running):
    conn, opened = _run(running, "t6")
    running.render("agui:t6", "messages", ["a plain reply"])
    running.render("agui:t6", "typing", False)
    frames = _events(conn, opened)
    text = "".join(e.get("delta", "") for e in frames
                   if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text == "a plain reply"
    conn.close()


@pytest.mark.store
def test_events_with_no_open_stream_are_buffered_not_dropped(running):
    """A scheduled subagent's report has nowhere else to go.

    AG-UI has no side channel, so anything produced between runs waits for the
    next one. Dropping it is silent and loses work the person asked for.
    """
    running.render("agui:t3", "messages", ["a background result"])
    conn, opened = _run(running, "t3")
    running.render("agui:t3", "typing", False)
    frames = _events(conn, opened)
    delivered = "".join(e.get("delta", "") for e in frames
                        if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert "a background result" in delivered
    conn.close()


# ── the four AG-UI has no words for ─────────────────────────────────

@pytest.mark.store
def _interrupts(frames):
    """The interrupts carried on this run's outcome."""
    for event in frames:
        if event["type"] == "RUN_FINISHED":
            outcome = event.get("outcome") or {}
            if outcome.get("type") == "interrupt":
                return outcome["interrupts"]
    return []


@pytest.mark.store
def test_an_approval_ends_the_run_as_an_interrupt(running):
    """The difference between an interrupt and a CUSTOM event is answerability.

    A CUSTOM event carries no way back, so an approval sent that way could be
    displayed and never resolved — the turn blocked while the client believed
    it had done its job. An interrupt has an id, a message and a
    ``responseSchema``, and the protocol says how to answer it.
    """
    conn, opened = _run(running, "t4")
    running.render("agui:t4", "approval",
                   {"id": "a1", "title": "Run shell?", "body": "git status",
                    "type": "boolean"})
    raised = _interrupts(_events(conn, opened))
    conn.close()

    assert len(raised) == 1
    assert raised[0]["id"] == "a1"
    assert raised[0]["reason"] == "confirmation"
    assert "Run shell?" in raised[0]["message"]
    assert "git status" in raised[0]["message"]
    # The kernel's own payload rides alongside, so a client that knows this
    # system renders the real thing rather than re-deriving it from prose.
    assert raised[0]["metadata"]["second_brain"]["request"]["id"] == "a1"


@pytest.mark.store
def test_an_enum_approval_offers_its_choices_and_their_labels(running):
    """``enum`` and ``enum_labels`` pair by index and both have to survive.

    Rendering the values would put internal spellings like
    ``always:api.search.brave.com`` on a person's buttons — written for a
    ledger row months later, not for somebody mid-decision.
    """
    conn, opened = _run(running, "t8")
    running.render("agui:t8", "approval",
                   {"id": "a2", "title": "Allow?", "type": "string",
                    "enum": ["allow", "always:x.com", "deny"],
                    "enum_labels": ["Allow once", "Always allow x.com", "Deny"]})
    raised = _interrupts(_events(conn, opened))
    conn.close()

    answer = raised[0]["responseSchema"]["properties"]["value"]
    assert answer["enum"] == ["allow", "always:x.com", "deny"]
    assert answer["enumLabels"][1] == "Always allow x.com"


@pytest.mark.store
def test_a_form_field_interrupts_as_input_required(running):
    """The form schema is already built; this only has to carry it."""
    conn, opened = _run(running, "t9")
    running.render("agui:t9", "form_field",
                   {"field": {"name": "port", "type": "integer"},
                    "display": {"prompt": "Which port?"}})
    raised = _interrupts(_events(conn, opened))
    conn.close()

    assert raised[0]["reason"] == "input_required"
    assert raised[0]["message"] == "Which port?"
    assert raised[0]["responseSchema"]["properties"]["value"]["type"] == "integer"
    form = raised[0]["metadata"]["second_brain"]["form"]
    assert form["field"]["name"] == "port"


@pytest.mark.store
def test_buttons_interrupt_with_their_values_as_choices(running):
    conn, opened = _run(running, "t10")
    running.render("agui:t10", "buttons",
                   [{"label": "Yes", "value": "y"},
                    {"label": "No", "value": "n"}])
    raised = _interrupts(_events(conn, opened))
    conn.close()

    answer = raised[0]["responseSchema"]["properties"]["value"]
    assert answer["enum"] == ["y", "n"]
    assert answer["enumLabels"] == ["Yes", "No"]


@pytest.mark.store
def test_resuming_an_approval_resolves_it(running):
    """The round trip, which is the whole reason for interrupts.

    The answer goes back the way the kernel already accepts one — ``resolve``
    by id — so an approval answered from the app takes exactly the path an
    approval answered from the REPL does.
    """
    conn, opened = _run(running, "t11")
    running.desk.pending["agui:t11"] = "a3"
    running.render("agui:t11", "approval",
                   {"id": "a3", "title": "Run shell?", "type": "boolean"})
    raised = _interrupts(_events(conn, opened))
    conn.close()

    body = json.dumps({"threadId": "t11", "runId": "r-resume", "messages": [],
                       "resume": [{"interruptId": raised[0]["id"],
                                   "status": "resolved",
                                   "payload": {"accepted": True}}]})
    conn = _open(running, _request("POST", "/agui", body=body))
    conn.close()
    assert running.desk.answered == ("agui:t11", "a3", True)


@pytest.mark.store
def test_a_cancelled_interrupt_denies_rather_than_hangs(running):
    """Closing the dialog is an answer, and the turn is entitled to hear it."""
    conn, opened = _run(running, "t12")
    running.desk.pending["agui:t12"] = "a4"
    running.render("agui:t12", "approval",
                   {"id": "a4", "title": "Delete everything?",
                    "type": "boolean"})
    raised = _interrupts(_events(conn, opened))
    conn.close()

    body = json.dumps({"threadId": "t12", "runId": "r-cancel", "messages": [],
                       "resume": [{"interruptId": raised[0]["id"],
                                   "status": "cancelled"}]})
    conn = _open(running, _request("POST", "/agui", body=body))
    conn.close()
    assert running.desk.answered == ("agui:t12", "a4", False)


@pytest.mark.store
def test_resuming_a_form_field_submits_the_value_as_text(running):
    """A form value is not an approval, and goes back the way text does."""
    conn, opened = _run(running, "t13")
    running.render("agui:t13", "form_field",
                   {"field": {"name": "port", "type": "integer"},
                    "display": {"prompt": "Which port?"}})
    raised = _interrupts(_events(conn, opened))
    conn.close()

    body = json.dumps({"threadId": "t13", "runId": "r-form", "messages": [],
                       "resume": [{"interruptId": raised[0]["id"],
                                   "status": "resolved",
                                   "payload": {"value": 8080}}]})
    conn = _open(running, _request("POST", "/agui", body=body))
    conn.close()
    assert ("agui:t13", "8080") in running.desk.submitted


@pytest.mark.store
def test_an_unknown_interrupt_id_is_ignored(running):
    """A stale resume must not be applied to whatever is waiting now."""
    conn = _open(running, _request("POST", "/agui", body=json.dumps(
        {"threadId": "t14", "runId": "r", "messages": [],
         "resume": [{"interruptId": "never-raised", "status": "resolved",
                     "payload": {"accepted": True}}]})))
    conn.close()
    assert not running.desk.submitted
    assert not getattr(running.desk, "answered", None)


@pytest.mark.store
def test_a_tool_call_becomes_agui_tool_events(running):
    conn, opened = _run(running, "t5")
    running.render("agui:t5", "tool_status",
                   {"call_id": "c1", "tool_name": "read_file",
                    "args": {"path": "x"}, "status": "started"})
    running.render("agui:t5", "tool_status",
                   {"call_id": "c1", "tool_name": "read_file",
                    "status": "finished", "result": "ok"})
    running.render("agui:t5", "typing", False)
    frames = _events(conn, opened)
    types = [e["type"] for e in frames]
    assert "TOOL_CALL_START" in types and "TOOL_CALL_RESULT" in types
    start = next(e for e in frames if e["type"] == "TOOL_CALL_START")
    assert start["toolCallName"] == "read_file"
    assert start["toolCallId"] == "c1"
    conn.close()


@pytest.mark.store
def test_an_error_ends_the_run_as_run_error(running):
    conn, opened = _run(running, "t7")
    running.render("agui:t7", "error", {"message": "it broke"})
    running.render("agui:t7", "typing", False)
    frames = _events(conn, opened)
    error = next(e for e in frames if e["type"] == "RUN_ERROR")
    assert error["message"] == "it broke"
    conn.close()
