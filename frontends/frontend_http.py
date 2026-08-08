"""Second Brain over HTTP: the whole frontend surface, translated by nothing.

This is a *general-purpose* frontend, not a protocol adapter. It exposes the
two halves a native frontend has and stops there:

  ``GET  /events?thread=X``   every ``render`` the kernel makes, verbatim
  ``POST /sdk/<request.type>?thread=X``   every Request the SDK has

Anything shaped like AG-UI, assistant-ui, or any other client vocabulary is a
layer built *on* this, in the client. That separation is deliberate and was
learned the expensive way: an earlier version of this file spoke AG-UI
natively and spent most of its thousand lines translating — synthesizing
message-start/content/end frames from ``stream_delta``, tracking message ids,
folding approvals and forms into interrupt outcomes — with the result that the
base capability was impossible to reuse and hard to reason about.

--------------------------------------------------------------------------
The outbound half
--------------------------------------------------------------------------

The kernel calls ``render(session_key, kind, payload)`` on a frontend. There
are nine kinds (``messages``, ``attachments``, ``form_field``, ``approval``,
``buttons``, ``error``, ``typing``, ``tool_status``, ``stream_delta``) and each
already crosses to the guest as plain JSON-safe data — ``approval`` in
particular arrives already projected to id/title/body/type/enum/enum_labels/
default, with the kernel's live objects stripped.

So this frontend does no mapping at all. One render, one SSE frame::

    data: {"kind": "stream_delta", "session_key": "http:main", "payload": {…}}

A client that can read those nine can do everything the REPL can.

**The stream is the attendance signal.** Opening it declares that somebody is
watching (``sdk.frontend.attended``); a push that comes back False means the
client hung up, and the session goes unattended again. That matters more than
it looks: attendance is what decides whether an unsafe Request raises a dialog
or is refused outright, so a stream that never noticed its own death would
leave authority switched on with nobody there.

--------------------------------------------------------------------------
The inbound half
--------------------------------------------------------------------------

``POST /sdk/conv.list``, ``POST /sdk/command.call``, ``POST /sdk/config.read``
— the body is the Request's arguments, the answer is its ``Result`` as JSON.
There is no curated list of routes, because curating one is how the previous
version ended up telling clients to *type slash commands at a parser*. A
machine that wants to load conversation 7 should say ``conv.load {"id": 7}``,
not ``/conversations 'Main' 7 'Load conversation'``.

Everything goes through ``sdk.frontend.act``, which runs one Request rooted at
the session rather than at this frontend. Three consequences worth knowing:

* **Unsafe Requests raise a real dialog**, delivered to this same client as an
  ``approval`` frame and answered with ``POST /sdk/frontend.resolve``. Nothing
  is exempted — ``classify`` sees exactly what it would see anywhere else.
* **It is asynchronous.** A box serves one call at a time and the dialog has to
  render back into this box to be seen, so waiting inline would deadlock. The
  answer is collected on a later tick; the client's HTTP request is held open
  until it arrives, so from outside it still looks like one call.
* **The client never states identity.** ``?thread=`` picks the session and the
  kernel supplies this frontend's token for ``frontend.*`` Requests. A token
  or ``session_key`` in the body is ignored.

--------------------------------------------------------------------------
Running it
--------------------------------------------------------------------------

Set ``secret_http_token`` and send it as ``Authorization: Bearer <token>`` on
every request, including static assets. ``/events`` additionally accepts
``?token=``, because the browser's ``EventSource`` cannot send headers and is
the only thing that reconnects on its own.

The kernel binds loopback only; expose it with a tunnel. Set
``http_allowed_origins`` when the app is served from anywhere but this same
port, or the browser refuses the preflight and tells you very little about why.
``http_static_dir`` serves a built app from disk.
"""

import json

from guest.bases import BaseFrontend

# Answered before anything else on a static route. Anything unlisted is served
# as an opaque download rather than guessed at — a mislabelled script is a
# bigger problem than a missing preview.
_TYPES = {"html": "text/html; charset=utf-8", "js": "text/javascript",
          "mjs": "text/javascript", "css": "text/css",
          "json": "application/json", "svg": "image/svg+xml",
          "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
          "gif": "image/gif", "webp": "image/webp", "ico": "image/x-icon",
          "woff": "font/woff", "woff2": "font/woff2", "ttf": "font/ttf",
          "map": "application/json", "txt": "text/plain; charset=utf-8"}

_JSON = {"Content-Type": "application/json"}

# How many frames to hold for a session whose stream is closed. A reconnecting
# client replays from here, so a page refresh does not lose the turn that ran
# while it was reloading — but a client that never comes back must not grow it
# without bound.
_MAX_BUFFERED = 500

# What a failed Request looks like over HTTP. Codes the kernel actually sets;
# anything else is a genuine server-side fault and says so.
_STATUS = {"approval_declined": 403, "not_permitted": 403,
           "not_found": 404, "invalid_argument": 400,
           "unavailable": 503, "timeout": 504, "cancelled": 499}


class HTTP(BaseFrontend):
    """Serves the render stream and the SDK over one loopback port."""

    name = "http"
    description = "HTTP/SSE access to Second Brain, for a web or native app."

    serves_http = 8787
    poll_interval = 0.02
    # Without this ``frontend.submit`` runs the whole agent turn inline and
    # holds the box, so nothing could render while the agent was thinking — and
    # the stream this frontend is serving would carry nothing until the turn
    # was already over.
    background_submit = True
    user_binding = "single"

    capabilities = {
        "supports_typing": True,
        "supports_streaming": True,
        "supports_buttons": True,
        "supports_rich_text": True,
        "supports_inline_forms": True,
        "supports_attachments_out": True,
        "supports_proactive_push": True,
        "max_message_chars": None,
    }

    config_settings = [
        ("HTTP API token", "secret_http_token",
         "Bearer token every request must carry. Generate a long random "
         "string; the app sends it as 'Authorization: Bearer <token>'.", "",
         {"type": "string"}),
        ("HTTP port", "http_port",
         "Port to serve on, loopback only. Expose it with a tunnel.", 8787,
         {"type": "integer"}),
        ("HTTP allowed origins", "http_allowed_origins",
         "Comma-separated origins allowed to call the API from a browser, or "
         "* for any. Needed whenever the app is served from anywhere but this "
         "same port.", "", {"type": "string"}),
        ("HTTP static directory", "http_static_dir",
         "Serve a built web app from this directory. Leave empty to serve the "
         "API only.", "", {"type": "string"}),
    ]

    requests = [
        "http.drain", "http.respond", "http.push", "http.close",
        "frontend.act", "frontend.collect", "frontend.attend",
        "secret.reveal", "config.read", "fs.read_bytes",
    ]

    agent_prompt = (
        "The person may be using a web client. It renders GitHub markdown, "
        "including tables and fenced code, so format normally."
    )

    def __init__(self):
        """Set up per-session bookkeeping."""
        self._token = ""
        self._origins = ""
        self._static = ""
        # session_key -> the http request id of its open event stream.
        self._streams = {}
        # session_key -> [(seq, frame), …] kept for a client that reconnects.
        self._buffered = {}
        # session_key -> how many frames it has ever been sent, which is what
        # ``Last-Event-ID`` counts in.
        self._seq = {}
        # act handle -> the http request id waiting on its answer.
        self._waiting = {}

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle.
    # ──────────────────────────────────────────────────────────────────

    def start(self, sdk):
        """Read settings and return. The kernel already holds the port."""
        # ``secret_*`` reads back as a handle, never plaintext, so this asks
        # for the real thing. Not gated: a plugin reading its own declared
        # setting is not asked, because configuring it *was* the consent —
        # ownership comes from the setting registry, which is a fact about
        # what is installed rather than anything this file can assert.
        try:
            self._token = str(
                sdk.secrets.reveal("secret_http_token") or "").strip()
        except Exception as exc:
            # Starting anyway, with no token, so every request answers 401.
            # Refusing to start would take the port down and leave a bare
            # "refused to start" in the log; a frontend you can curl and be
            # told 401 by is one whose problem you can actually find.
            sdk.log(f"could not read the HTTP token ({exc}); every request "
                    f"will be refused. Is the frontend installed?", "warning")
            self._token = ""
        if not self._token:
            sdk.log("secret_http_token is not set; the HTTP frontend will "
                    "refuse every request. Set it in /config.", "warning")
        self._origins = str(sdk.config.read("http_allowed_origins") or "").strip()
        self._static = str(sdk.config.read("http_static_dir") or "").strip()
        return True

    def stop(self, sdk):
        """Close every stream, so no client is left holding a dead socket."""
        for session_key in list(self._streams):
            self._drop(sdk, session_key)
        return True

    def session_key(self, sdk, ctx):
        """One session per thread.

        ``thread`` is the client's own conversation handle and is opaque to us,
        exactly as a chat id is to Telegram. Keying on it means a client with
        two threads open gets two sessions, which is what makes their
        conversations independent.
        """
        return self._key_for(str((ctx or {}).get("thread") or "default"))

    @staticmethod
    def _key_for(thread: str) -> str:
        """The session key for a thread name.

        Separate from ``session_key`` because that one is the kernel's entry
        point and takes an ``sdk`` this frontend's own routing has no reason to
        be holding. Same answer, and it has to stay the same answer — two
        spellings of one key would put a client's stream and its Requests on
        different sessions.
        """
        return f"http:{thread or 'default'}"

    # ──────────────────────────────────────────────────────────────────
    # The loop.
    # ──────────────────────────────────────────────────────────────────

    def poll(self, sdk):
        """Answer whatever arrived, and deliver whatever finished."""
        worked = self._deliver(sdk)
        arrived = sdk.http.drain()
        if not arrived:
            return worked
        for request in arrived:
            try:
                self._route(sdk, request)
            except Exception as exc:
                # A route that raises must still answer, or the client waits
                # for a reply that is never coming.
                sdk.log(f"HTTP route failed: {exc}", "warning")
                self._reply(sdk, request, 500, {"error": "internal error"})
        return True

    def _deliver(self, sdk):
        """Answer any held request whose Request has finished.

        This is the other half of ``act`` being asynchronous. The client's
        connection was never closed, so from outside it still looks like one
        call that took a moment — which is what it was, except that the moment
        may have included somebody answering a dialog.
        """
        if not self._waiting:
            return False
        worked = False
        for handle, request_id in list(self._waiting.items()):
            outcome = sdk.frontend.collect(handle)
            if outcome is None:
                continue
            del self._waiting[handle]
            worked = True
            self._answer(sdk, request_id, outcome)
        return worked

    def _route(self, sdk, request):
        """One request, dispatched by method and path."""
        path = (request.get("path") or "/").rstrip("/") or "/"
        method = request.get("method") or "GET"

        # Preflight is answered before authentication on purpose: a browser
        # sends no Authorization header on OPTIONS, so checking the token here
        # would refuse every cross-origin request before it was ever made.
        # It reveals nothing — the answer is the same for any origin we allow.
        if method == "OPTIONS":
            return sdk.http.respond(request["id"], status=204,
                                    headers=self._cors())

        if not self._authorized(request, stream=(path == "/events")):
            return self._reply(sdk, request, 401, {"error": "unauthorized"})

        if path == "/events" and method == "GET":
            return self._open_stream(sdk, request)
        if path.startswith("/sdk/") and method == "POST":
            return self._sdk(sdk, request, path[len("/sdk/"):])
        if method == "GET" and self._static:
            return self._file(sdk, request, path)
        return self._reply(sdk, request, 404, {"error": "no such route"})

    # ──────────────────────────────────────────────────────────────────
    # Outbound: the render stream.
    # ──────────────────────────────────────────────────────────────────

    def _open_stream(self, sdk, request):
        """Hold a connection open and send every render for this session.

        One stream per thread. A second one for the same thread replaces the
        first — a reload gets a working stream rather than a 409 nobody can act
        on, and the old socket is almost certainly already dead.
        """
        key = self._session_of(request)
        if key in self._streams:
            self._drop(sdk, key)

        headers = dict(self._cors())
        sdk.http.respond(request["id"], status=200, headers=headers,
                         stream=True)
        self._streams[key] = request["id"]

        # Somebody is watching this session now, which is what lets an unsafe
        # Request raise a dialog instead of being refused with nobody to ask.
        self._attend(sdk, key, True)

        for seq, frame in self._replay(key, request):
            sdk.http.push(request["id"], frame, ident=str(seq))
        return True

    def _replay(self, key, request):
        """Frames this client missed, from its ``Last-Event-ID``.

        A page refresh takes a second or two and the agent does not stop
        talking during it. Without this the turn that ran across the reload is
        simply gone, which looks exactly like the agent having said nothing.
        """
        since = str((request.get("headers") or {}).get("last-event-id") or "")
        if not since:
            since = self._query(request, "since")
        try:
            after = int(since)
        except (TypeError, ValueError):
            return []
        return [(seq, frame) for seq, frame in self._buffered.get(key, [])
                if seq > after]

    def render(self, sdk, session_key, kind, payload):
        """One render, one frame. No translation, on purpose.

        The nine kinds are the kernel's own vocabulary and they are already
        JSON-safe by the time they reach a guest, so a client sees precisely
        what a native frontend would be handed. Anything that wants a different
        shape can build it; nothing has to un-build ours first.
        """
        frame = json.dumps({"kind": kind, "session_key": session_key,
                            "payload": payload})
        seq = self._seq.get(session_key, 0) + 1
        self._seq[session_key] = seq

        held = self._buffered.setdefault(session_key, [])
        held.append((seq, frame))
        if len(held) > _MAX_BUFFERED:
            del held[:-_MAX_BUFFERED]

        request_id = self._streams.get(session_key)
        if request_id is None:
            return True
        try:
            sdk.http.push(request_id, frame, ident=str(seq))
        except Exception:
            # The client hung up. Learned on the write after it went, which is
            # how SSE works and is soon enough — what must not happen is going
            # on believing somebody is there, because attendance is what
            # decides whether an unsafe Request gets a dialog or a refusal.
            self._drop(sdk, session_key)
        return True

    def _drop(self, sdk, session_key):
        """Forget a stream and say nobody is watching that session."""
        request_id = self._streams.pop(session_key, None)
        if request_id is not None:
            try:
                sdk.http.close(request_id)
            except Exception:
                pass    # already gone, which is the case this exists for
        self._attend(sdk, session_key, False)

    def _attend(self, sdk, session_key, present):
        """Declare attendance, tolerating a session the runtime has not made.

        A brand-new thread has no session until something is submitted for it,
        and ``set_session_attended`` is a no-op for a key it does not know. The
        stream opening first is the normal order, so this is expected rather
        than exceptional.
        """
        try:
            sdk.frontend.attended(session_key, present)
        except Exception as exc:
            sdk.log(f"could not set attendance for {session_key}: {exc}",
                    "debug")

    # ──────────────────────────────────────────────────────────────────
    # Inbound: the SDK.
    # ──────────────────────────────────────────────────────────────────

    def _sdk(self, sdk, request, request_type: str):
        """Run one Request as the named session and hold the reply for it.

        The client says *what*; this says *who*, and the kernel decides whether
        that is allowed. Note what is deliberately absent: no allowlist of
        permitted types, no re-implementation of policy, no special cases. A
        Request that would be refused from here is refused by ``classify`` for
        the same reason it would be anywhere else, and one that needs a person
        raises a dialog this same client is about to be shown.
        """
        request_type = request_type.strip("/")
        if not request_type:
            return self._reply(sdk, request, 404,
                               {"error": "name a Request type"})

        key = self._session_of(request)
        args = self._sealed(self._body(request), request_type, key)

        try:
            handle = sdk.frontend.act(key, request_type, args)
        except Exception as exc:
            # A refusal from ``act`` itself — an unknown type, the transport
            # family, somebody else's session. The Request never ran.
            return self._reply(sdk, request, 400, {"error": str(exc)})
        self._waiting[handle] = request["id"]
        return True

    @staticmethod
    def _sealed(args: dict, request_type: str, key: str) -> dict:
        """Strip every way a body could name somebody other than its own thread.

        Identity is ours to state. ``?thread=`` says which session and the
        kernel fills in our desk token, so a body carrying either is claiming
        to be somebody it is not.

        The catch is that "which session" has **two spellings**, and one of
        them is ambiguous. ``frontend.*`` and ``agent.complete`` say
        ``session_key``; the ``session.*`` family says ``key``. But ``key``
        means a *setting name* to ``config.read``, so stripping it everywhere
        would quietly break ordinary reads — hence per-family rather than
        blanket.

        Why it matters: of the eleven session Requests that accept an explicit
        key, only ``session.add_prompt_extra`` compares it against the caller's
        own session. The rest — ``cancel``, ``push``, ``state_set``,
        ``remove_tool``, ``add_attachment`` — are unconditionally safe, so a
        body naming ``telegram:12345`` would reach straight into somebody
        else's session with no dialog and nothing in the way. That asymmetry is
        the kernel's and predates this file; what is new is that a bearer token
        now reaches it, so this is where it gets closed.
        """
        sealed = {name: value for name, value in args.items()
                  if name not in ("token", "session_key")}
        if request_type.startswith("session."):
            sealed["key"] = key
        elif (request_type.startswith("frontend.")
                or request_type == "agent.complete"):
            sealed["session_key"] = key
        return sealed

    def _answer(self, sdk, request_id, outcome):
        """Turn a finished Request's Result into an HTTP reply."""
        if outcome.get("ok"):
            return self._send(sdk, request_id, 200, {"data": outcome.get("data")})
        code = str(outcome.get("code") or "")
        return self._send(sdk, request_id, _STATUS.get(code, 500),
                          {"error": outcome.get("error") or "failed",
                           "code": code})

    # ──────────────────────────────────────────────────────────────────
    # Static files.
    # ──────────────────────────────────────────────────────────────────

    def _file(self, sdk, request, path: str):
        """Serve the built app, if one is configured.

        Every path is resolved against the configured root and anything that
        escapes it is refused. ``fs.read_bytes`` is SAFE, so policy will not
        catch a careless join — this check is the only thing between a URL and
        the rest of the disk.
        """
        relative = path.lstrip("/") or "index.html"
        if ".." in relative.split("/") or relative.startswith("/"):
            return self._reply(sdk, request, 403, {"error": "forbidden"})
        full = f"{self._static.rstrip('/')}/{relative}"
        data = self._read(sdk, full)
        if data is None and "." not in relative.rpartition("/")[2]:
            # A client-side router's route, not a file. Hand back the shell and
            # let the app work out what to draw.
            full = f"{self._static.rstrip('/')}/index.html"
            data = self._read(sdk, full)
        if data is None:
            return self._reply(sdk, request, 404, {"error": "not found"})
        suffix = full.rpartition(".")[2].lower()
        headers = dict(self._cors())
        headers["Content-Type"] = _TYPES.get(suffix,
                                             "application/octet-stream")
        headers["Content-Length"] = str(len(data))
        return sdk.http.respond(request["id"], status=200, headers=headers,
                                body=data)

    @staticmethod
    def _read(sdk, path: str):
        """A file's bytes, or None if it is not there.

        Bytes rather than text for every asset, not just the obviously binary
        ones: a build's fonts and images would be mangled by a UTF-8 decode,
        and deciding per extension would be one more table to get wrong.
        ``http.respond`` takes bytes, so nothing has to decode at all.
        """
        try:
            return sdk.fs.read_bytes(path)
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # Plumbing.
    # ──────────────────────────────────────────────────────────────────

    def _authorized(self, request, stream: bool = False) -> bool:
        """Whether this request carries the configured token.

        Checked on every route including the static one. A token checked on
        some paths is not a perimeter, and the app's own HTML is as much a
        thing worth not serving to strangers as the conversation is.

        ``stream`` allows ``?token=`` as well, and only for ``/events``. The
        browser's ``EventSource`` cannot send headers at all — that is the API,
        not an oversight — and it is the only thing that gives automatic
        reconnection with ``Last-Event-ID``, which is what makes a page refresh
        resume rather than lose the turn it happened during. The alternative is
        a hand-written ``fetch`` reader that reimplements both. Narrow rather
        than general because a token in a URL can reach logs and history: any
        other route wanting one should send the header.
        """
        if not self._token:
            return False
        header = str((request.get("headers") or {}).get("authorization") or "")
        if header.strip() == f"Bearer {self._token}":
            return True
        return bool(stream) and self._query(request, "token") == self._token

    def _cors(self) -> dict:
        """Headers letting a browser on another origin talk to us.

        The kernel adds none, deliberately: which origins may reach a frontend
        is a fact about a deployment. Empty config means same-origin only,
        which is what the static route gives you.
        """
        if not self._origins:
            return {}
        return {"Access-Control-Allow-Origin": self._origins,
                "Access-Control-Allow-Headers":
                    "Content-Type, Authorization, Last-Event-ID",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Max-Age": "86400"}

    @staticmethod
    def _body(request) -> dict:
        """The request's JSON body, or an empty dict."""
        try:
            parsed = json.loads(request.get("body") or "{}")
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _query(request, name: str) -> str:
        """One query parameter, or an empty string."""
        for pair in (request.get("query") or "").split("&"):
            key, _, value = pair.partition("=")
            if key == name and value:
                return value
        return ""

    def _session_of(self, request) -> str:
        """Which session a request is about, from ``?thread=``."""
        return self._key_for(self._query(request, "thread"))

    def _send(self, sdk, request_id, status: int, payload):
        """One JSON answer, with CORS."""
        headers = dict(self._cors())
        headers.update(_JSON)
        return sdk.http.respond(request_id, status=status, headers=headers,
                                body=json.dumps(payload))

    def _reply(self, sdk, request, status: int, payload):
        """One JSON answer to a request still in hand."""
        return self._send(sdk, request["id"], status, payload)
