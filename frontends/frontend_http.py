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
are twelve kinds (``messages``, ``attachments``, ``form_field``, ``approval``,
``approval_settled``, ``buttons``, ``error``, ``typing``, ``tool_status``,
``stream_delta``, ``notification``, ``callable_output``) and each
already crosses to the guest as plain JSON-safe data — ``approval`` in
particular arrives already projected to id/title/body/type/enum/enum_labels/
default, with the kernel's live objects stripped.

So this frontend does no mapping at all. One render, one SSE frame::

    data: {"kind": "stream_delta", "session_key": "http:main", "payload": {…}}

A client that can read those ten can do everything the REPL can.

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

There is no upload route, and there is deliberately no need for one: a file
becomes a *path* first (``fs.temp`` for scratch, then ``fs.write_bytes``, in
``append`` chunks for anything larger than one wire message), and the path is
what ``frontend.submit`` carries. Scratch is a safe write, so none of it raises
a dialog, and ``ingest`` moves the file into the attachment cache on the way
in.

**Several files go in one submit** — ``input_kind: "attachment"`` with
``files: [{path, file_name, …}]`` and the message's own ``caption``. Not three
submits: a submitted attachment hands the turn to the agent, so the second
would arrive at a session that is already busy and come back ``busy``, which
is what made a transport with a file picker a one-file transport. One submit
is one action, one turn, and every file in the same model call.

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
``http_static_dir`` serves a built app from disk, and ``GET /files?path=`` one
file from anywhere the kernel will allow — the transport a client needs to put
a host file in an ``<img>``, a ``<video>`` or an ``<audio>``, since a Request
answers base64 inside JSON and those want a URL.
"""

import json
import mimetypes
from urllib.parse import unquote

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
          "map": "application/json", "txt": "text/plain; charset=utf-8",
          # Types a built app never contains but a *host file* often is. These
          # are what ``GET /files`` exists for: name the type and the browser
          # renders it natively, which is the whole reason not to hand a client
          # base64 and make it assemble a Blob.
          "pdf": "application/pdf", "md": "text/plain; charset=utf-8",
          "csv": "text/csv; charset=utf-8", "bmp": "image/bmp",
          "avif": "image/avif", "tif": "image/tiff", "tiff": "image/tiff",
          "mp4": "video/mp4", "webm": "video/webm", "mov": "video/quicktime",
          "mkv": "video/x-matroska", "mp3": "audio/mpeg", "wav": "audio/wav",
          "ogg": "audio/ogg", "oga": "audio/ogg", "m4a": "audio/mp4",
          "flac": "audio/flac", "opus": "audio/opus",
          # The rest of what the kernel's modality map names. `mimetypes` reads
          # the *host's* mime database, so what it knows differs between a dev
          # box and the machine this actually runs on; these are stated so the
          # answer is the same on both. An element handed
          # `application/octet-stream` refuses to play whatever the bytes are.
          "heic": "image/heic", "heif": "image/heif", "avi": "video/x-msvideo",
          "aac": "audio/aac", "wma": "audio/x-ms-wma"}

# The most a single ``fs.read_bytes`` answer can carry, mirroring the kernel's
# ``MAX_READ_BINARY`` — one wire message less a megabyte of headroom, times
# three quarters for base64. Windowed reads are what let a 200 MB video be
# served at all, and what makes a Range request cheap: a seek asks for the
# bytes it lands on rather than everything before them.
_WINDOW = (16 * 1024 * 1024 - 1024 * 1024) * 3 // 4


def _content_type(path: str) -> str:
    """What to label a host file, so the browser decodes it natively.

    ``_TYPES`` first, because :mod:`mimetypes` reads the *host's* database and
    therefore answers differently on different machines — a built app's assets
    and anything the kernel can name a modality for must be labelled the same
    way everywhere. ``mimetypes`` then covers the long tail, which is what
    keeps this table from having to grow a row per format.

    Two tables answering one question is how they drift, so the fallback is a
    library rather than more rows here — and
    ``test_every_native_modality_gets_a_playable_type`` checks the pair against
    ``parsing._NATIVE_DEFAULTS`` so a modality the kernel recognises can never
    arrive unlabelled.

    Bytes are served for **every** extension either way; this only decides the
    label, and an unknown one is an honest download rather than a guess.
    """
    guessed = _TYPES.get(path.rpartition(".")[2].lower())
    if guessed:
        return guessed
    return mimetypes.guess_type(path)[0] or "application/octet-stream"

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
        # Both directions. Inbound is not a route of ours — a client writes
        # the bytes to scratch through ``fs.write_bytes`` and submits the
        # path — but a capability nobody declares is one a client has no way
        # to discover, and this transport has a file picker behind it.
        "supports_attachments_in": True,
        "supports_attachments_out": True,
        "supports_proactive_push": True,
        # A client here has somewhere to put a notification that is not the
        # transcript, which is the whole reason the kind exists. Without this
        # the kernel flattens each one into markdown and sends it as
        # ``messages`` — the pre-notification behaviour, and right for a
        # transport whose only surface is a chat log. It is wrong for this one,
        # and wrong invisibly: a plugin registration would arrive looking like
        # something a person said.
        "supports_notifications": True,
        # And the same argument for what a slash command answered with. A
        # `/config` listing and the agent's reply are not the same kind of
        # thing, and a client drawing a conversation could not tell them apart
        # while both arrived as ``messages``.
        "supports_callable_output": True,
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
        "secret.reveal", "config.read", "fs.read_bytes", "fs.stat",
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

        if not self._authorized(request,
                                query_token=path in ("/events", "/files")):
            return self._reply(sdk, request, 401, {"error": "unauthorized"})

        if path == "/events" and method == "GET":
            return self._open_stream(sdk, request)
        if path.startswith("/sdk/") and method == "POST":
            return self._sdk(sdk, request, path[len("/sdk/"):])
        if path == "/files" and method in ("GET", "HEAD"):
            return self._host_file(sdk, request, head=(method == "HEAD"))
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

        The twelve kinds are the kernel's own vocabulary and they are already
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

        # A stream normally opens before its session exists. The first
        # attendance declaration is therefore intentionally a no-op in the
        # runtime; once an earlier Request (usually ``conv.create``) has made
        # the session, refresh that declaration before acting again. Without
        # this, the first unsafe click after boot is treated as unattended and
        # refused, even though this exact stream is waiting to show its dialog.
        if key in self._streams:
            self._attend(sdk, key, True)

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

    # ──────────────────────────────────────────────────────────────────
    # Host files.
    # ──────────────────────────────────────────────────────────────────

    def _host_file(self, sdk, request, head: bool = False):
        """Serve one file from the host by absolute path.

        The point is the *transport*, not new authority. A client can already
        read any file through ``POST /sdk/fs.read_bytes``; what it cannot do is
        hand the result to an ``<img>``, a ``<video>`` or an ``<audio>``
        element, because a Request answers base64 inside JSON and those want a
        URL. Assembling a Blob works for a picture and is hopeless for media:
        it buffers the whole file before the first frame and cannot seek.

        So this reads through exactly the same ``fs.read_bytes`` — same
        classification, same ledger row, a refusal still refused — and spends
        the answer on a real HTTP body with a real ``Content-Type``. What that
        buys is native decoding, browser caching, and Range requests.

        **Every read is still policy's to allow.** ``fs.read`` is safe except
        for protected files, so the kernel already refuses Second Brain's own
        source and data directories; this route inherits that and adds no
        exemption of its own.
        """
        path = unquote(self._query(request, "path"))
        if not path:
            return self._reply(sdk, request, 400,
                               {"error": "GET /files requires ?path="})

        info = self._stat(sdk, path)
        if info is None:
            return self._reply(sdk, request, 404, {"error": "not found"})
        if not info.get("is_file"):
            return self._reply(sdk, request, 400, {"error": "not a file"})
        size = int(info.get("size") or 0)

        headers = dict(self._cors())
        headers["Content-Type"] = _content_type(path)
        # Advertised unconditionally: a media element decides whether to seek
        # by looking for this before it asks for anything.
        headers["Accept-Ranges"] = "bytes"

        window = self._wanted(request, size)
        if window is None:
            headers["Content-Range"] = f"bytes */{size}"
            return sdk.http.respond(request["id"], status=416,
                                    headers=headers, body=b"")
        start, end = window
        # Clamped whether or not a Range was asked for, because one response
        # body has to cross the wire in one message — and if this frontend is
        # ever subprocessed, that message is capped. So a large file always
        # comes back as 206 and the client asks for the rest; a media element
        # does that by itself, which is the case this route exists for.
        end = min(end, start + _WINDOW - 1)
        partial = (start, end) != (0, size - 1) and size > 0

        if head:
            headers["Content-Length"] = str(size)
            return sdk.http.respond(request["id"], status=200,
                                    headers=headers, body=b"")

        data = self._read_span(sdk, path, start, end - start + 1)
        if data is None:
            return self._reply(sdk, request, 403, {"error": "not readable"})

        headers["Content-Length"] = str(len(data))
        if partial:
            headers["Content-Range"] = (
                f"bytes {start}-{start + len(data) - 1}/{size}")
        return sdk.http.respond(request["id"], status=206 if partial else 200,
                                headers=headers, body=data)

    @staticmethod
    def _wanted(request, size: int):
        """``(start, end)`` inclusive for this request, or ``None`` if
        unsatisfiable.

        Only the single-range form, which is the only one a media element
        sends. A multipart range would need a different body shape for a case
        that does not arise, so an unreadable header is treated as no header
        at all — serving the whole file is always a valid answer to a Range
        request, where guessing at one is not.
        """
        raw = str((request.get("headers") or {}).get("range") or "").strip()
        whole = (0, max(0, size - 1))
        if not raw.startswith("bytes=") or "," in raw:
            return whole
        first, _, last = raw[len("bytes="):].partition("-")
        try:
            if not first:                     # bytes=-500: the final 500
                length = int(last)
                if length <= 0:
                    return whole
                return (max(0, size - length), size - 1)
            start = int(first)
            end = int(last) if last else size - 1
        except ValueError:
            return whole
        if start >= size or start > end:
            return None
        return (start, min(end, size - 1))

    def _read_span(self, sdk, path: str, offset: int, length: int):
        """``length`` bytes from ``offset``, or None if the file will not open.

        One ``fs.read_bytes`` answer is capped, so a long span arrives as
        several and is joined here. A short read means end of file, which is
        the documented way to stop without having to trust the size first —
        so a file being written underneath this returns what exists rather
        than hanging or lying about the length.
        """
        chunks, taken = [], 0
        while taken < length:
            piece = self._read(sdk, path, offset + taken,
                               min(_WINDOW, length - taken))
            if piece is None:
                return None if not chunks else b"".join(chunks)
            if not piece:
                break
            chunks.append(piece)
            taken += len(piece)
        return b"".join(chunks)

    @staticmethod
    def _stat(sdk, path: str):
        """Metadata for one path, or None if it is missing or refused."""
        try:
            return sdk.fs.stat(path)
        except Exception:
            return None

    @staticmethod
    def _read(sdk, path: str, offset: int = 0, length: int = 0):
        """A file's bytes, or None if it is not there.

        Bytes rather than text for every asset, not just the obviously binary
        ones: a build's fonts and images would be mangled by a UTF-8 decode,
        and deciding per extension would be one more table to get wrong.
        ``http.respond`` takes bytes, so nothing has to decode at all.

        ``offset``/``length`` are how a span longer than one answer is read;
        the static path passes neither and is unchanged.
        """
        try:
            return sdk.fs.read_bytes(path, offset=offset, length=length)
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # Plumbing.
    # ──────────────────────────────────────────────────────────────────

    def _authorized(self, request, query_token: bool = False) -> bool:
        """Whether this request carries the configured token.

        Checked on every route including the static one. A token checked on
        some paths is not a perimeter, and the app's own HTML is as much a
        thing worth not serving to strangers as the conversation is.

        ``query_token`` allows ``?token=`` as well, and the rule for granting
        it is narrow and mechanical: **the browser issues the request itself
        and cannot be given a header.** Two routes qualify and no others.

        ``/events`` is one — ``EventSource`` cannot send headers at all, and it
        is the only thing that gives automatic reconnection with
        ``Last-Event-ID``, which is what makes a page refresh resume rather
        than lose the turn it happened during.

        ``/files`` is the other, for exactly the same reason one step over: an
        ``<img>``, ``<video>`` or ``<audio>`` element fetches its own ``src``
        and there is nowhere to put a header. Without this the route could only
        be reached by ``fetch``, which means rebuilding a Blob — which is the
        thing it exists to avoid, and which cannot seek.

        A token in a URL can reach logs and history, so this stays a list of
        two rather than a general option. Any route a *script* calls should
        send the header.
        """
        if not self._token:
            return False
        header = str((request.get("headers") or {}).get("authorization") or "")
        if header.strip() == f"Bearer {self._token}":
            return True
        return (bool(query_token)
                and self._query(request, "token") == self._token)

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
