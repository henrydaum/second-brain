"""AG-UI frontend — the whole connection between Second Brain and its app.

**This file is the entire API.** Nothing else in the system is reachable from
the client: no second endpoint, no direct database, no kernel import. What is
not exposed here, the app cannot do. That is the point of speaking a protocol
rather than a private wire — any AG-UI client connects, and Second Brain never
learns which one.

**The loop does not invert here, and that is the whole difference from
Telegram.** `frontend_telegram` had to slice an asyncio loop out of `poll`
because python-telegram-bot owns a thread and a socket. Here the kernel owns
both (`sandbox/http_server.py`): `poll` drains parsed requests, answers them,
and returns. The guest holds no loop, no socket and nothing thread-affine —
which is why this file is stdlib-only and why it does not structurally require
subprocess isolation the way Telegram does.

**Reads are an API; writes are a typed command.** A frontend's chain roots at
its session key, so ``Chain.attended`` is false and an unsafe Request is
*refused rather than asked* — there is nobody to show a dialog to. So
``conv.delete``, ``config.write`` and ``command.call`` would fail silently if
called from here, while ``conv.read``, ``command.list`` and ``config.read`` are
SAFE and serve fine. Anything mutating therefore goes back through
``submit_text("/command …")``, which is the only path ``bridge._root_for``
grants ``user:command`` — earning the command's declared approval gate and a
real dialog. The app inherits every guarantee the REPL has without restating
one of them.

**A run ends when typing stops.** AG-UI is request-scoped — one POST, an SSE
stream, ``RUN_FINISHED`` — but ``background_submit`` returns the moment the
turn is queued and renders arrive later from another thread. ``render("typing",
False)`` is the signal: ``BaseFrontend.on_bus_session_turn_changed`` fires it
when turn *priority* hands back to the user, deliberately not per-drive, so a
barrier-held turn or an escalation re-drive keeps it on until the **logical**
turn ends and a crash clears it too. ``supports_typing`` must therefore stay
True below, or no stream ever closes.

The mapping, which is the contract a client is written against:

===================  ======================================================
render kind          AG-UI event
===================  ======================================================
``typing`` True      ``RUN_STARTED``
``stream_delta``     ``TEXT_MESSAGE_START`` / ``_CONTENT`` / ``_END``
``messages``         the same three, for text that never streamed
``tool_status``      ``TOOL_CALL_START`` / ``_ARGS`` / ``_RESULT``
``error``            ``RUN_ERROR``
``typing`` False     ``RUN_FINISHED``, then the stream closes
``approval``         ``CUSTOM`` — ``second_brain.approval``
``form_field``       ``CUSTOM`` — ``second_brain.form``
``buttons``          ``CUSTOM`` — ``second_brain.buttons``
``attachments``      ``CUSTOM`` — ``second_brain.attachments``
===================  ======================================================

The last four are ``CUSTOM`` because AG-UI standardises the *conversation* and
has no vocabulary for an approval dialog or a form. ``INTERRUPT`` is reported
by some write-ups and absent from the protocol's own event reference; guessing
wrong means a client silently ignoring every approval, so these stay CUSTOM
until the pinned SDK says otherwise.

Text crosses **as markdown, verbatim**. Telegram needed a converter because its
API wants HTML; assistant-ui renders markdown natively, so the kernel's own
output format is already the right one.
"""

dependencies_pip = []
requests = [
    "http.drain", "http.respond", "http.push", "http.close",
    "frontend.submit", "frontend.pending", "frontend.resolve",
    "secret.reveal", "config.read", "command.list",
    "conv.list", "conv.read", "conv.create", "conv.load",
    "session.get", "user.read", "fs.read_bytes",
]

import json
import time
import uuid

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

# How many events to hold for a session with no open stream before dropping the
# oldest. A background turn that nobody is watching still produced something,
# and losing it silently is worse than any other failure here — but a client
# that never comes back must not grow this without bound.
_MAX_BUFFERED = 500


class AGUI(BaseFrontend):
    """Serves AG-UI over SSE, plus the read API the protocol has no words for."""

    name = "agui"
    description = "AG-UI endpoint for assistant-ui and other AG-UI clients."

    serves_http = 8787
    poll_interval = 0.02
    # Without this ``submit_text`` runs the whole agent turn inline and holds
    # the box, so nothing could render while the agent was thinking — and the
    # SSE stream this frontend just opened would carry nothing until the turn
    # was already over.
    background_submit = True
    user_binding = "single"

    capabilities = {
        # Load-bearing rather than cosmetic: ``BaseFrontend._route_typing``
        # returns early without it, so the turn-end signal never arrives and
        # every stream stays open forever. See the module docstring.
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
        ("AG-UI API token", "secret_agui_token",
         "Bearer token every request must carry. Generate a long random "
         "string; the app sends it as 'Authorization: Bearer <token>'.", "",
         {"type": "string"}),
        ("AG-UI port", "agui_port",
         "Port to serve on, loopback only. Expose it with a tunnel.", 8787,
         {"type": "integer"}),
        ("AG-UI allowed origins", "agui_allowed_origins",
         "Comma-separated origins allowed to call the API from a browser, or "
         "* for any. Needed whenever the app is served from anywhere but this "
         "same port.", "", {"type": "string"}),
        ("AG-UI static directory", "agui_static_dir",
         "Serve a built web app from this directory. Leave empty to serve the "
         "API only.", "", {"type": "string"}),
    ]

    agent_prompt = (
        "The person may be using a web client (AG-UI). It renders GitHub "
        "markdown, including tables and fenced code, so format normally."
    )

    def __init__(self):
        """Set up per-session bookkeeping."""
        self._token = ""
        self._origins = ""
        self._static = ""
        # session_key -> the stream currently carrying its run.
        self._runs = {}
        # session_key -> events produced with no stream open to carry them.
        self._pending = {}
        # (session_key, stream_id) -> True once TEXT_MESSAGE_START was sent.
        self._streaming = {}
        # Text already delivered as a stream, so the whole-message render that
        # follows it is not shown twice.
        self._streamed = {}
        # The last approval each session was shown. Whether one is still
        # *pending* is asked, never remembered — it can be answered elsewhere
        # or time out.
        self._approvals = {}

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
                sdk.secrets.reveal("secret_agui_token") or "").strip()
        except Exception as exc:
            # Starting anyway, with no token, so every request answers 401.
            # Refusing to start would take the port down and leave a bare
            # "refused to start" in the log; a frontend you can curl and be
            # told 401 by is one whose problem you can actually find.
            sdk.log(f"AG-UI could not read its token ({exc}); every request "
                    f"will be refused. Is the frontend installed?", "warning")
            self._token = ""
        if not self._token:
            sdk.log("secret_agui_token is not set; the AG-UI frontend will "
                    "refuse every request. Set it in /config.", "warning")
        self._origins = str(sdk.config.read("agui_allowed_origins") or "").strip()
        self._static = str(sdk.config.read("agui_static_dir") or "").strip()
        return True

    def stop(self, sdk):
        """Close anything still streaming, so no client is left hanging."""
        for session_key in list(self._runs):
            self._finish(sdk, session_key)
        return True

    def session_key(self, ctx):
        """One session per AG-UI thread.

        ``threadId`` is the protocol's conversation handle and it is opaque to
        us, exactly as a chat id is to Telegram. Keying on it means a client
        with two threads open gets two sessions, which is what makes their
        conversations independent.
        """
        thread = str((ctx or {}).get("thread") or "default")
        return f"agui:{thread}"

    # ──────────────────────────────────────────────────────────────────
    # The loop.
    # ──────────────────────────────────────────────────────────────────

    def poll(self, sdk):
        """Answer whatever arrived. Never blocks."""
        arrived = sdk.http.drain()
        if not arrived:
            return False
        for request in arrived:
            try:
                self._route(sdk, request)
            except Exception as exc:
                # A route that raises must still answer, or the client waits
                # for a reply that is never coming.
                sdk.log(f"AG-UI route failed: {exc}", "warning")
                self._reply(sdk, request, 500, {"error": "internal error"})
        return True

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

        if not self._authorized(request):
            return self._reply(sdk, request, 401, {"error": "unauthorized"})

        if path == "/agui" and method == "POST":
            return self._run(sdk, request)
        if path == "/conversations" and method == "GET":
            return self._reply(sdk, request, 200,
                               {"conversations": sdk.conv.list(details=True)})
        if path == "/conversations" and method == "POST":
            body = self._body(request)
            return self._reply(sdk, request, 200, {
                "conversation": sdk.conv.create(
                    title=str(body.get("title") or ""),
                    activate=bool(body.get("activate")))})
        if path.startswith("/conversations/"):
            return self._conversation(sdk, request, path, method)
        if path == "/commands" and method == "GET":
            return self._reply(sdk, request, 200, {
                "commands": sdk.commands.list(details=True, visible=True)})
        if path == "/config" and method == "GET":
            return self._reply(sdk, request, 200,
                               {"config": sdk.config.read()})
        if path == "/session" and method == "GET":
            key = self._session_of(request)
            return self._reply(sdk, request, 200, {
                "session": sdk.session.get(key, details=True),
                "user": sdk.users.read()})
        if method == "GET" and self._static:
            return self._file(sdk, request, path)
        return self._reply(sdk, request, 404, {"error": "no such route"})

    def _conversation(self, sdk, request, path: str, method: str):
        """``/conversations/<id>`` and ``/conversations/<id>/load``.

        Deleting is deliberately absent: ``conv.delete`` is UNSAFE and this
        chain is unattended, so calling it would be refused rather than asked —
        silently. ``/conversations delete`` submitted as text is the route that
        works, and it asks the person first.
        """
        rest = path[len("/conversations/"):]
        ident, _, action = rest.partition("/")
        try:
            conversation_id = int(ident)
        except ValueError:
            return self._reply(sdk, request, 400,
                               {"error": "conversation id must be a number"})
        if action == "load" and method == "POST":
            sdk.conv.load(conversation_id)
            return self._reply(sdk, request, 200, {"loaded": conversation_id})
        if not action and method == "GET":
            return self._reply(sdk, request, 200, {
                "conversation": sdk.conv.read(conversation_id, details=True)})
        return self._reply(sdk, request, 404, {"error": "no such route"})

    # ──────────────────────────────────────────────────────────────────
    # The AG-UI run.
    # ──────────────────────────────────────────────────────────────────

    def _run(self, sdk, request):
        """Open a stream for one run and hand the text to the state machine.

        The reply is opened *before* submitting, so the events a fast turn
        produces have somewhere to land. Submitting first is a race the
        buffering below would paper over but which would still cost the first
        token of every reply.
        """
        body = self._body(request)
        thread = str(body.get("threadId") or "default")
        run_id = str(body.get("runId") or uuid.uuid4().hex)
        session_key = self.session_key({"thread": thread})
        text = self._last_user_message(body)

        # One run at a time per session. A second POST while one is live is a
        # client bug, but abandoning the first stream silently would present as
        # a reply that simply stopped, so the old one is closed properly.
        if session_key in self._runs:
            self._finish(sdk, session_key)

        sdk.http.respond(request["id"], stream=True, headers=self._cors())
        self._runs[session_key] = {"request_id": request["id"],
                                   "thread": thread, "run": run_id}
        self._emit(sdk, session_key, "RUN_STARTED",
                   {"threadId": thread, "runId": run_id})
        # Anything produced while no stream was open — a scheduled subagent
        # finishing, a pushed notification — is delivered before this run's own
        # events, in the order it happened.
        for event in self._pending.pop(session_key, []):
            self._push(sdk, session_key, event)

        if not text:
            # Nothing to say: a client opening a stream to collect buffered
            # events is legitimate, and it should get them and a clean end
            # rather than an empty turn.
            self._finish(sdk, session_key)
            return True
        if not self._absorb_approval(sdk, session_key, text):
            sdk.frontend.submit_text(session_key, text)
        return True

    @staticmethod
    def _last_user_message(body) -> str:
        """The text this run is about.

        AG-UI resends the whole thread on every run; Second Brain keeps its own
        history, so only the newest user turn is new information. Taking the
        last *user* entry rather than the last entry skips a trailing assistant
        message a client may have optimistically appended.
        """
        for message in reversed(list(body.get("messages") or [])):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, list):
                    # The multi-part shape: keep the text parts.
                    return "".join(str(part.get("text") or "")
                                   for part in content
                                   if isinstance(part, dict)).strip()
                return str(content or "").strip()
        return ""

    def _absorb_approval(self, sdk, session_key: str, text: str) -> bool:
        """Answer a pending approval by typed text. True if it was consumed.

        Whether an approval is still pending is *asked* every time rather than
        remembered: it can be answered from another frontend or time out, and
        acting on a stale record means swallowing an ordinary message as a yes.
        """
        request_id = sdk.frontend.pending_approval(session_key)
        if not request_id:
            return False
        value = self._approval_value(session_key, text)
        if value is None:
            return False
        return bool(sdk.frontend.resolve(session_key, value, str(request_id)))

    def _approval_value(self, session_key: str, answer: str):
        """What an answer means, in the shape the waiting frame accepts.

        An ``enum`` request's choices go back **verbatim** — coercing them is
        what once made every sandbox Request dialog unanswerable by button, as
        ``frontend_telegram._approval_value`` records at length. Only the
        boolean fallback spells things ``allow``/``deny``, and only that frame
        wants a bool.
        """
        request = self._approvals.get(session_key) or {}
        if request.get("enum"):
            return answer if answer in [str(v) for v in request["enum"]] else None
        lowered = answer.strip().lower()
        if lowered in ("yes", "y", "allow", "approve", "true", "ok"):
            return True
        if lowered in ("no", "n", "deny", "reject", "false", "cancel"):
            return False
        return None

    # ──────────────────────────────────────────────────────────────────
    # Rendering — one render in, zero or more AG-UI events out.
    # ──────────────────────────────────────────────────────────────────

    def render(self, sdk, session_key: str, kind: str, payload):
        """Show one thing to whoever is watching this session."""
        if kind == "typing":
            if payload:
                return          # RUN_STARTED was sent when the run opened.
            return self._finish(sdk, session_key)
        if kind == "stream_delta":
            return self._delta(sdk, session_key, payload or {})
        if kind == "messages":
            for text in payload or []:
                if text and not self._already_streamed(session_key, text):
                    self._whole_message(sdk, session_key, str(text))
            return None
        if kind == "tool_status":
            return self._tool(sdk, session_key, payload or {})
        if kind == "error":
            message = (payload.get("message") if isinstance(payload, dict)
                       else payload)
            return self._emit(sdk, session_key, "RUN_ERROR",
                              {"message": str(message or "error")})
        if kind == "approval":
            self._approvals[session_key] = dict(payload or {})
            return self._custom(sdk, session_key, "approval", payload)
        if kind in ("form_field", "buttons", "attachments"):
            return self._custom(sdk, session_key, kind.replace("_field", ""),
                                payload)
        return None

    def _delta(self, sdk, session_key: str, payload: dict):
        """Streamed assistant text, as an AG-UI message in three acts."""
        stream_id = str(payload.get("stream_id") or "")
        marker = (session_key, stream_id)
        if payload.get("done"):
            if self._streaming.pop(marker, None):
                self._emit(sdk, session_key, "TEXT_MESSAGE_END",
                           {"messageId": stream_id})
            final = payload.get("final_text")
            if final and not payload.get("aborted"):
                # Remembered so the whole-message render that follows a
                # completed stream is not shown a second time.
                self._streamed.setdefault(session_key, []).append(str(final))
            return None
        delta = payload.get("delta") or ""
        if not delta:
            return None
        if not self._streaming.get(marker):
            self._streaming[marker] = True
            self._emit(sdk, session_key, "TEXT_MESSAGE_START",
                       {"messageId": stream_id, "role": "assistant"})
        return self._emit(sdk, session_key, "TEXT_MESSAGE_CONTENT",
                          {"messageId": stream_id, "delta": str(delta)})

    def _whole_message(self, sdk, session_key: str, text: str):
        """A message that never streamed, as one complete AG-UI message."""
        message_id = uuid.uuid4().hex
        self._emit(sdk, session_key, "TEXT_MESSAGE_START",
                   {"messageId": message_id, "role": "assistant"})
        self._emit(sdk, session_key, "TEXT_MESSAGE_CONTENT",
                   {"messageId": message_id, "delta": text})
        return self._emit(sdk, session_key, "TEXT_MESSAGE_END",
                          {"messageId": message_id})

    def _already_streamed(self, session_key: str, text: str) -> bool:
        """Whether this exact text already went out as a stream."""
        seen = self._streamed.get(session_key) or []
        if str(text).strip() in [s.strip() for s in seen]:
            seen.remove(next(s for s in seen if s.strip() == str(text).strip()))
            return True
        return False

    def _tool(self, sdk, session_key: str, payload: dict):
        """A tool call, as AG-UI's three-part tool vocabulary."""
        call_id = str(payload.get("call_id") or uuid.uuid4().hex)
        status = payload.get("status")
        if status == "started":
            self._emit(sdk, session_key, "TOOL_CALL_START", {
                "toolCallId": call_id,
                "toolCallName": str(payload.get("tool_name")
                                    or payload.get("name") or "tool")})
            return self._emit(sdk, session_key, "TOOL_CALL_ARGS", {
                "toolCallId": call_id,
                "delta": json.dumps(payload.get("args") or {})})
        if status == "finished":
            self._emit(sdk, session_key, "TOOL_CALL_END",
                       {"toolCallId": call_id})
            return self._emit(sdk, session_key, "TOOL_CALL_RESULT", {
                "toolCallId": call_id, "messageId": uuid.uuid4().hex,
                "content": str(payload.get("result")
                               or payload.get("narration") or "")})
        # "progressed", and anything a later kernel adds: pass it through
        # rather than dropping it, so a client can show it if it knows how.
        return self._custom(sdk, session_key, "tool_progress", payload)

    def _custom(self, sdk, session_key: str, name: str, payload):
        """One of the four things AG-UI has no vocabulary for."""
        return self._emit(sdk, session_key, "CUSTOM",
                          {"name": f"second_brain.{name}", "value": payload})

    # ──────────────────────────────────────────────────────────────────
    # The stream underneath.
    # ──────────────────────────────────────────────────────────────────

    def _emit(self, sdk, session_key: str, event_type: str, fields: dict):
        """Build one AG-UI event and get it to the client, or hold it."""
        event = dict(fields)
        event["type"] = event_type
        event["timestamp"] = int(time.time() * 1000)
        return self._push(sdk, session_key, event)

    def _push(self, sdk, session_key: str, event: dict):
        """Send one event, buffering it when no stream is open.

        A turn nobody is watching still produced something — a scheduled
        subagent's report, a pushed notification — and AG-UI has no side
        channel to deliver it on. Dropping it is silent and loses work the
        person asked for, so it waits for the next run.
        """
        run = self._runs.get(session_key)
        if run is None:
            held = self._pending.setdefault(session_key, [])
            held.append(event)
            while len(held) > _MAX_BUFFERED:
                held.pop(0)
            return False
        if not sdk.http.push(run["request_id"], json.dumps(event)):
            # The client went away mid-run. Ordinary, and the kernel tells us
            # so rather than letting a whole turn render into a closed socket.
            self._runs.pop(session_key, None)
            return False
        return True

    def _finish(self, sdk, session_key: str):
        """End the run this session is streaming, if any."""
        run = self._runs.get(session_key)
        if run is None:
            return None
        self._emit(sdk, session_key, "RUN_FINISHED",
                   {"threadId": run["thread"], "runId": run["run"]})
        self._runs.pop(session_key, None)
        self._streamed.pop(session_key, None)
        for marker in [m for m in self._streaming if m[0] == session_key]:
            self._streaming.pop(marker, None)
        sdk.http.close(run["request_id"])
        return None

    # ──────────────────────────────────────────────────────────────────
    # Serving files, and the plumbing.
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

    def _authorized(self, request) -> bool:
        """Whether this request carries the configured token.

        Checked on every route including the static one. A token checked on
        some paths is not a perimeter, and the app's own HTML is as much a
        thing worth not serving to strangers as the conversation is.
        """
        if not self._token:
            return False
        header = str((request.get("headers") or {}).get("authorization") or "")
        return header.strip() == f"Bearer {self._token}"

    def _cors(self) -> dict:
        """Headers letting a browser on another origin talk to us.

        The kernel adds none, deliberately: which origins may reach a frontend
        is a fact about a deployment. Empty config means same-origin only,
        which is what the static route gives you.
        """
        if not self._origins:
            return {}
        return {"Access-Control-Allow-Origin": self._origins,
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
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

    def _session_of(self, request) -> str:
        """Which session a query is about, from ``?thread=``."""
        query = request.get("query") or ""
        for pair in query.split("&"):
            key, _, value = pair.partition("=")
            if key == "thread" and value:
                return self.session_key({"thread": value})
        return self.session_key({"thread": "default"})

    def _reply(self, sdk, request, status: int, payload):
        """One JSON answer, with CORS."""
        headers = dict(self._cors())
        headers.update(_JSON)
        return sdk.http.respond(request["id"], status=status, headers=headers,
                                body=json.dumps(payload))
