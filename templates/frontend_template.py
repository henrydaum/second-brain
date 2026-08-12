"""
FRONTEND TEMPLATE
=================
Reference for writing a frontend against the SDK. Not imported by the running
system — it exists to be read.

Read docs/SDK.md for the Request surface and sandbox/guest/bases.py for every
attribute and method BaseFrontend defines. What follows is only what is
specific to frontends, and most of it is not guessable.

Before writing: read docs/SDK.md, then this entire template. For details not
defined here, inspect sandbox/guest/bases.py (BaseFrontend declarations),
sandbox/frontends.py (the host bridge), sandbox/guest/sdk.py (frontend
Requests), and state_machine/form_display.py (render payloads). Validate the
finished file before enabling it.

  Where it goes:  DATA_DIR/workspace/frontends/frontend_<name>.py
                  (store packages install under DATA_DIR/installed/frontends/)
  Entry points:   start / poll / stop / render / session_key
  Gets:           sdk — never a context
  Box:            persistent, by definition. It is loaded once and stays.

A frontend is a transport: a terminal, a chat network, a socket, an HTTP
surface. Prefer a command, tool, or task for app behaviour; write a frontend
only when you are adding a new way for a person to reach the system.


THE LOOP INVERTS — THE ONE THING THAT SURPRISES PEOPLE
------------------------------------------------------
A native frontend blocks in start() forever, reading its transport. **You
cannot do that here.** A box takes one call at a time, so code that never
returns from start() holds the box and no render() can get in — the frontend
would go deaf the moment it started listening.

So the kernel drives:

    start(sdk)   open the transport and RETURN. Setup only.
    poll(sdk)    called over and over. Take what is waiting, submit it, return.
    stop(sdk)    close the transport.

poll() must return promptly. Between polls is the only moment the kernel can
call render(), so a slow poll is a frozen display. Return truthy when you did
work — you are called straight back, so a busy transport is never rate-limited
by us — and falsy when idle, which earns a `poll_interval` pause.

A long-poll with a SHORT server-side timeout is the right shape. An unbounded
wait is not.


RENDERING IS NOT A REQUEST
--------------------------
The kernel calls render(sdk, session_key, kind, payload) on you. One method,
and `kind` says what:

    messages      list[str] of markdown        attachments   list of paths
    form_field    dict (name/field/collected/display)
    approval      dict (id/title/body/type/enum/default)
    approval_settled  dict (request_id/reason)
    buttons       list[dict]                   error         dict
    typing        bool                         tool_status   dict
    stream_delta      dict       (only if you set supports_streaming)
    notification      dict       (only if you set supports_notifications)
    callable_output   list[str]  (only if you set supports_callable_output)

Handle what your transport can show and ignore the rest. A frontend that only
renders `messages` is a working frontend.

`messages` is the conversation and nothing else — the agent's replies and the
person's own words. A refusal is `error`, an announcement is `notification`,
and what a slash command answered with is `callable_output`. The last three
kinds above are opt-in, and declining any of them is free: the kernel sends
that content as `messages` instead, exactly as it did before the kind existed.
So a transport gains a surface by asking and loses nothing by not.

Output is **markdown on the wire** — that is the interchange format, because it
is also what the model emits. Render it however your transport prefers.

An `approval` carries an id; answer it with sdk.frontend.resolve(). Holding the
id is enough to answer and only enough to answer — the action being authorized
never crosses to you.


CARRYING INPUT BACK
-------------------
This half IS Requests:

    sdk.frontend.submit_text(key, text)
    sdk.frontend.submit_attachment(key, path, extension="")
    sdk.frontend.submit_action(key, action_type, payload=None)
    sdk.frontend.cancel(key)
    sdk.frontend.resolve(key, value, request_id="")
    sdk.frontend.attended(key, present=True)
    sdk.frontend.bind(key, external_id=None, user_type="user", config=None)

They work only inside a loaded frontend, and each reaches YOUR frontend's
adapter — you cannot submit on another's behalf, and a tool that imported the
same namespace reaches nothing at all.

Never build a session key by hand in two places. session_key(sdk, ctx) is the
one place that decides, and the kernel treats two contexts with the same key as
the same person in the same place.


ATTENDANCE — IS A HUMAN ACTUALLY THERE?
---------------------------------------
The kernel refuses interactive tools in unattended sessions, and it only
*reads* attendance — you own the policy. A single-operator transport says
nothing and inherits the default. A concurrent one should say so explicitly:

    sdk.frontend.attended(key, True)     # on connect
    sdk.frontend.attended(key, False)    # on disconnect

That is the hook for multi-user surfaces. Without it, a website's idle tab
looks exactly like a person sitting there waiting.


USER BINDING — WHICH USER OWNS A SESSION  (READ THIS, IT IS NOT INTUITIVE)
--------------------------------------------------------------------------
A session_key identifies a *conversation stream*. A user_id identifies *whose
data* it is — conversations, per-user settings, credits. They are NOT the same
thing: giving every visitor a distinct session_key does NOT give them distinct
accounts. If you never bind a user, EVERY session — every website visitor
included — acts as the SAME default user, and they share data.

Two declared attributes decide it:

    user_binding    = "single" | "per_user"     (default "single")
    default_user_id = <uid>                     (default = the base user, 1)

The kernel auto-binds each new session to default_user_id while it is still
unbound, so most frontends declare these and write no per-session code.

  1. ONE FIXED USER — REPL, a personal Telegram bot, a single-operator tool.
        user_binding = "single"
        default_user_id = 1
     Every session is the base user. Nothing else to do.

  1b. ONE FIXED *SHARED* USER — a kiosk or public demo where everyone is the
      same sandbox account. Same mechanism, just not the base user.

  2. A DIFFERENT USER PER PERSON — a real multi-user website.
        user_binding = "per_user"
        default_user_id = <a GUEST uid, NOT the base user>
     Anonymous sessions land on the guest user. On login, upgrade:
        sdk.frontend.bind(key, external_id=email_or_username)
     `external_id` is whatever is unique within THIS frontend. The user is
     created on first sight. Pass user_type= to label app-specific classes
     ("creator", "paid"); the kernel stores the label and does not interpret it.

WARNING for "per_user": if you leave default_user_id at the base user,
anonymous visitors act as the OPERATOR and see operator data. Point it at a
dedicated guest user.

Authenticating is YOUR job. The kernel ships no crypto and stores
password_hash opaquely — when you bind a session, you are asserting you did the
work, and the kernel takes your word for it.

Binding is the "whose data" axis ONLY. It does not decide permissions, which
commands run, or which agent is used — that is the frontend_profile. And a
user is only as isolated as the tools their profile exposes: the conversation
guard protects the built-in conversation surface, but a permissive tool like
raw SQL can still read across users.


A NOTE ON TERMINALS
-------------------
You cannot write a REPL on this contract. input() is refused (it would block
the box), and a subprocess box's stdin is the wire protocol, so sandboxed code
has no route to a terminal. The kernel's REPL is deliberately still native.
Network-driven frontends have no such problem, which is why the example below
is one.
"""

from guest.bases import BaseFrontend


class Chat(BaseFrontend):
    """A chat network reached over HTTP. The shape most frontends have."""

    name = "chat"
    description = "Relays a chat service into Second Brain."
    # Guidance added to the agent's system prompt for sessions running on
    # this frontend. A method (``def agent_prompt(self, sdk)``) works too.
    agent_prompt = "## Chat\nReplies render as markdown; keep them short."

    # Paid only when a poll finds nothing. Keep it small: it is also the
    # longest a render can be delayed by an idle loop.
    poll_interval = 0.05

    # A real multi-user surface: each account its own user, anonymous traffic
    # on a guest. Declared as a plain dict because a box cannot hold a
    # dataclass — the kernel rebuilds it into FrontendCapabilities.
    user_binding = "per_user"
    default_user_id = 2
    capabilities = {
        "supports_typing": True,
        "supports_buttons": True,
        "supports_attachments_in": True,
        "supports_rich_text": True,
        "max_message_chars": 4096,
    }

    config_settings = [
        ("Chat API token", "secret_chat_api_token",
         "Bot token for the chat service.", "", {"type": "text"}),
    ]

    def start(self, sdk):
        """Open the transport and return. Do NOT loop here."""
        self._cursor = 0
        # A <secret:...> handle, not the token. Safe to hold and log; the
        # kernel swaps in the real value inside sdk.net.http.
        self._token = sdk.config.read("secret_chat_api_token")
        sdk.log("chat frontend ready")
        return True

    def poll(self, sdk):
        """Take whatever is waiting and hand it over. Must return promptly."""
        updates = sdk.net.http_json(
            "https://chat.example.com/updates",
            params={"after": self._cursor, "timeout": 20},
            headers={"Authorization": f"Bearer {self._token}"},
        )
        items = (updates.get("body") or {}).get("items") or []
        for item in items:
            self._cursor = item["id"]
            key = self.session_key(sdk, item)
            # Each identity its own user. Binding an already-bound session is
            # harmless, so there is no need to track who has logged in.
            sdk.frontend.bind(key, external_id=item["from"])
            sdk.frontend.submit_text(key, item["text"])
        # Truthy: there was work, so come straight back rather than sleeping.
        return bool(items)

    def stop(self, sdk):
        """Close the transport. Must tolerate never having started."""
        self._cursor = 0

    def session_key(self, sdk, ctx):
        """One key per conversational surface. The single place that decides."""
        return f"chat:{(ctx or {}).get('room', 'unknown')}"

    def render(self, sdk, session_key, kind, payload):
        """Show one thing. Ignore the kinds this transport cannot show."""
        room = session_key.split(":", 1)[-1]

        if kind == "messages":
            for text in payload or []:
                self._send(sdk, room, text)

        elif kind == "approval":
            # payload is a projection: id, title, body, type, enum, default.
            self._send(sdk, room, f"**{payload['title']}**\n\n{payload['body']}")
            # A real transport shows buttons and answers when one is pressed;
            # answering immediately would defeat the point of asking.
            self._pending = payload["id"]

        elif kind == "form_field":
            display = payload.get("display") or {}
            field = payload.get("field") or {}
            self._send(sdk, room,
                       display.get("prompt") or field.get("prompt")
                       or field.get("name") or "Input required")

        elif kind == "error":
            self._send(sdk, room, f"⚠️ {(payload or {}).get('message') or payload}")

        elif kind == "typing":
            sdk.net.http("https://chat.example.com/typing", method="POST",
                         headers={"Authorization": f"Bearer {self._token}"},
                         json={"room": room, "on": bool(payload)})

    def _send(self, sdk, room, text):
        """One outbound message. Internal — the kernel never calls this."""
        sdk.net.http("https://chat.example.com/send", method="POST",
                     headers={"Authorization": f"Bearer {self._token}"},
                     json={"room": room, "text": text})
