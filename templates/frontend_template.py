"""
FRONTEND TEMPLATE
=================
This file is a self-contained reference for creating frontend plugins.
It is NOT imported by the running system — it exists for LLM consumption only.

Frontends are transports: REPL, Telegram, HTTP, desktop UI, etc. They turn
user input into state-machine actions and render RuntimeResult/events back to
the user. Prefer commands/tools/tasks for app behavior; create a frontend only
when adding a new transport or presentation layer.
The Lite kernel ships the REPL. Other transports are normally installed
packages or sandbox drafts; add one to plugins/frontends/ only when it is true
kernel infrastructure.

Frontend authoring flow:
  1. Read this template, then read plugins/frontends/frontend_repl.py or the
     closest installed frontend for style.
  2. Create sandbox_plugins/frontends/frontend_<your_name>.py using whatever
     file-editing capability is installed and in scope.
  3. The code MUST inherit from BaseFrontend and include:
       from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
  4. Fill in name, description, capabilities, lifecycle, session_key(), and render_* methods.
  5. If a test_plugin tool is installed, call
     test_plugin(plugin_path="sandbox_plugins/frontends/frontend_<your_name>.py").
     Otherwise run focused pytest/compile checks from outside the runtime.
  6. If testing fails, read the error, edit the same file, and retry.
  7. Valid plugins are discovered on startup; plugin_watcher live-loads adds/edits when enabled.
  8. To update: edit the file; plugin_watcher reloads it when enabled.
  9. To remove live and durably: delete the sandbox file; plugin_watcher unloads it when enabled.

AUTO-DISCOVERY RULES
--------------------
- File must be in plugins/frontends/, sandbox_plugins/frontends/, or installed_plugins/frontends/
- File name must start with "frontend_"
- Class must inherit from BaseFrontend
- Class must have a non-empty `name`
- Import host APIs from plugins.* and helper code with relative imports.

FRONTEND CONTRACT
-----------------
The base class owns command parsing, form/approval state, runtime submission,
and bus subscriptions. Subclasses own transport-specific IO and rendering:

- start()/stop(): start or stop the transport
- session_key(ctx): derive a stable conversation/session key
- render_messages, render_attachments, render_form_field, render_approval_request,
  render_buttons, render_error: display runtime output
- Optional: render_typing and render_tool_status

Use submit_text(), submit_attachment(), submit(), and cancel() instead of
calling runtime.handle_action directly.

USER BINDING — WHICH USER OWNS A SESSION  (READ THIS, IT IS NOT INTUITIVE)
-------------------------------------------------------------------------
A session_key identifies a *conversation stream*. A user_id identifies *whose
data* it is (conversations, per-user /config settings, credits, etc.). They are
NOT the same thing: handing every visitor a distinct session_key does NOT give
them distinct accounts. If you never bind a user, EVERY session — every website
visitor included — acts as the SAME default user, and they share data.

Binding is a declared part of the contract via two attributes:

    user_binding    = "single" | "per_user"     (default "single")
    default_user_id = <uid>                       (default = the base user, 1)

The base auto-binds each new session to ``default_user_id`` (only while it is
still unbound), so you usually declare attributes and nothing else. There are
exactly THREE pathways, built from those two attributes:

  1. ONE FIXED USER  (REPL, installed Telegram, single-operator tools)
        user_binding = "single"            # default
        default_user_id = DEFAULT_USER_ID  # the base user (1)
     Every session is the base user. No per-session code. This is pathway (1).

  1b. ONE FIXED *SHARED* USER  (kiosk, public demo where everyone is the same
      sandbox account)
        user_binding = "single"
        default_user_id = <some shared uid>
     Same mechanism as (1), just not the base user.

  2. A DIFFERENT USER PER PERSON  (a real multi-user website)
        user_binding = "per_user"
        default_user_id = <a GUEST uid you create on start>   # NOT the base user!
     Anonymous sessions land on the guest user (because default_user_id points
     there). When a visitor logs in, UPGRADE that session to their real account:
        uid = self.bind_session(key, external_id=their_username_or_email)
     `external_id` is whatever is unique within THIS frontend (a username, email,
     or cookie). bind_session() creates the user on first sight and rebinds the
     session. This is pathways (2) and (3) — "each user is somebody else".
     Frontends that need app-specific user classes can pass a label when minting:
        uid = self.identify(key, external_id=email, user_type="creator")
     The kernel stores that label on users.user_type but does not treat it as an
     admin bypass; frontend/policy plugins decide what labels mean.

WARNING for "per_user": if you leave default_user_id at the base user, anonymous
visitors would act as the OPERATOR and see operator data. Always point a per_user
frontend's default_user_id at a dedicated guest user you upsert on start():
        self.default_user_id = self.runtime.db.upsert_user(self.name, "guest")

Binding is the "whose data" axis ONLY. It does NOT decide permissions, which
commands run, or which agent is used — that is the frontend_profile. A user is
only as isolated as the tools/commands their frontend_profile exposes (the
conversation guard protects the built-in conversation surface; a permissive tool
like raw SQL can still read across users).
"""

# =====================================================================
# BASE SHAPE (shortened from plugins/BaseFrontend.py)
# =====================================================================

from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from pipeline.database import DEFAULT_USER_ID


# =====================================================================
# EXAMPLE: A minimal print-style frontend skeleton
# =====================================================================

# from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
#
#
# class MinimalFrontend(BaseFrontend):
#     name = "minimal"
#     description = "Tiny example frontend backed by the conversation state machine."
#     capabilities = FrontendCapabilities(
#         supports_typing=True,
#         supports_proactive_push=True,
#     )
#     # Single local operator — every session is the base user (pathway 1).
#     user_binding = "single"
#     default_user_id = DEFAULT_USER_ID
#
#     def session_key(self, ctx=None) -> str:
#         return "minimal"
#
#     def start(self) -> None:
#         key = self.session_key(None)
#         notice = self.runtime.restore_last_active(key)
#         if notice:
#             self.render_messages(key, [notice])
#         # Real transports start their event loop here and call submit_text()
#         # or submit_attachment() when user input arrives.
#
#     def stop(self) -> None:
#         self.unbind()
#
#     def render_messages(self, session_key: str, messages: list[str]) -> None:
#         for message in messages:
#             print(message)
#
#     def render_attachments(self, session_key: str, paths: list[str]) -> None:
#         for path in paths:
#             print(f"[attachment] {path}")
#
#     def render_form_field(self, session_key: str, form: dict) -> None:
#         display = form.get("display") or {}
#         field = form.get("field") or {}
#         print(display.get("prompt") or field.get("prompt") or field.get("name") or "Input required")
#
#     def render_approval_request(self, session_key: str, req) -> None:
#         print(f"{req.title}\n{req.body}")
#
#     def render_buttons(self, session_key: str, buttons: list[dict]) -> None:
#         for button in buttons:
#             print(button.get("label") or button.get("text") or button.get("value"))
#
#     def render_error(self, session_key: str, error: dict) -> None:
#         print(f"[error] {(error or {}).get('message') or error}")


# =====================================================================
# EXAMPLE: Handling inbound transport events
# =====================================================================

# def on_user_text(frontend: BaseFrontend, transport_ctx, text: str):
#     key = frontend.session_key(transport_ctx)
#     frontend.submit_text(key, text)
#
#
# def on_user_file(frontend: BaseFrontend, transport_ctx, path: str):
#     key = frontend.session_key(transport_ctx)
#     frontend.submit_attachment(key, path)


# =====================================================================
# EXAMPLE: A multi-user ("per_user") web frontend — pathways 2 & 3
# =====================================================================
#
# class WebFrontend(BaseFrontend):
#     name = "web"
#     description = "Public website; anonymous chat, accounts for saved data."
#     user_binding = "per_user"          # each identity gets its own user
#
#     def start(self) -> None:
#         # Point the anonymous default at a GUEST user, never the base user, so
#         # logged-out visitors never act as the operator. Sessions auto-bind here
#         # until a visitor logs in.
#         self.default_user_id = self.runtime.db.upsert_user(self.name, "guest")
#         # ... start the HTTP/WebSocket server ...
#
#     def session_key(self, ctx) -> str:
#         return f"web:{ctx.connection_id}"     # one stream per connection
#
#     # On login, UPGRADE the session to the visitor's real account. external_id
#     # is unique within this frontend (username/email/cookie); bind_session
#     # creates the user on first sight and rebinds the session.
#     def on_login(self, ctx, username: str) -> None:
#         self.bind_session(self.session_key(ctx), external_id=username)
#
#     # On logout, drop back to the anonymous guest user.
#     def on_logout(self, ctx) -> None:
#         self.bind_session(self.session_key(ctx))   # no external_id -> default_user_id
