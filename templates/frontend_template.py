"""
FRONTEND TEMPLATE
=================
This file is a self-contained reference for creating frontend plugins.
It is NOT imported by the running system — it exists for LLM consumption only.

Frontends are transports: REPL, Telegram, HTTP, desktop UI, etc. They turn
user input into state-machine actions and render RuntimeResult/events back to
the user. Prefer commands/tools/tasks for app behavior; create a frontend only
when adding a new transport or presentation layer.

Frontend authoring flow:
  1. Read this template, then read plugins/frontends/frontend_repl.py or
     frontend_telegram.py for the closest existing transport.
  2. Create sandbox_frontends/frontend_<your_name>.py with edit_file.
  3. The code MUST inherit from BaseFrontend and include:
       from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
  4. Fill in name, description, capabilities, lifecycle, session_key(), and render_* methods.
  5. Call test_plugin(plugin_path="sandbox_frontends/frontend_<your_name>.py").
  6. If testing fails, read the error, edit the same file, and retry.
  7. Valid plugins are discovered on startup; plugin_watcher live-loads adds/edits when enabled.
  8. To update: edit the file; plugin_watcher reloads it when enabled.
  9. To remove live and durably: delete the sandbox file; plugin_watcher unloads it when enabled.

AUTO-DISCOVERY RULES
--------------------
- File must be in plugins/frontends/ (baked-in) or the sandbox frontends dir
- File name must start with "frontend_"
- Class must inherit from BaseFrontend
- Class must have a non-empty `name`

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
"""

# =====================================================================
# BASE SHAPE (shortened from plugins/BaseFrontend.py)
# =====================================================================

from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities


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
