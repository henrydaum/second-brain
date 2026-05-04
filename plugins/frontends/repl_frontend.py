from __future__ import annotations

import logging
import threading

from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST

logger = logging.getLogger("REPL")


class ReplFrontend(BaseFrontend):
    name = "repl"
    description = "Blocking terminal frontend backed by the conversation state machine."
    capabilities = FrontendCapabilities(
        supports_attachments_in=True,
        supports_proactive_push=True,
    )

    def __init__(self, shutdown_fn=None, shutdown_event: threading.Event | None = None):
        super().__init__()
        self.shutdown_fn = shutdown_fn
        self.shutdown_event = shutdown_event or threading.Event()

    def session_key(self, _ctx=None) -> str:
        return "default"

    def start(self) -> None:
        key = self.session_key(None)
        print("Second Brain REPL ready. Type /help for commands, /quit to exit.")
        while not self.shutdown_event.is_set():
            try:
                raw = input("\n").strip()
            except KeyboardInterrupt:
                if self.shutdown_fn:
                    self.shutdown_fn()
                return
            except EOFError:
                logger.info("REPL stdin closed; stopping REPL without shutting down the app.")
                return
            if not raw:
                continue
            if self.has_pending_approval(key):
                self._handle_raw(key, raw)
                continue
            threading.Thread(target=self._handle_raw, args=(key, raw), daemon=True, name="repl-submit").start()

    def _handle_raw(self, key: str, raw: str) -> None:
        try:
            if raw.startswith("/attach"):
                _, _, path = raw.partition(" ")
                self.submit_attachment(key, path.strip()) if path.strip() else self.render_error(key, {"message": "Usage: /attach <path>"})
            else:
                self.submit_text(key, raw)
        except Exception as e:
            logger.exception("REPL submit failed")
            self.render_error(key, {"message": str(e)})

    def submit_text(self, session_key: str, text: str):
        if self._current_phase(session_key) != PHASE_APPROVING_REQUEST and self.has_pending_approval(session_key):
            approved = self._parse_approval(text)
            if approved is None:
                self.render_error(session_key, {"message": "Approval needs yes or no."})
                return None
            ok = self.resolve_next_approval(session_key, approved, self.name)
            self.render_messages(session_key, ["Approval granted." if ok and approved else "Approval denied." if ok else "No pending approvals."])
            return None
        return super().submit_text(session_key, text)

    def stop(self) -> None:
        self.shutdown_event.set()
        self.unbind()

    def render_messages(self, _session_key: str, messages: list[str]) -> None:
        for msg in messages:
            if msg:
                print(f"\n{msg}")

    def render_attachments(self, _session_key: str, paths: list[str]) -> None:
        for path in paths:
            print(f"\n[attachment] {path}")

    def render_form_field(self, _session_key: str, form: dict) -> None:
        field = form.get("field") or {}
        hints = self._hints(field)
        print(f"\n{form.get('name')}: {field.get('prompt') or field.get('name')}{hints}")

    def render_approval_request(self, _session_key: str, req) -> None:
        hints = self._hints({"type": getattr(req, "type", "boolean"), "enum": getattr(req, "enum", None), "default": getattr(req, "default", None)})
        body = f"\n{req.body}" if getattr(req, "body", "") else ""
        print(f"\n{getattr(req, 'title', 'Approval requested')}{body}{hints}")

    def render_buttons(self, _session_key: str, buttons: list[dict]) -> None:
        for i, button in enumerate(buttons, 1):
            label = button.get("label") or button.get("text") or button.get("value") or "Option"
            print(f"{i}. {label}")

    def render_error(self, _session_key: str, error: dict) -> None:
        print(f"\n[error] {(error or {}).get('message') or error}")

    def render_tool_status(self, _session_key: str, payload: dict) -> None:
        name = payload.get("tool_name") or "tool"
        status = "..." if payload.get("status") == "started" else "ok" if payload.get("ok") else "failed"
        print(f"\n[tool] {name} {status}")

    def _live_session_keys(self) -> list[str]:
        return [self.session_key(None)]

    @staticmethod
    def _hints(field: dict) -> str:
        parts = []
        if field.get("enum"):
            parts.append("options: " + ", ".join(map(str, field["enum"])))
        if field.get("required") is False:
            parts.append("/skip to skip")
        if field.get("default") is not None:
            parts.append(f"default: {field['default']}")
        if field.get("type") and field.get("type") != "string":
            parts.append(f"type: {field['type']}")
        return f" ({'; '.join(parts)})" if parts else ""

    @staticmethod
    def _parse_approval(text: str) -> bool | None:
        value = (text or "").strip().lower()
        if value in {"/cancel", "n", "no", "deny", "denied", "false", "0"}:
            return False
        if value in {"y", "yes", "approve", "approved", "true", "1"}:
            return True
        return None
