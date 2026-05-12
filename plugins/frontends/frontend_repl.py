"""REPL frontend plugin backed by the conversation runtime."""

from __future__ import annotations

import logging
import sys
import threading

from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from state_machine.conversation_phases import PHASE_APPROVING_REQUEST

logger = logging.getLogger("REPL")


class ReplFrontend(BaseFrontend):
    """Blocking terminal frontend backed by the conversation runtime."""
    name = "repl"
    description = "Blocking terminal frontend backed by the conversation state machine."
    capabilities = FrontendCapabilities(
        supports_attachments_in=True,
        supports_proactive_push=True,
    )

    def __init__(self, shutdown_fn=None, shutdown_event: threading.Event | None = None):
        """Initialize the REPL frontend."""
        super().__init__()
        self.shutdown_fn = shutdown_fn
        self.shutdown_event = shutdown_event or threading.Event()

    def session_key(self, _ctx=None) -> str:
        """Return the singleton REPL session key."""
        return "default"

    def start(self) -> None:
        """Start REPL frontend."""
        key = self.session_key(None)
        print("Second Brain REPL ready. Type /quit to exit.")
        try:
            notice = self.runtime.restore_last_active(key)
            if notice:
                self.render_messages(key, [notice])
        except Exception:
            logger.exception("REPL restore_last_active failed")
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
        """Route one raw REPL line into text or attachment submission."""
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
        """Submit text."""
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
        """Stop REPL frontend."""
        self.shutdown_event.set()
        self.unbind()

    def render_messages(self, _session_key: str, messages: list[str]) -> None:
        """Render messages."""
        for msg in messages:
            if msg:
                print(f"\n{msg}")

    def render_attachments(self, _session_key: str, paths: list[str]) -> None:
        """Render attachments."""
        for path in paths:
            print(f"\n[attachment] {path}")

    def render_form_field(self, _session_key: str, form: dict) -> None:
        """Render form field."""
        field = form.get("field") or {}
        display = form.get("display") or {}
        prompt = display.get("prompt") or field.get("prompt") or field.get("name") or "Input required"
        hints = self._hints(display or field)
        print(f"\n{prompt}{hints}")

    def render_approval_request(self, _session_key: str, req) -> None:
        """Render approval request."""
        hints = self._hints({"type": getattr(req, "type", "boolean"), "enum": getattr(req, "enum", None), "default": getattr(req, "default", None)})
        body = f"\n{req.body}" if getattr(req, "body", "") else ""
        print(f"\n{getattr(req, 'title', 'Approval requested')}{body}{hints}")

    def render_buttons(self, _session_key: str, buttons: list[dict]) -> None:
        """Render buttons."""
        for i, button in enumerate(buttons, 1):
            label = button.get("label") or button.get("text") or button.get("value") or "Option"
            print(f"{i}. {label}")

    def render_error(self, _session_key: str, error: dict) -> None:
        """Render error."""
        print(f"\n[error] {(error or {}).get('message') or error}")

    def render_tool_status(self, _session_key: str, payload: dict) -> None:
        """Render tool status."""
        name = payload.get("tool_name") or payload.get("command_name") or "call"
        if payload.get("status") == "started":
            sys.stdout.write(f"\n⋯ {name}...")
            sys.stdout.flush()
            return
        if payload.get("status") != "finished":
            return
        sys.stdout.write(f"\r{'✓' if payload.get('ok') else '✕'} {name}   \n")
        sys.stdout.flush()

    def _live_session_keys(self) -> list[str]:
        """Return the REPL sessions that can receive proactive output."""
        return [self.session_key(None)]

    @staticmethod
    def _hints(field: dict) -> str:
        """Render short REPL hints for a form field or approval."""
        parts = []
        choices = field.get("choices") or []
        if choices:
            parts.append("options: " + ", ".join(str(c.get("label") or c.get("value")) for c in choices))
        elif field.get("enum"):
            display = field.get("enum_labels") or field["enum"]
            parts.append("options: " + ", ".join(map(str, display)))
        if field.get("assist"):
            parts.append(str(field["assist"]))
        if field.get("allow_back"):
            parts.append("/back to go back")
        elif field.get("required") is False:
            parts.append("/skip to skip")
        if field.get("default") is not None and not field.get("assist"):
            parts.append(f"default: {field['default']}")
        return f" ({'; '.join(parts)})" if parts else ""

    @staticmethod
    def _parse_approval(text: str) -> bool | None:
        """Parse a yes-or-no REPL reply into an approval decision."""
        value = (text or "").strip().lower()
        if value in {"/cancel", "n", "no", "deny", "denied", "false", "0"}:
            return False
        if value in {"y", "yes", "approve", "approved", "true", "1"}:
            return True
        return None
