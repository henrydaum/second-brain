"""Sandboxed REPL frontend backed by the conversation runtime."""

from guest.bases import BaseFrontend


class ReplFrontend(BaseFrontend):
    """Terminal frontend using the kernel-owned nonblocking console."""

    name = "repl"
    description = "Terminal frontend backed by the conversation state machine."
    isolation = "subprocess"
    uses_console = True
    background_submit = True
    restore_on_start = True
    capabilities = {
        "supports_attachments_in": True,
        "supports_proactive_push": True,
        "supports_streaming": True,
    }
    user_binding = "single"
    default_user_id = 1
    requests = [
        "console.read", "console.write", "frontend.submit",
        "frontend.pending", "frontend.resolve", "session.get",
    ]

    def start(self, sdk):
        """Initialize display state and announce readiness."""
        self._stream_wrote = False
        self._prompted = False
        sdk.console.write("Second Brain REPL ready. Type /quit to exit.")
        return True

    def poll(self, sdk):
        """Drain one console line without blocking."""
        raw = sdk.console.read_line()
        if raw is None:
            if not self._prompted:
                sdk.console.write("\n", end="")
                self._prompted = True
            return False
        self._prompted = False
        raw = raw.strip()
        if not raw:
            return True

        key = "default"
        session = sdk.session.get(key) or {}
        pending = sdk.frontend.pending_approval(key)
        if pending and session.get("phase") != "approving_request":
            approved = self._parse_approval(raw)
            if approved is None:
                self._error(sdk, "Approval needs yes or no.")
                return True
            request_id = pending if isinstance(pending, str) else ""
            ok = sdk.frontend.resolve(key, approved, request_id)
            message = (
                "Approval granted." if ok and approved
                else "Approval denied." if ok
                else "No pending approvals."
            )
            self._messages(sdk, [message])
            return True

        if raw.startswith("/attach"):
            _, _, path = raw.partition(" ")
            path = path.strip()
            if not path:
                self._error(sdk, "Usage: /attach <path>")
            else:
                sdk.frontend.submit_attachment(key, path)
            return True

        sdk.frontend.submit_text(key, raw)
        return True

    def render(self, sdk, session_key, kind, payload):
        """Render one projected frontend payload to the terminal."""
        if kind == "messages":
            self._messages(sdk, payload or [])
        elif kind == "attachments":
            for path in payload or []:
                sdk.console.write(f"\n[attachment] {path}")
        elif kind == "form_field":
            form = payload or {}
            field = form.get("field") or {}
            display = form.get("display") or {}
            prompt = (
                display.get("prompt") or field.get("prompt")
                or field.get("name") or "Input required"
            )
            sdk.console.write(
                f"\n{sdk.md.plain(str(prompt))}{self._hints(display or field)}"
            )
        elif kind == "approval":
            request = payload or {}
            hints = self._hints({
                "type": request.get("type", "boolean"),
                "enum": request.get("enum"),
                "default": request.get("default"),
            })
            body = (
                f"\n{sdk.md.plain(str(request.get('body')))}"
                if request.get("body") else ""
            )
            sdk.console.write(
                f"\n{request.get('title') or 'Approval requested'}"
                f"{body}{hints}"
            )
        elif kind == "buttons":
            for index, button in enumerate(payload or [], 1):
                label = (
                    button.get("label") or button.get("text")
                    or button.get("value") or "Option"
                )
                sdk.console.write(f"{index}. {label}")
        elif kind == "error":
            self._error(sdk, (payload or {}).get("message") or payload)
        elif kind == "stream_delta":
            self._stream(sdk, payload or {})
        elif kind == "tool_status":
            self._tool_status(sdk, payload or {})

    def session_key(self, sdk, ctx):
        """Return the singleton REPL session key."""
        return "default"

    @staticmethod
    def _messages(sdk, messages):
        for message in messages:
            if message:
                sdk.console.write(f"{sdk.md.plain(message)}\n")

    @staticmethod
    def _error(sdk, message):
        sdk.console.write(f"\n[error] {message}")

    def _stream(self, sdk, payload):
        if payload.get("done"):
            if self._stream_wrote:
                sdk.console.write(
                    "", end="\n\n" if not payload.get("aborted") else "\n"
                )
            self._stream_wrote = False
            return
        delta = payload.get("delta") or ""
        if delta:
            self._stream_wrote = True
            sdk.console.write(delta, end="")

    @staticmethod
    def _tool_status(sdk, payload):
        name = payload.get("tool_name") or payload.get("command_name") or "call"
        if payload.get("status") == "started":
            sdk.console.write(f"\n⋯ {name}...", end="")
        elif payload.get("status") == "finished":
            mark = "✓" if payload.get("ok") else "✕"
            sdk.console.write(f"\r{mark} {name}   ")

    @staticmethod
    def _hints(field):
        parts = []
        choices = field.get("choices") or []
        if choices:
            parts.append(
                "options: " + ", ".join(
                    str(choice.get("label") or choice.get("value"))
                    for choice in choices
                )
            )
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
    def _parse_approval(text):
        value = (text or "").strip().lower()
        if value in {"/cancel", "n", "no", "deny", "denied", "false", "0"}:
            return False
        if value in {"y", "yes", "approve", "approved", "true", "1"}:
            return True
        return None
