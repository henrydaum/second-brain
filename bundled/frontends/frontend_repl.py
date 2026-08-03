"""Sandboxed REPL frontend backed by the conversation runtime."""

from guest.bases import BaseFrontend


class ReplFrontend(BaseFrontend):
    """Terminal frontend using the kernel-owned nonblocking console."""

    name = "repl"
    description = "Terminal frontend backed by the conversation state machine."
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
        self._approval = {}
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
            request_id = pending if isinstance(pending, str) else ""
            # A multi-choice dialog cannot be answered yes/no. Without this the
            # y/n parser rejects every option name outright, and resolving with
            # a bare True fails the enum on the far side — so the person is
            # locked out of a question they can see.
            shown = self._approval if self._approval.get("id") == request_id \
                else {}
            if shown.get("enum"):
                answer = self._match_option(shown, raw)
                if answer is None:
                    self._error(sdk, "Answer with one of: "
                                + ", ".join(self._option_labels(shown)))
                    return True
            else:
                answer = self._parse_approval(raw)
                if answer is None:
                    self._error(sdk, "Approval needs yes or no.")
                    return True
            ok = sdk.frontend.resolve(key, answer, request_id)
            self._messages(sdk, [
                f"Answered: {answer}." if ok and shown.get("enum")
                else "Approval granted." if ok and answer
                else "Approval denied." if ok
                else "No pending approvals."])
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
            # Kept so ``poll``'s out-of-phase branch knows whether the pending
            # question has options; the id is what proves it is the same one.
            self._approval = dict(request)
            hints = self._hints({
                "type": request.get("type", "boolean"),
                "enum": request.get("enum"),
                "enum_labels": request.get("enum_labels"),
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
    def _narration(payload):
        """The tool's own words about why it was called, if it declared them.

        Reserved parameter name: the model fills it, the kernel strips it before
        the tool runs, and it arrives inside ``args`` on the started event and at
        the top level on the finished one (which has no ``args``).
        """
        text = payload.get("narration") or (payload.get("args") or {}).get("narration")
        text = " ".join(str(text).split()) if text else ""
        if not text:
            return ""
        return f" *{text[:77]}...*" if len(text) > 80 else f" *{text}*"

    @staticmethod
    def _tool_status(sdk, payload):
        name = payload.get("tool_name") or payload.get("command_name") or "call"
        blurb = ReplFrontend._narration(payload)
        if payload.get("status") == "started":
            sdk.console.write(f"\n⋯ {name}{blurb}...", end="")
        elif payload.get("status") == "finished":
            mark = "✓" if payload.get("ok") else "✕"
            sdk.console.write(f"\r{mark} {name}{blurb}   ")

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
        elif field.get("type") == "boolean":
            # A yes/no question has no enum to list, so without this the
            # accepted vocabulary appeared only in the retry message after a
            # wrong guess. Kept next to ``_parse_approval``, which is what
            # actually defines the words.
            parts.append("yes/no")
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
    def _option_labels(payload):
        """The choices as a person was shown them."""
        values = payload.get("enum") or []
        labels = payload.get("enum_labels") or []
        return [str(labels[i]) if i < len(labels) else str(value)
                for i, value in enumerate(values)]

    @staticmethod
    def _match_option(payload, text):
        """Resolve typed text to an option *value*, or None.

        A superset of ``FormStep.match_enum``: this also accepts a 1-based
        index, because the REPL is the one surface where a person types instead
        of clicking, and "Always allow C:\\some\\long\\path" is not something
        anyone wants to retype.
        """
        values = payload.get("enum") or []
        labels = ReplFrontend._option_labels(payload)
        text = (text or "").strip()
        if not text:
            return None
        if text.isdigit() and 1 <= int(text) <= len(values):
            return values[int(text) - 1]
        folded = text.casefold()
        for index, value in enumerate(values):
            if folded in (str(value).casefold(), labels[index].casefold()):
                return value
        return None

    @staticmethod
    def _parse_approval(text):
        value = (text or "").strip().lower()
        if value in {"/cancel", "n", "no", "deny", "denied", "false", "0"}:
            return False
        if value in {"y", "yes", "approve", "approved", "true", "1"}:
            return True
        return None
