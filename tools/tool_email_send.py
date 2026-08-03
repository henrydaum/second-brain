"""Send a new Gmail message or reply to an existing thread."""

dependencies_files = ["tools/helpers/email_context.py"]
dependencies_pip = []
requests = [
    "service.call", "config.read", "session.get", "conv.read", "fs.list",
    "ui.approve",
]

import re

from guest.bases import BaseTool
from .email_context import allowed_addresses, is_main_conversation

ADDRESS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _addresses(header):
    return [match.group(0).lower() for match in ADDRESS.finditer(header or "")]


def _exists(sdk, path):
    try:
        return bool(sdk.fs.list(path))
    except sdk.Failed:
        return False


class EmailSend(BaseTool):
    name = "email_send"
    description = (
        "Send or reply through Gmail. AI aliases may send autonomously; sending "
        "from the user's main account requires approval and is irreversible."
    )
    config_settings = [
        ("AI Agent Email Addresses", "ai_email_addresses",
         "Gmail send-as aliases the agent may autonomously use.", [],
         {"type": "json_list"}),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "as_ai": {"type": "boolean", "default": False},
            "from_address": {"type": "string"},
            "message_id": {"type": "string", "description": "Message to reply to."},
            "to": {"type": "string"}, "cc": {"type": "string"},
            "subject": {"type": "string"}, "body": {"type": "string"},
            "attachments": {"type": "array", "items": {"type": "string"}},
            "narration": {"type": "string", "description": "A few words on what you are sending and to whom, shown to the user beside the call. E.g. 'replying to Sarah about Thursday'."},
        },
        "required": ["body"],
    }
    requires_services = ["gmail"]

    def run(self, sdk, **kwargs):
        as_ai = bool(kwargs.get("as_ai", False))
        allowed = allowed_addresses(sdk)
        requested_from = str(kwargs.get("from_address") or "").strip()
        message_id = str(kwargs.get("message_id") or "").strip()

        if not is_main_conversation(sdk):
            if not allowed:
                return sdk.fail("Non-main conversation has no configured AI send access.")
            as_ai = True

        original = None
        if message_id:
            original = sdk.services.call(
                "gmail", "get_message", message_id=message_id)
            if not original:
                return sdk.fail(f"Message {message_id} not found.")
        if as_ai and not requested_from and original:
            recipients = _addresses(original.get("recipients", ""))
            copies = _addresses(original.get("cc", ""))
            requested_from = next(
                (address for address in allowed
                 if address.lower() in recipients or address.lower() in copies), "")

        from_address, identity = self._identity(
            sdk, as_ai, requested_from, allowed)
        if identity.startswith("ERROR:"):
            return sdk.fail(identity[6:])

        body = str(kwargs.get("body") or "").strip()
        if not body:
            return sdk.fail("Email body cannot be empty.")
        attachments = [
            str(path).strip() for path in (kwargs.get("attachments") or [])
            if isinstance(path, str) and path.strip()
        ]
        missing = [path for path in attachments if not _exists(sdk, path)]
        if missing:
            return sdk.fail("Attachment file(s) not found: " + ", ".join(missing))

        if message_id:
            recipient = _addresses(original.get("sender", ""))
            recipient = recipient[0] if recipient else original.get("sender", "")
            preview = _preview(
                "Reply", identity, recipient,
                original.get("subject", ""), "", body, attachments)
            if not as_ai and not _approve(sdk, f"Reply to {recipient}", preview):
                return sdk.fail("Email send denied. STOP and do not retry.")
            sent = sdk.services.call(
                "gmail", "reply_to", message_id=message_id, body=body,
                attachments=attachments, from_address=from_address)
            mode = "reply"
        else:
            recipient = str(kwargs.get("to") or "").strip()
            subject = str(kwargs.get("subject") or "").strip()
            cc = str(kwargs.get("cc") or "").strip()
            if not recipient:
                return sdk.fail("Recipient ('to') is required.")
            if not subject:
                return sdk.fail("Subject is required.")
            preview = _preview(
                "New message", identity, recipient, subject, cc, body, attachments)
            if not as_ai and not _approve(
                    sdk, f"Send email to {recipient}", preview):
                return sdk.fail("Email send denied. STOP and do not retry.")
            sent = sdk.services.call(
                "gmail", "send_message", to=recipient, subject=subject,
                body=body, cc=cc, attachments=attachments,
                from_address=from_address)
            mode = "new"
        if not sent:
            return sdk.fail("Failed to send email.")
        return sdk.ok(
            {"sent": True, "message_id": sent, "mode": mode, "as_ai": as_ai},
            llm_summary=f"Email {mode} sent from {identity}. Message ID: {sent}",
        )

    @staticmethod
    def _identity(sdk, as_ai, requested, allowed):
        if not as_ai:
            return None, "your main account"
        allowed_by_lower = {address.lower(): address for address in allowed}
        if requested:
            chosen = allowed_by_lower.get(requested.lower())
            if not chosen:
                return None, f"ERROR:from_address '{requested}' is not allowed."
            return chosen, f"alias {chosen}"
        self_address = sdk.services.call("gmail", "get_self_address") or ""
        if self_address.lower() in allowed_by_lower:
            return None, f"main account {self_address} (autonomous)"
        if allowed:
            return allowed[0], f"alias {allowed[0]}"
        return None, "ERROR:ai_email_addresses is empty."


def _approve(sdk, action, preview):
    try:
        return bool(sdk.ui.approve(action, preview))
    except sdk.Denied:
        return False


def _preview(mode, identity, recipient, subject, cc, body, attachments):
    lines = [f"Mode: {mode}", f"From: {identity}", f"To: {recipient}"]
    if cc:
        lines.append(f"CC: {cc}")
    if subject:
        lines.append(f"Subject: {subject}")
    lines.extend([
        f"Attachments: {', '.join(attachments) if attachments else 'none'}",
        "", "Body:", body,
    ])
    return "\n".join(lines)
