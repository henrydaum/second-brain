"""
tool_email_send — Send a new email or reply to an existing thread via Gmail.

Two identities are supported:
- User account (as_ai=False): requires user approval before sending. Use when
  the user explicitly wants to send mail from their personal address.
- AI alias (as_ai=True): sends from the configured ai_email_address. No
  approval prompt — autonomous. Use for agent-driven outreach.

Subagent guard: if no approval UI is subscribed (i.e. running inside a
subagent), as_ai is forced to True regardless of the caller's argument so a
subagent can never send from the user's main address. If as_ai is forced on
but ai_email_address is unset, the call fails loudly rather than silently
falling back.

Config (set in UI → Settings → Plugin Config):
    ai_email_address — Gmail send-as alias (must be added under Gmail
                       Settings → Accounts → Send mail as).
"""

import logging

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("tool_email_send")


class EmailSend(BaseTool):
    name = "email_send"
    description = (
        "Send a new email or reply to an existing message thread via Gmail. "
        "Set as_ai=true to send from the agent's dedicated alias autonomously "
        "(no approval prompt). Set as_ai=false to send from the user's main "
        "address (requires approval). For replies, pass message_id; for new "
        "messages, pass to and subject. Sending is irreversible."
    )
    config_settings = [
        (
            "AI Agent Email Address",
            "ai_email_address",
            "Gmail send-as alias (e.g. agent@yourdomain.com). "
            "Must be added under Gmail Settings → Accounts → Send mail as.",
            "",
            {"type": "text"},
        ),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "as_ai": {
                "type": "boolean",
                "description": (
                    "If true, send from the configured ai_email_address with "
                    "no approval prompt (autonomous mode). If false, send "
                    "from the user's main account and require approval."
                ),
                "default": False,
            },
            "message_id": {
                "type": "string",
                "description": (
                    "Gmail message ID to reply to. If omitted, a new message is sent. "
                    "When provided, the reply is placed in the same thread."
                ),
            },
            "to": {
                "type": "string",
                "description": "Recipient email address. Required for new messages.",
            },
            "cc": {
                "type": "string",
                "description": "Optional CC recipient(s), comma-separated if multiple.",
            },
            "subject": {
                "type": "string",
                "description": (
                    "Email subject line. Required for new messages. "
                    "For replies, auto-prefixed with 'Re: ' if not present."
                ),
            },
            "body": {
                "type": "string",
                "description": "Email body text (plain text).",
            },
            "attachments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of absolute file paths to attach.",
            },
        },
        "required": ["body"],
    }
    requires_services = ["gmail"]
    max_calls = 10
    background_safe = True

    def run(self, context, **kwargs) -> ToolResult:
        gmail = context.services.get("gmail")
        if not gmail:
            return ToolResult.failed("Gmail service not available.")
        if not gmail.loaded:
            if not gmail.load():
                return ToolResult.failed(
                    "Gmail not connected. Ensure credentials.json is present in the data directory."
                )

        as_ai = bool(kwargs.get("as_ai", False))

        # Subagent guard: force AI mode so a scheduled subagent can never
        # send from the user's main address.
        if context.is_subagent and not as_ai:
            logger.warning(
                "[EmailSend] Subagent context — forcing as_ai=True."
            )
            as_ai = True

        ai_email = (context.config.get("ai_email_address") or "").strip()
        if as_ai and not ai_email:
            return ToolResult.failed(
                "as_ai is required (subagent context or explicit) but "
                "ai_email_address is not set. Go to Settings → Plugin Config "
                "→ AI Agent Email Address and enter your Gmail send-as alias."
            )

        from_address = ai_email if as_ai else None

        body = kwargs.get("body", "").strip()
        if not body:
            return ToolResult.failed("Email body cannot be empty.")

        message_id = kwargs.get("message_id")
        to = kwargs.get("to", "").strip()
        cc = kwargs.get("cc", "").strip()
        subject = kwargs.get("subject", "").strip()
        raw_attachments = kwargs.get("attachments")
        attachments = (
            [p.strip() for p in raw_attachments if isinstance(p, str) and p.strip()]
            if raw_attachments else None
        )

        identity_label = f"alias {ai_email}" if as_ai else "your main account"

        # ── Reply mode ─────────────────────────────────────────────────────
        if message_id:
            original = gmail.get_message(message_id)
            reply_to_addr = ""
            original_subject = ""
            if original:
                from_email = original.get("sender", "")
                if "<" in from_email:
                    from_email = from_email.split("<")[1].rstrip(">")
                reply_to_addr = from_email
                original_subject = original.get("subject", "")

            if not as_ai:
                preview = "\n".join(line for line in [
                    f"Mode:        Reply (from {identity_label})",
                    f"To:          {reply_to_addr}",
                    f"Subject:     Re: {original_subject}" if original_subject else "",
                    f"Attachments: {', '.join(attachments) if attachments else 'none'}",
                    "",
                    "Body:",
                    body,
                ] if line)
                denied = _require_approval(context, f"Reply to {reply_to_addr}", preview)
                if denied:
                    return denied

            logger.info(f"[EmailSend] Replying to {message_id} as {identity_label}")
            result = gmail.reply_to(
                message_id=message_id, body=body,
                attachments=attachments, from_address=from_address,
            )
            if result:
                return ToolResult(
                    success=True,
                    data={"sent": True, "message_id": result, "mode": "reply", "as_ai": as_ai},
                    llm_summary=f"Reply sent from {identity_label}. Message ID: {result}",
                )
            return ToolResult.failed("Failed to send reply. Check the message_id and try again.")

        # ── New message ────────────────────────────────────────────────────
        if not to:
            return ToolResult.failed("Recipient ('to') is required for new messages.")
        if not subject:
            return ToolResult.failed("Subject is required for new messages.")

        if not as_ai:
            preview = "\n".join(line for line in [
                f"From:        {identity_label}",
                f"To:          {to}",
                f"CC:          {cc}" if cc else "",
                f"Subject:     {subject}",
                f"Attachments: {', '.join(attachments) if attachments else 'none'}",
                "",
                "Body:",
                body,
            ] if line)
            denied = _require_approval(context, f"Send email to {to}", preview)
            if denied:
                return denied

        logger.info(f"[EmailSend] Sending new message to {to} as {identity_label}")
        result = gmail.send_message(
            to=to, subject=subject, body=body, cc=cc,
            attachments=attachments, from_address=from_address,
        )
        if result:
            return ToolResult(
                success=True,
                data={"sent": True, "message_id": result, "mode": "new", "as_ai": as_ai},
                llm_summary=(
                    f"Email sent from {identity_label} to {to}. "
                    f"Subject: '{subject}'. Message ID: {result}"
                ),
            )
        return ToolResult.failed("Failed to send email. Check recipient address and try again.")


def _require_approval(context, action_summary: str, detail: str) -> ToolResult | None:
    approve_fn = context.approve_command
    if approve_fn is None:
        return ToolResult.failed(
            "Approval dialog is not available — cannot send email from user account."
        )
    try:
        approved = approve_fn(action_summary, detail)
    except Exception as e:
        logger.error(f"[EmailSend] Approval callback failed: {e}")
        return ToolResult.failed(f"Approval dialog error: {e}")
    if not approved:
        return ToolResult.failed(
            "Email send denied by user. STOP — do not retry. "
            "Ask the user what they would like to do instead."
        )
    return None
