"""
tool_email_send — Send a new email or reply to an existing thread via Gmail.

Identity is controlled by two parameters:
- as_ai=False: send from the authenticated Google account, with approval prompt.
- as_ai=True, from_address=<alias>: send from that alias (must be in
  ai_email_addresses), no approval — autonomous.
- as_ai=True, from_address blank: send from the authenticated Google account
  ("me") IF it is listed in ai_email_addresses. Otherwise fall back to the
  first entry in ai_email_addresses. This prevents a blank from_address
  from bypassing the configured access list.

Subagent guard: subagents are forced to as_ai=True, and only run when
ai_email_addresses is non-empty (configured access list).

Config (set in UI → Settings → Plugin Config):
    ai_email_addresses — list of Gmail send-as aliases (each must be added
                         under Gmail Settings → Accounts → Send mail as).
                         Empty list = no agent send access.
"""

import logging
import os
import re

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("tool_email_send")

_ADDR_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _extract_addrs(header: str) -> list[str]:
    return [m.group(0).lower() for m in _ADDR_RE.finditer(header or "")]


def _allowed_addresses(config) -> list[str]:
    raw = config.get("ai_email_addresses") or []
    if not isinstance(raw, list):
        return []
    return [str(a).strip() for a in raw if str(a).strip()]


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
            "AI Agent Email Addresses",
            "ai_email_addresses",
            "List of Gmail send-as aliases the agent may send from. Each must "
            "be added under Gmail Settings → Accounts → Send mail as. Empty "
            "list = no agent send access.",
            [],
            {"type": "json_list"},
        ),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "as_ai": {
                "type": "boolean",
                "description": (
                    "If true, send from one of the configured ai_email_addresses "
                    "entries with no approval prompt (autonomous mode). If false, "
                    "send from the user's main account and require approval."
                ),
                "default": False,
            },
            "from_address": {
                "type": "string",
                "description": (
                    "Pick which configured alias to send from (must be in "
                    "ai_email_addresses). Leave blank to send from the "
                    "authenticated Google account ('me')."
                ),
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
        allowed = _allowed_addresses(context.config)
        requested_from = (kwargs.get("from_address") or "").strip()

        # Subagent guard: force AI mode so a subagent can never send from the
        # user's main address. Empty list = no send access.
        if context.is_subagent:
            if not allowed:
                return ToolResult.failed(
                    "Subagent context but ai_email_addresses is empty — no "
                    "send access. Configure at least one alias under Settings "
                    "→ Plugin Config → AI Agent Email Addresses."
                )
            if not as_ai:
                logger.warning("[EmailSend] Subagent context — forcing as_ai=True.")
                as_ai = True

        if as_ai and not requested_from and kwargs.get("message_id"):
            original_for_id = gmail.get_message(kwargs["message_id"])
            if original_for_id:
                to_addrs = _extract_addrs(original_for_id.get("recipients", ""))
                cc_addrs = _extract_addrs(original_for_id.get("cc", ""))
                allowed_lower = [a.lower() for a in allowed]
                inferred = (
                    next((a for a in allowed_lower if a in to_addrs), None)
                    or next((a for a in allowed_lower if a in cc_addrs), None)
                )
                if inferred:
                    requested_from = inferred
                    logger.info(f"[EmailSend] Reply-identity inferred: {inferred}")

        if as_ai and requested_from:
            allowed_lower = {a.lower() for a in allowed}
            if requested_from.lower() not in allowed_lower:
                return ToolResult.failed(
                    f"from_address '{requested_from}' is not in "
                    f"ai_email_addresses ({', '.join(allowed) or 'empty'})."
                )
            ai_email = requested_from
            from_address = ai_email
        elif as_ai:
            # Blank from_address with as_ai=True: send from the authenticated
            # Google account ("me") only if it's listed in ai_email_addresses
            # — otherwise fall back to the first allowed alias to prevent a
            # configured-allowlist bypass.
            self_address = gmail.get_self_address()
            allowed_lower = {a.lower() for a in allowed}
            if self_address and self_address.lower() in allowed_lower:
                ai_email = self_address
                from_address = None
            elif allowed:
                ai_email = allowed[0]
                from_address = ai_email
            else:
                return ToolResult.failed(
                    "as_ai=True with blank from_address but ai_email_addresses "
                    "is empty and the authenticated account is unknown — "
                    "configure at least one alias under Settings → Plugin "
                    "Config → AI Agent Email Addresses."
                )
        else:
            ai_email = ""
            from_address = None

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
        missing = [p for p in attachments or [] if not os.path.isfile(p)]
        if missing:
            return ToolResult.failed(
                "Attachment file(s) not found; email was not sent: " + ", ".join(missing)
            )

        if as_ai and from_address is None:
            identity_label = f"your main account ({ai_email}, autonomous)"
        elif as_ai:
            identity_label = f"alias {ai_email}"
        else:
            identity_label = "your main account"

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
