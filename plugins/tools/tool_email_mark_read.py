"""
tool_email_mark_read — Mark a Gmail message as read or unread.
"""

import logging

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("tool_email_mark_read")


class EmailMarkRead(BaseTool):
    name = "email_mark_read"
    description = (
        "Mark a Gmail message as read (default) or unread. "
        "Provide the message_id; set unread=true to add the UNREAD label instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": "The Gmail message ID to mark.",
            },
            "unread": {
                "type": "boolean",
                "description": "If true, mark as unread. Default false (mark as read).",
                "default": False,
            },
        },
        "required": ["message_id"],
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
                return ToolResult.failed("Gmail not connected.")

        message_id = (kwargs.get("message_id") or "").strip()
        if not message_id:
            return ToolResult.failed("message_id is required.")

        # Subagent guard: only allow marking messages that involve the AI alias.
        if context.is_subagent:
            ai_email = (context.config.get("ai_email_address") or "").strip().lower()
            if not ai_email:
                return ToolResult.failed(
                    "Subagent context but ai_email_address is not set — "
                    "cannot verify message ownership."
                )
            msg = gmail.get_message(message_id)
            if not msg:
                return ToolResult.failed(f"Message {message_id} not found.")
            haystack = " ".join([
                msg.get("sender", ""),
                msg.get("recipients", ""),
                msg.get("cc", ""),
            ]).lower()
            if ai_email not in haystack:
                logger.warning(
                    f"[EmailMarkRead] Subagent rejected: {message_id} does not "
                    f"involve {ai_email}."
                )
                return ToolResult.failed(
                    "Subagent context: this message does not involve the AI "
                    "alias and cannot be modified."
                )

        unread = bool(kwargs.get("unread", False))
        ok = gmail.mark_unread(message_id) if unread else gmail.mark_read(message_id)
        action = "unread" if unread else "read"
        if ok:
            logger.info(f"[EmailMarkRead] Marked {message_id} as {action}")
            return ToolResult(
                success=True,
                data={"message_id": message_id, "marked": action},
                llm_summary=f"Message {message_id} marked as {action}.",
            )
        return ToolResult.failed(f"Failed to mark message {message_id}.")
