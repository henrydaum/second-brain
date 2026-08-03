"""Mark a Gmail message read or unread."""

dependencies_files = ["tools/helpers/email_context.py"]
dependencies_pip = []
requests = ["service.call", "config.read", "session.get", "conv.read"]

from guest.bases import BaseTool
from .email_context import allowed_addresses, is_main_conversation, message_involves


class EmailMarkRead(BaseTool):
    name = "email_mark_read"
    description = "Mark a Gmail message as read or unread."
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string"},
            "unread": {"type": "boolean", "default": False},
            "narration": {"type": "string", "description": "A few words on which message and why, shown to the user beside the call. E.g. 'marking the newsletter read'."},
        },
        "required": ["message_id"],
    }
    requires_services = ["gmail"]

    def run(self, sdk, **kwargs):
        message_id = str(kwargs.get("message_id") or "").strip()
        if not message_id:
            return sdk.fail("message_id is required.")
        if not is_main_conversation(sdk):
            allowed = allowed_addresses(sdk)
            if not allowed:
                return sdk.fail("Non-main conversation has no configured AI email access.")
            message = sdk.services.call(
                "gmail", "get_message", message_id=message_id)
            if not message:
                return sdk.fail(f"Message {message_id} not found.")
            if not message_involves(message, allowed):
                return sdk.fail(
                    "This message does not involve a configured AI alias and cannot be modified.")
        unread = bool(kwargs.get("unread", False))
        method, action = ("mark_unread", "unread") if unread else ("mark_read", "read")
        ok = sdk.services.call("gmail", method, message_id=message_id)
        if not ok:
            return sdk.fail(f"Failed to mark message {message_id} as {action}.")
        sdk.log(f"marked Gmail message {message_id} as {action}")
        return sdk.ok(
            {"message_id": message_id, "marked": action},
            llm_summary=f"Message {message_id} marked as {action}.",
        )
