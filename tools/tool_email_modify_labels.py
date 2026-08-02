"""Add or remove Gmail labels on a message."""

dependencies_files = ["tools/helpers/email_context.py"]
dependencies_pip = []
requests = ["service.call", "config.read", "session.get", "conv.read"]

from guest.bases import BaseTool
from .email_context import allowed_addresses, is_main_conversation, message_involves

SYSTEM_LABELS = {
    "INBOX", "UNREAD", "SPAM", "TRASH", "IMPORTANT", "STARRED", "SENT",
    "DRAFT", "CHAT",
}


def _resolve(sdk, names):
    labels = sdk.services.call("gmail", "list_labels") or []
    mapping = {str(label.get("name") or "").lower(): label.get("id")
               for label in labels if label.get("id")}
    resolved, unknown = [], []
    for raw in names:
        name = str(raw).strip()
        if not name:
            continue
        if name.upper() in SYSTEM_LABELS:
            resolved.append(name.upper())
        elif mapping.get(name.lower()):
            resolved.append(mapping[name.lower()])
        else:
            unknown.append(name)
    if unknown:
        labels = sdk.services.call("gmail", "list_labels", force_refresh=True) or []
        mapping = {str(label.get("name") or "").lower(): label.get("id")
                   for label in labels if label.get("id")}
        still_unknown = []
        for name in unknown:
            if mapping.get(name.lower()):
                resolved.append(mapping[name.lower()])
            else:
                still_unknown.append(name)
        unknown = still_unknown
    return resolved, unknown


class EmailModifyLabels(BaseTool):
    name = "email_modify_labels"
    description = (
        "Add or remove Gmail labels. Remove INBOX to archive; add STARRED to star."
    )
    config_settings = [
        ("AI Agent Email Addresses", "ai_email_addresses",
         "Gmail send-as aliases the agent may autonomously access.", [],
         {"type": "json_list"}),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string"},
            "add": {"type": "array", "items": {"type": "string"}, "default": []},
            "remove": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "required": ["message_id"],
    }
    requires_services = ["gmail"]

    def run(self, sdk, **kwargs):
        message_id = str(kwargs.get("message_id") or "").strip()
        add = kwargs.get("add") or []
        remove = kwargs.get("remove") or []
        if not message_id:
            return sdk.fail("message_id is required.")
        if not isinstance(add, list) or not isinstance(remove, list):
            return sdk.fail("'add' and 'remove' must be lists.")
        add = [str(value) for value in add if str(value).strip()]
        remove = [str(value) for value in remove if str(value).strip()]
        if not add and not remove:
            return sdk.fail("At least one of 'add' or 'remove' must be non-empty.")

        if not is_main_conversation(sdk):
            allowed = allowed_addresses(sdk)
            if not allowed:
                return sdk.fail("Non-main conversation has no configured AI email access.")
            message = sdk.services.call(
                "gmail", "get_message", message_id=message_id)
            if not message or not message_involves(message, allowed):
                return sdk.fail(
                    "This message does not involve a configured AI alias and cannot be modified.")

        add_ids, unknown_add = _resolve(sdk, add)
        remove_ids, unknown_remove = _resolve(sdk, remove)
        unknown = unknown_add + unknown_remove
        if unknown:
            return sdk.fail(
                f"Unknown labels: {', '.join(unknown)}. Create them in Gmail first.")
        ok = sdk.services.call(
            "gmail", "modify_labels", message_id=message_id,
            add_ids=add_ids, remove_ids=remove_ids)
        if not ok:
            return sdk.fail(f"Failed to modify labels on {message_id}.")
        parts = []
        if add:
            parts.append(f"added {add}")
        if remove:
            parts.append(f"removed {remove}")
        summary = f"Message {message_id}: {', '.join(parts)}."
        return sdk.ok(
            {"message_id": message_id, "added": add, "removed": remove},
            llm_summary=summary,
        )
