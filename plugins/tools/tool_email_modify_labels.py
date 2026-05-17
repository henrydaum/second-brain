"""
tool_email_modify_labels — Add or remove Gmail labels on a message.

System labels (INBOX, UNREAD, SPAM, TRASH, IMPORTANT, STARRED, SENT, DRAFT, CHAT)
pass through by name. Custom labels are resolved by name to their Gmail label ID
via the labels list; unknown custom labels fail (labels are not auto-created).

Common idioms:
    archive a message → remove=["INBOX"]
    star a message    → add=["STARRED"]
    mark as spam      → add=["SPAM"], remove=["INBOX"]
"""

import logging

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.email_context import is_main_conversation

logger = logging.getLogger("tool_email_modify_labels")

SYSTEM_LABELS = {
    "INBOX", "UNREAD", "SPAM", "TRASH", "IMPORTANT",
    "STARRED", "SENT", "DRAFT", "CHAT",
}


def _allowed_addresses(config) -> list[str]:
    """Internal helper to handle allowed addresses."""
    raw = config.get("ai_email_addresses") or []
    if not isinstance(raw, list):
        return []
    return [str(a).strip().lower() for a in raw if str(a).strip()]


def _resolve_labels(gmail, names: list[str]) -> tuple[list[str], list[str]]:
    """Resolve a list of label names to Gmail label IDs.

    Returns (resolved_ids, unknown_names). System labels pass through as their
    uppercase canonical name. Custom labels are looked up by case-insensitive
    name; on first miss, the label cache is force-refreshed once before
    declaring the name unknown.
    """
    resolved: list[str] = []
    unknown: list[str] = []
    cache: list[dict] | None = None
    refreshed = False

    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name.upper() in SYSTEM_LABELS:
            resolved.append(name.upper())
            continue
        if cache is None:
            cache = gmail.list_labels() or []
        match = next(
            (l["id"] for l in cache if l["name"].lower() == name.lower() and l.get("id")),
            None,
        )
        if match is None and not refreshed:
            cache = gmail.list_labels(force_refresh=True) or []
            refreshed = True
            match = next(
                (l["id"] for l in cache if l["name"].lower() == name.lower() and l.get("id")),
                None,
            )
        if match is None:
            unknown.append(name)
        else:
            resolved.append(match)
    return resolved, unknown


class EmailModifyLabels(BaseTool):
    """Email modify labels."""
    name = "email_modify_labels"
    description = (
        """Add or remove Gmail labels on a message. Pass add and/or remove as lists of label names. System labels (INBOX, UNREAD, SPAM, TRASH, IMPORTANT, STARRED, SENT, DRAFT) pass through by name. Custom labels are resolved by name — unknown custom labels fail and are not auto-created. To archive: remove=["INBOX"]. To star: add=["STARRED"]."""
    )
    config_settings = [
        (
            "AI Agent Email Addresses",
            "ai_email_addresses",
            "List of Gmail send-as aliases the agent may read, mark, and send from. "
            "Empty list = no agent access (subagents will fail).",
            [],
            {"type": "json_list"},
        ),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": "The Gmail message ID to modify.",
            },
            "add": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Label names to add to the message.",
                "default": [],
            },
            "remove": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Label names to remove from the message. "
                               "Use ['INBOX'] to archive.",
                "default": [],
            },
        },
        "required": ["message_id"],
    }
    requires_services = ["gmail"]
    max_calls = 10
    background_safe = True
    plan_mode_safe = False

    def run(self, context, **kwargs) -> ToolResult:
        """Run email modify labels."""
        gmail = context.services.get("gmail")
        if not gmail:
            return ToolResult.failed("Gmail service not available.")
        if not gmail.loaded:
            if not gmail.load():
                return ToolResult.failed("Gmail not connected.")

        message_id = (kwargs.get("message_id") or "").strip()
        if not message_id:
            return ToolResult.failed("message_id is required.")

        add_names = kwargs.get("add") or []
        remove_names = kwargs.get("remove") or []
        if not isinstance(add_names, list) or not isinstance(remove_names, list):
            return ToolResult.failed("'add' and 'remove' must be lists of label names.")
        add_names = [str(n) for n in add_names if str(n).strip()]
        remove_names = [str(n) for n in remove_names if str(n).strip()]
        if not add_names and not remove_names:
            return ToolResult.failed("At least one of 'add' or 'remove' must be non-empty.")

        # Non-main conversations may only modify messages involving an allowed alias.
        if not is_main_conversation(context):
            allowed = _allowed_addresses(context.config)
            if not allowed:
                return ToolResult.failed(
                    "Non-main conversation but ai_email_addresses is empty — no "
                    "mail access. Configure it under Settings → Plugin Config."
                )
            msg = gmail.get_message(message_id)
            if not msg:
                return ToolResult.failed(f"Message {message_id} not found.")
            haystack = " ".join([
                msg.get("sender", ""),
                msg.get("recipients", ""),
                msg.get("cc", ""),
            ]).lower()
            if not any(addr in haystack for addr in allowed):
                logger.warning(
                    f"[EmailModifyLabels] Subagent rejected: {message_id} does not "
                    f"involve any of {allowed}."
                )
                return ToolResult.failed(
                    "Non-main conversation: this message does not involve any "
                    "configured AI alias and cannot be modified."
                )

        add_ids, unknown_add = _resolve_labels(gmail, add_names)
        remove_ids, unknown_remove = _resolve_labels(gmail, remove_names)
        unknown = unknown_add + unknown_remove
        if unknown:
            return ToolResult.failed(
                f"Unknown labels: {', '.join(unknown)}. "
                "Create them in the Gmail UI first — labels are not auto-created."
            )

        ok = gmail.modify_labels(message_id, add_ids, remove_ids)
        if not ok:
            return ToolResult.failed(f"Failed to modify labels on {message_id}.")

        added_summary = add_names if add_names else []
        removed_summary = remove_names if remove_names else []
        parts = []
        if added_summary:
            parts.append(f"added {added_summary}")
        if removed_summary:
            parts.append(f"removed {removed_summary}")
        summary = f"Message {message_id}: {', '.join(parts)}."
        logger.info(f"[EmailModifyLabels] {summary}")
        return ToolResult(
            success=True,
            data={
                "message_id": message_id,
                "added": added_summary,
                "removed": removed_summary,
            },
            llm_summary=summary,
        )
