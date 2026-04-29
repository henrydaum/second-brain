"""
tool_email_check — Read mail directly from Gmail (no SQL mirror).

Use scope to pick a common view, or pass a raw Gmail search query for
anything else. Returns message summaries; set include_body=true to also
fetch the body of each result (slower).

Scopes:
    inbox     — recent messages in the INBOX label.
    ai_sent   — messages sent FROM the configured ai_email_address.
    ai_inbox  — messages addressed TO the configured ai_email_address.
    custom    — use the `query` parameter as a raw Gmail search string.

Config:
    ai_email_address — shared with tool_email_send.
"""

import logging

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("tool_email_check")


class EmailCheck(BaseTool):
    name = "email_check"
    description = (
        "Read mail from Gmail. Pick a scope (inbox, ai_sent, ai_inbox, "
        "custom) or pass a raw Gmail query. Returns message summaries; "
        "set include_body=true to also fetch bodies."
    )
    config_settings = [
        (
            "AI Agent Email Address",
            "ai_email_address",
            "Gmail send-as alias (e.g. agent@yourdomain.com).",
            "",
            {"type": "text"},
        ),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["inbox", "ai_sent", "ai_inbox", "custom"],
                "description": "Which view of mail to read. Default 'inbox'.",
                "default": "inbox",
            },
            "query": {
                "type": "string",
                "description": (
                    "Raw Gmail search query, only used when scope='custom' "
                    "(e.g. 'is:unread', 'from:foo@bar.com newer_than:7d')."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum messages to return. Default 20.",
                "default": 20,
            },
            "include_body": {
                "type": "boolean",
                "description": (
                    "If true, fetch the body of each message. Off by default "
                    "to keep responses small."
                ),
                "default": False,
            },
        },
        "required": [],
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

        scope = kwargs.get("scope") or "inbox"
        limit = int(kwargs.get("limit", 20))
        include_body = bool(kwargs.get("include_body", False))
        query = (kwargs.get("query") or "").strip()
        ai_email = (context.config.get("ai_email_address") or "").strip()

        if scope == "inbox":
            summaries = gmail.fetch_inbox(max_results=limit)
            label = "inbox"
        elif scope == "ai_sent":
            if not ai_email:
                return ToolResult.failed("ai_email_address is not set.")
            summaries = gmail.search(f"from:{ai_email}", max_results=limit)
            label = f"sent from {ai_email}"
        elif scope == "ai_inbox":
            if not ai_email:
                return ToolResult.failed("ai_email_address is not set.")
            summaries = gmail.fetch_inbox_aliased(ai_email, max_results=limit)
            label = f"addressed to {ai_email}"
        elif scope == "custom":
            if not query:
                return ToolResult.failed("scope='custom' requires a 'query' argument.")
            summaries = gmail.search(query, max_results=limit)
            label = f"matching {query!r}"
        else:
            return ToolResult.failed(f"Unknown scope: {scope}")

        summaries = summaries or []

        if include_body:
            for s in summaries:
                full = gmail.get_message(s["message_id"])
                if full:
                    s["body_plain"] = full.get("body_plain", "")
                    s["body_html"] = full.get("body_html", "")
                    s["recipients"] = full.get("recipients", "")

        logger.info(f"[EmailCheck] {len(summaries)} message(s) — {label}")

        if not summaries:
            llm_summary = f"No messages {label}."
        else:
            lines = [f"Found {len(summaries)} message(s) {label}:"]
            for i, s in enumerate(summaries, 1):
                read_flag = "" if s.get("is_read") else " [UNREAD]"
                line = (
                    f"{i}. id={s.get('message_id','')}{read_flag}\n"
                    f"   from:    {s.get('sender','')}\n"
                    f"   subject: {s.get('subject','(no subject)')}\n"
                    f"   snippet: {(s.get('snippet','') or '')[:200]}"
                )
                if include_body and s.get("body_plain"):
                    body = s["body_plain"].strip().replace("\r\n", "\n")
                    if len(body) > 1500:
                        body = body[:1500] + "…[truncated]"
                    line += f"\n   body:\n{body}"
                lines.append(line)
            llm_summary = "\n".join(lines)

        return ToolResult(
            success=True,
            data={"emails": summaries, "count": len(summaries), "scope": scope},
            llm_summary=llm_summary,
        )
