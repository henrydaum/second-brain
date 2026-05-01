"""
tool_email_check — Read mail directly from Gmail (no SQL mirror).

Use scope to pick a common view, or pass a raw Gmail search query for
anything else. Returns message summaries; set include_body=true to also
fetch the body of each result (slower).

Scopes:
    inbox     — recent messages in the INBOX label.
    ai_sent   — messages sent FROM any configured ai_email_addresses entry.
    ai_inbox  — messages addressed TO any configured ai_email_addresses entry.
    custom    — use the `query` parameter as a raw Gmail search string.

Config:
    ai_email_addresses — list of Gmail send-as aliases the agent may use.
                         Empty list = no agent access (subagents fail).
                         Shared with tool_email_send and tool_email_mark_read.
"""

import logging

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("tool_email_check")


def _allowed_addresses(config) -> list[str]:
    raw = config.get("ai_email_addresses") or []
    if not isinstance(raw, list):
        return []
    return [str(a).strip().lower() for a in raw if str(a).strip()]


def _alias_scope_clause(allowed: list[str], include_from: bool = False) -> str:
    ops = ["to", "cc", "bcc", "deliveredto"] + (["from"] if include_from else [])
    return " OR ".join(f'{op}:"{a}"' for a in allowed for op in ops)


class EmailCheck(BaseTool):
    name = "email_check"
    description = (
        "Read mail from Gmail. Pick a scope (inbox, ai_sent, ai_inbox, "
        "custom) or pass a raw Gmail query. Returns message summaries; "
        "set include_body=true to also fetch bodies."
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
        allowed = _allowed_addresses(context.config)

        # Subagent guard: subagents can only read mail involving an allowed
        # address. Empty list = no access.
        if context.is_subagent:
            if not allowed:
                return ToolResult.failed(
                    "Subagent context but ai_email_addresses is empty — no "
                    "mail access. Configure it under Settings → Plugin Config "
                    "→ AI Agent Email Addresses."
                )
            if scope == "inbox":
                logger.warning("[EmailCheck] Subagent context: rewriting scope 'inbox' → 'ai_inbox'.")
                scope = "ai_inbox"
            elif scope == "custom":
                if not query:
                    return ToolResult.failed("scope='custom' requires a 'query' argument.")
                scope_clause = _alias_scope_clause(allowed, include_from=True)
                query = f"({query}) AND ({scope_clause})"
                logger.info(f"[EmailCheck] Subagent scope=custom rewritten: {query}")

        if scope == "inbox":
            summaries = gmail.fetch_inbox(max_results=limit)
            label = "inbox"
        elif scope == "ai_sent":
            if not allowed:
                return ToolResult.failed("ai_email_addresses is empty.")
            q = " OR ".join(f'from:"{a}"' for a in allowed)
            summaries = gmail.search(f"({q})", max_results=limit)
            label = f"sent from {', '.join(allowed)}"
        elif scope == "ai_inbox":
            if not allowed:
                return ToolResult.failed("ai_email_addresses is empty.")
            q = _alias_scope_clause(allowed)
            summaries = gmail.search(f"({q})", max_results=limit)
            label = f"addressed to {', '.join(allowed)}"
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
