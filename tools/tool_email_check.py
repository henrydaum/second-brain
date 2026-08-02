"""Read Gmail messages without mirroring them into the database."""

dependencies_files = ["tools/helpers/email_context.py"]
dependencies_pip = []
requests = ["service.call", "config.read", "session.get", "conv.read"]

from guest.bases import BaseTool
from .email_context import allowed_addresses, is_main_conversation


def _alias_clause(addresses, include_from=False):
    operations = ["to", "cc", "bcc", "deliveredto"]
    if include_from:
        operations.append("from")
    return " OR ".join(
        f'{operation}:"{address}"'
        for address in addresses for operation in operations
    )


class EmailCheck(BaseTool):
    name = "email_check"
    description = (
        "Read Gmail by inbox, AI-sent, AI-inbox, or custom query. Returns "
        "summaries and optionally message bodies."
    )
    config_settings = [
        ("AI Agent Email Addresses", "ai_email_addresses",
         "Gmail send-as aliases the agent may autonomously access.", [],
         {"type": "json_list"}),
    ]
    parameters = {
        "type": "object",
        "properties": {
            "scope": {"type": "string",
                      "enum": ["inbox", "ai_sent", "ai_inbox", "custom"],
                      "default": "inbox"},
            "query": {"type": "string", "description": "Raw Gmail query for custom scope."},
            "limit": {"type": "integer", "default": 20},
            "include_body": {"type": "boolean", "default": False},
        },
        "required": [],
    }
    requires_services = ["gmail"]

    def run(self, sdk, **kwargs):
        scope = str(kwargs.get("scope") or "inbox")
        try:
            limit = max(1, min(int(kwargs.get("limit", 20)), 100))
        except (TypeError, ValueError):
            limit = 20
        include_body = bool(kwargs.get("include_body", False))
        query = str(kwargs.get("query") or "").strip()
        allowed = allowed_addresses(sdk)

        if not is_main_conversation(sdk):
            if not allowed:
                return sdk.fail("Non-main conversation has no configured AI email access.")
            if scope == "inbox":
                scope = "ai_inbox"
            elif scope == "custom":
                if not query:
                    return sdk.fail("scope='custom' requires query.")
                query = f"({query}) AND ({_alias_clause(allowed, True)})"

        if scope == "inbox":
            messages = sdk.services.call(
                "gmail", "fetch_inbox", max_results=limit) or []
            label = "inbox"
        elif scope in {"ai_sent", "ai_inbox"}:
            if not allowed:
                return sdk.fail("ai_email_addresses is empty.")
            clause = (" OR ".join(f'from:"{value}"' for value in allowed)
                      if scope == "ai_sent" else _alias_clause(allowed))
            messages = sdk.services.call(
                "gmail", "search", query=f"({clause})", max_results=limit) or []
            label = "sent from AI aliases" if scope == "ai_sent" else "addressed to AI aliases"
        elif scope == "custom":
            if not query:
                return sdk.fail("scope='custom' requires query.")
            messages = sdk.services.call(
                "gmail", "search", query=query, max_results=limit) or []
            label = f"matching {query!r}"
        else:
            return sdk.fail(f"Unknown scope: {scope}")

        if include_body:
            for message in messages:
                full = sdk.services.call(
                    "gmail", "get_message", message_id=message.get("message_id"))
                if full:
                    for key in ("body_plain", "body_html", "recipients", "cc"):
                        message[key] = full.get(key, "")

        lines = [f"Found {len(messages)} message(s) {label}:"]
        for index, message in enumerate(messages, 1):
            unread = " [UNREAD]" if not message.get("is_read") else ""
            lines.append(
                f"{index}. id={message.get('message_id', '')}{unread}\n"
                f"   from: {message.get('sender', '')}\n"
                f"   subject: {message.get('subject') or '(no subject)'}\n"
                f"   snippet: {str(message.get('snippet') or '')[:200]}"
            )
            if include_body and message.get("body_plain"):
                body = str(message["body_plain"]).strip()[:1500]
                lines[-1] += f"\n   body:\n{body}"
        summary = "\n".join(lines) if messages else f"No messages {label}."
        sdk.log(f"email check returned {len(messages)} message(s) {label}")
        return sdk.ok(
            {"emails": messages, "count": len(messages), "scope": scope},
            llm_summary=summary,
        )
