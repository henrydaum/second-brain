"""Shared, SDK-only email access guards."""

dependencies_files = []
dependencies_pip = []


def allowed_addresses(sdk):
    raw = sdk.config.read("ai_email_addresses") or []
    if not isinstance(raw, list):
        return []
    return [str(value).strip() for value in raw if str(value).strip()]


def is_main_conversation(sdk):
    session = sdk.session.get() or {}
    conversation_id = session.get("conversation_id")
    if not conversation_id:
        return True
    row = (sdk.conv.read(conversation_id) or {}).get("conversation") or {}
    return str(row.get("category") or "").strip() in {"", "Main"}


def message_involves(message, addresses):
    haystack = " ".join([
        message.get("sender", ""), message.get("recipients", ""),
        message.get("cc", ""),
    ]).lower()
    return any(address.lower() in haystack for address in addresses)
