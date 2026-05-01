"""
Conversation history utilities.

``heal_orphan_tool_calls`` repairs invalid message sequences caused by
interrupted tool execution — e.g. the process being killed mid-tool
leaves an assistant message with ``tool_calls`` but no matching
``tool_result``. Loading such a history would trip the LLM provider's
strict message-sequence validation.
"""

import json
import logging

logger = logging.getLogger("Agent")

_INTERRUPTED_CONTENT = json.dumps(
    {"error": "Tool execution was interrupted before completion — previous session ended unexpectedly."}
)


def messages_to_history(messages: list[dict]) -> list[dict]:
    """Convert persisted conversation_messages rows into Agent.history dicts.

    Reverses the role-specific encoding done by Agent's _on_message persistence
    path (assistant turns with tool_calls are JSON-packed into the content column).
    Heals orphan tool_calls so the result is safe to feed back to a provider.
    """
    history: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        content = msg.get("content") or ""
        if role == "assistant":
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    history.append({
                        "role": "assistant",
                        "content": parsed.get("content"),
                        "tool_calls": parsed["tool_calls"],
                    })
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
            history.append({"role": "assistant", "content": content})
        elif role == "tool":
            history.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "content": content,
            })
        else:
            history.append({"role": role, "content": content})
    heal_orphan_tool_calls(history)
    return history


def heal_orphan_tool_calls(history: list[dict]) -> tuple[int, int]:
    """Repair a message history in place.

    Two passes:
        1. Drop ``role=tool`` messages whose ``tool_call_id`` does not appear
           in any preceding assistant ``tool_calls`` entry.
        2. For each assistant ``tool_call`` with no matching ``tool_result``,
           insert a synthetic error tool-result immediately after the
           assistant message.

    Returns (removed, inserted) for logging.
    """
    # Collect every tool_call id referenced by an assistant message
    call_ids: set[str] = set()
    for msg in history:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                if tc_id:
                    call_ids.add(tc_id)

    # Pass 1 — drop stray tool messages
    kept: list[dict] = []
    removed = 0
    for msg in history:
        if msg.get("role") == "tool" and msg.get("tool_call_id") not in call_ids:
            removed += 1
            continue
        kept.append(msg)

    # Pass 2 — insert synthetic results after the existing tool_results of
    # each assistant-with-tool_calls group, so surviving results keep their
    # original position and synthetics fill the gaps in tool_call order.
    repaired: list[dict] = []
    inserted = 0
    i = 0
    while i < len(kept):
        msg = kept[i]
        repaired.append(msg)
        i += 1
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue

        got_ids: set[str] = set()
        while i < len(kept) and kept[i].get("role") == "tool":
            tm = kept[i]
            repaired.append(tm)
            tcid = tm.get("tool_call_id")
            if tcid:
                got_ids.add(tcid)
            i += 1

        for tc in msg["tool_calls"]:
            tc_id = tc.get("id")
            if not tc_id or tc_id in got_ids:
                continue
            name = tc.get("function", {}).get("name") or tc.get("name") or "unknown"
            repaired.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": name,
                "content": _INTERRUPTED_CONTENT,
            })
            inserted += 1

    history[:] = repaired
    if removed or inserted:
        logger.info(
            f"Healed history: removed {removed} stray tool result(s), "
            f"inserted {inserted} synthetic result(s) for orphan tool_call(s)"
        )
    return removed, inserted
