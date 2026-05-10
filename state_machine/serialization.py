from __future__ import annotations

"""Persistence helpers for the parallel state-machine runtime.

Normal chat history stays in the exact existing role/content/tool_call shape.
State-machine metadata is stored as system JSON marker rows so no database
schema changes are needed and LLM history reconstruction can ignore it.
"""

import json
import time
from typing import Any

from state_machine.forms import history_tool_calls_from_content

STATE_MARKER = "__second_brain_state_machine__"
COMPACTION_MARKER = "__second_brain_compaction__"


def pack_state(state: dict[str, Any]) -> str:
    return json.dumps({STATE_MARKER: True, "state": state}, default=str)


def unpack_state(content: str) -> dict[str, Any] | None:
    try:
        data = json.loads(content or "")
    except (TypeError, json.JSONDecodeError):
        return None
    return data.get("state") if isinstance(data, dict) and data.get(STATE_MARKER) else None


def pack_compaction(summary: str, tail_count: int = 2) -> str:
    return json.dumps({COMPACTION_MARKER: True, "summary": summary, "tail_count": tail_count, "created_at": time.time()}, default=str)


def unpack_compaction(content: str) -> dict[str, Any] | None:
    try:
        data = json.loads(content or "")
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get(COMPACTION_MARKER) else None


def is_state_marker(row: dict[str, Any]) -> bool:
    return row.get("role") == "system" and unpack_state(row.get("content") or "") is not None


def is_compaction_marker(row: dict[str, Any]) -> bool:
    return row.get("role") == "system" and unpack_compaction(row.get("content") or "") is not None


def messages_to_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert DB rows to provider-compatible history, skipping SM markers."""
    compact = latest_compaction(rows)
    if compact:
        idx, marker = compact
        rows = rows[idx + 1:]
        summary = (marker.get("summary") or "").strip()
        history = ([{"role": "user", "content": f"[Conversation summary from earlier]\n{summary}"}, {"role": "assistant", "content": "Understood - I have the earlier context."}] if summary else [])
    else:
        history = []
    for msg in rows:
        if is_state_marker(msg) or is_compaction_marker(msg) or msg.get("role") == "system":
            continue
        role, content = msg.get("role"), msg.get("content") or ""
        if role == "assistant":
            packed = history_tool_calls_from_content(content)
            history.append({"role": "assistant", "content": packed.get("content"), "tool_calls": packed["tool_calls"]} if packed else {"role": "assistant", "content": content})
        elif role == "tool":
            history.append({"role": "tool", "tool_call_id": msg.get("tool_call_id"), "name": msg.get("tool_name"), "content": content})
        elif role == "user":
            history.append({"role": "user", "content": content})
    return heal_orphans(history)


def heal_orphans(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair interrupted assistant/tool-call sequences before LLM replay."""
    call_ids = {tc.get("id") for m in history if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])}
    kept = [m for m in history if m.get("role") != "tool" or m.get("tool_call_id") in call_ids]
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(kept):
        msg = kept[i]; out.append(msg); i += 1
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        got = set()
        while i < len(kept) and kept[i].get("role") == "tool":
            got.add(kept[i].get("tool_call_id")); out.append(kept[i]); i += 1
        for tc in msg["tool_calls"]:
            if tc.get("id") not in got:
                out.append({"role": "tool", "tool_call_id": tc.get("id"), "name": tc.get("function", {}).get("name") or tc.get("name"), "content": json.dumps({"error": "Tool execution was interrupted before completion."})})
    return out


def save_history_message(db, conversation_id: int, msg: dict[str, Any]) -> None:
    """Persist one provider-history message using the existing DB contract."""
    role, content = msg.get("role"), msg.get("content") or ""
    if role == "assistant" and msg.get("tool_calls"):
        content = json.dumps({"content": msg.get("content"), "tool_calls": msg["tool_calls"]})
    db.save_message(conversation_id, role, content, tool_call_id=msg.get("tool_call_id"), tool_name=msg.get("name"))


def save_state_marker(db, conversation_id: int, state: dict[str, Any]) -> None:
    db.save_message(conversation_id, "system", pack_state(state))


def save_compaction_marker(db, conversation_id: int, summary: str, tail_count: int = 2) -> None:
    db.save_message(conversation_id, "system", pack_compaction(summary, tail_count))


def latest_state(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in reversed(rows):
        state = unpack_state(row.get("content") or "")
        if state is not None:
            return state
    return None


def latest_compaction(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    for i in range(len(rows) - 1, -1, -1):
        marker = unpack_compaction(rows[i].get("content") or "")
        if marker is not None:
            return i, marker
    return None
