"""/conversations — unified picker for conversation history + new chats.

Walks a multi-step form:

    1. Pick a category (or "New conversation").
    2a. (New) Pick an agent profile.
    2b. (New) Pick a category for the new conversation; "Main" is always
        offered, plus an option to type a fresh category.
    2c. (New + new category) Free-text the category name.

       (Existing) Pick one of the 15 most-recent conversations under
       the category.
    3.  (Existing) Pick "Load conversation" or "Delete conversation".
        The step prompt previews the chosen conversation's agent and
        most recent messages.
"""

from __future__ import annotations

import time

from plugins.BaseCommand import BaseCommand
from runtime.notifications import NOTIFICATION_MODES, notification_mode
from state_machine.conversation import FormStep
from state_machine.serialization import latest_state


_LIMIT = 15
_MAIN = "Main"
_NEW_CONV = "➕ New conversation"
_NEW_CAT = "➕ New category"
_LOAD = "Load conversation"
_DELETE = "Delete conversation"
_CHANGE_NOTIF = "Change notification mode"


class ConversationsCommand(BaseCommand):
    name = "conversations"
    description = "Browse, switch, or start conversations"
    category = "Conversation"

    def form(self, args, context):
        db = getattr(context, "db", None)
        if db is None:
            return []

        # Step 1 — pick a top-level category, or branch off to "New".
        cats = _existing_categories(db)
        cat_enum = list(cats) + [_NEW_CONV]
        steps = [FormStep("category", "Choose a conversation category, or start a new conversation.", True, enum=cat_enum, columns=1)]

        picked = args.get("category")
        if not picked:
            return steps

        if picked == _NEW_CONV:
            return steps + _new_conversation_steps(args, context)
        return steps + _existing_conversation_steps(args, context, picked)

    def run(self, args, context):
        runtime = getattr(context, "runtime", None)
        db = getattr(context, "db", None)
        session_key = getattr(context, "session_key", None)
        if runtime is None or db is None or not session_key:
            return "Conversations are not available in this context."

        if args.get("category") == _NEW_CONV:
            return _create_and_switch(args, runtime, session_key)

        cid = _decode_id(args.get("conversation_id"))
        if cid is None:
            return "No conversation selected."

        action = args.get("action") or _LOAD
        if action == _DELETE:
            db.delete_conversation(cid)
            return f"Deleted conversation #{cid}."
        if action == _CHANGE_NOTIF:
            mode = runtime.set_conversation_notification_mode(cid, args.get("mode"))
            return f"Notifications for #{cid} → {mode}."

        # Default: load. load_history reads the conversation's stored
        # state marker, so the agent profile follows the conversation
        # automatically — no need to pass it explicitly.
        result = runtime.load_history(session_key, cid)
        return "\n".join(m for m in result.messages if m).strip() or f"Loaded conversation #{cid}."


class NewCommand(BaseCommand):
    name = "new"
    description = "Start a default Main conversation"
    category = "Conversation"

    def run(self, args, context):
        runtime = getattr(context, "runtime", None)
        db = getattr(context, "db", None)
        session_key = getattr(context, "session_key", None)
        if runtime is None or db is None or not session_key:
            return "Conversations are not available in this context."
        return _create_and_switch({"category": _NEW_CONV, "agent_profile": "default", "new_category": _MAIN}, runtime, session_key)


# ──────────────────────────────────────────────────────────────────────
# Step builders
# ──────────────────────────────────────────────────────────────────────

def _new_conversation_steps(args, context):
    profiles = list((getattr(context, "config", None) or {}).get("agent_profiles") or {})
    if not profiles:
        profiles = ["default"]
    steps = [FormStep("agent_profile", "Choose the agent profile for the new conversation.", True, enum=profiles, columns=1)]
    if not args.get("agent_profile"):
        return steps

    db = getattr(context, "db", None)
    cats = _existing_categories(db) if db else []
    if _MAIN not in cats:
        cats = [_MAIN] + cats
    cat_enum = list(cats) + [_NEW_CAT]
    steps.append(FormStep("new_category", "Choose where to file the new conversation.", True, enum=cat_enum, columns=1))

    if args.get("new_category") == _NEW_CAT:
        steps.append(FormStep("custom_category", "Enter a name for the new category.", True, columns=1))
    return steps


def _existing_conversation_steps(args, context, category):
    db = getattr(context, "db", None)
    rows, _ = db.list_conversations_page(offset=0, limit=_LIMIT, category=_lookup_value(category))
    if not rows:
        return [FormStep("conversation_id", f"No conversations found under '{category}'.", True,
                         enum=["(none)"], enum_labels=["(none)"], columns=1)]

    enum = [str(r.get("id")) for r in rows]
    labels = [_label_for(db, r) for r in rows]
    steps = [FormStep("conversation_id", f"Choose a recent conversation under '{category}'.",
                      True, enum=enum, enum_labels=labels, columns=1)]

    cid = _decode_id(args.get("conversation_id"))
    if cid is None:
        return steps

    prompt = f"What do you want to do with this conversation?\n\n{_preview_for(db, cid) or ''}".strip()
    steps.append(FormStep("action", prompt, True, enum=[_LOAD, _DELETE, _CHANGE_NOTIF], columns=1))
    if args.get("action") == _CHANGE_NOTIF:
        steps.append(FormStep("mode", "Choose how this conversation should notify you while it runs in the background.", True, enum=list(NOTIFICATION_MODES), columns=1))
    return steps


# ──────────────────────────────────────────────────────────────────────
# Category helpers
# ──────────────────────────────────────────────────────────────────────

def _existing_categories(db) -> list[str]:
    """Distinct, user-facing category labels currently in the DB.

    NULL/empty categories surface as ``Main`` so the bucket has a name.
    """
    out: list[str] = []
    for v in db.list_conversation_categories():
        label = _MAIN if v in (None, "") else v
        if label not in out:
            out.append(label)
    return out


def _lookup_value(label: str) -> str:
    """Map a UI label back to the value stored in the DB."""
    return "" if label == _MAIN else label


# ──────────────────────────────────────────────────────────────────────
# Listing + previewing
# ──────────────────────────────────────────────────────────────────────

def _label_for(db, row: dict) -> str:
    title = (row.get("title") or "").strip() or "(untitled)"
    rel = _relative_time(row.get("updated_at"))
    return f"{title}  ({rel})" if rel else title


def _relative_time(timestamp) -> str:
    """Format an absolute timestamp as a coarse "(N units ago)" string."""
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return ""
    delta = max(0, time.time() - ts)
    units = (
        (60, "second", "seconds"),
        (60, "minute", "minutes"),
        (24, "hour", "hours"),
        (7, "day", "days"),
        (4, "week", "weeks"),
        (12, "month", "months"),
        (None, "year", "years"),
    )
    value = delta
    for step, singular, plural in units:
        if step is None or value < step:
            n = int(value) if value >= 1 else 1
            return f"just now" if singular == "second" and n < 5 else f"{n} {singular if n == 1 else plural} ago"
        value /= step
    return ""


def _agent_for(db, conversation_id) -> str:
    marker = latest_state(db.get_conversation_messages(conversation_id)) or {}
    return (marker.get("profile_override") or marker.get("active_agent_profile") or "").strip()


def _notification_mode_for(db, conversation_id) -> str:
    return notification_mode((latest_state(db.get_conversation_messages(conversation_id)) or {}).get("notification_mode"))


def _preview_for(db, conversation_id) -> str:
    """A scannable header for the Load/Delete step.

    Shows the agent profile and the last 1-2 chat turns, truncated. Only
    rendered after the user picks a conversation, so the cost is paid
    once, not for every list item.
    """
    msgs = db.get_conversation_messages(conversation_id) or []
    agent = _agent_for(db, conversation_id) or "(unknown)"
    mode = _notification_mode_for(db, conversation_id)
    title = ""
    row = db.get_conversation(conversation_id) if hasattr(db, "get_conversation") else None
    if row:
        title = (row.get("title") or "").strip()
    snippets: list[str] = []
    for m in reversed(msgs):
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        snippets.append(f"{role}: {_truncate(content, 120)}")
        if len(snippets) >= 2:
            break
    snippets.reverse()
    head = f"{title or '(untitled)'} · agent: {agent} · notifications: {mode}"
    return head + ("\n" + "\n".join(snippets) if snippets else "")


def _truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ──────────────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────────────

def _create_and_switch(args, runtime, session_key) -> str:
    profile = (args.get("agent_profile") or "").strip() or "default"
    chosen = (args.get("new_category") or "").strip()
    category = (args.get("custom_category") or "").strip() if chosen == _NEW_CAT else chosen
    title = f"New conversation ({category or _MAIN})"
    db_category = _lookup_value(category or _MAIN)
    new_id = runtime.create_conversation(title, kind="user", category=db_category or None)
    if new_id is None:
        return "Failed to create conversation."
    existing = runtime.sessions.get(session_key)
    if existing is not None and existing.conversation_id not in (None, new_id):
        runtime.close_session(session_key)
    runtime.load_conversation(session_key, new_id, agent_profile=profile)
    return f"Started new conversation #{new_id} under '{category or _MAIN}'.\nAgent: {profile}"


def _decode_id(value) -> int | None:
    if value in (None, "", "(none)"):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith("#"):
        text = text[1:]
    head = text.split(" ", 1)[0].strip()
    try:
        return int(head)
    except (TypeError, ValueError):
        return None
