"""/conversations picker and /new conversation command."""

from guest.bases import BaseCommand
from guest.forms import FormStep


_LIMIT = 15
_MAIN = "Main"
_NEW_CAT = "Add New category"
_LOAD = "Load conversation"
_DELETE = "Delete conversation"
_CHANGE_CATEGORY = "Change category"
_CHANGE_NOTIF = "Change notification mode"
_NOTIFICATION_MODES = ("on", "off")


class ConversationsCommand(BaseCommand):
    """Browse, switch, or manage current-user conversations."""

    name = "conversations"
    description = "Browse, switch, or manage conversations"
    category = "Conversation"
    # Deleting is the only destructive action here; loading and relabelling
    # are not, and asking about them would be the approval fatigue this design
    # exists to avoid. Declared so the state machine asks before the body runs
    # rather than leaving ``conv.delete`` to the execution-time approver.
    # Spelled out rather than written as ``(_DELETE,)``: declarations are read
    # by AST, which sees literals and not a name it would have to resolve.
    approval_actions = ("Delete conversation",)
    approval_actor_id = "user"
    requests = [
        "conv.list", "conv.read", "conv.load", "conv.delete",
        "conv.set_category", "conv.set_notification_mode", "session.get",
    ]

    def form(self, sdk, args):
        """Build category, conversation, action, and action-value steps."""
        try:
            overview = sdk.conv.list(details=True, limit=_LIMIT)
        except sdk.Failed:
            return []
        categories = _category_labels(overview.get("categories"))
        steps = [FormStep(
            "category",
            "Choose a conversation category.",
            True,
            enum=categories,
            columns=1,
        )]
        category = args.get("category")
        if not category:
            return steps
        return steps + _existing_conversation_steps(sdk, args, category)

    def run(self, sdk, args):
        """Execute the selected lifecycle action."""
        if not _session_available(sdk):
            return "Conversations are not available in this context."
        cid = _decode_id(args.get("conversation_id"))
        if cid is None:
            return "No conversation selected."

        action = args.get("action") or _LOAD
        try:
            if action == _DELETE:
                if not sdk.conv.delete(cid):
                    return "No such conversation."
                return f"Deleted conversation #{cid}."
            if action == _CHANGE_NOTIF:
                mode = sdk.conv.set_notification_mode(
                    cid, args.get("mode"))
                return f"Notifications for #{cid} → {mode}."
            if action == _CHANGE_CATEGORY:
                category = _resolve_category(args)
                if not sdk.conv.set_category(
                    cid, _lookup_value(category) or None
                ):
                    return "No such conversation."
                return f"Conversation #{cid} moved to '{category}'."

            result = sdk.conv.load(cid)
        except sdk.Failed as exc:
            if "not available to this user" in exc.error.lower():
                return "No such conversation."
            raise
        # ``callable_output`` first, ``messages`` as the fallback: the load
        # confirmation is this command's own output, and reads back on the
        # channel a command answers on.
        said = result.get("callable_output") or result.get("messages") or []
        return "\n".join(
            message for message in said if message
        ).strip() or f"Loaded conversation #{cid}."


def _existing_conversation_steps(sdk, args, category):
    overview = sdk.conv.list(
        details=True, category=_lookup_value(category), limit=_LIMIT)
    rows = overview.get("items") or []
    if not rows:
        return [FormStep(
            "conversation_id",
            f"No conversations found under '{category}'.",
            True,
            enum=["(none)"],
            enum_labels=["(none)"],
            columns=1,
        )]

    steps = [FormStep(
        "conversation_id",
        f"Choose a recent conversation under '{category}'.",
        True,
        enum=[str(row.get("id")) for row in rows],
        enum_labels=[_label_for(row) for row in rows],
        columns=1,
    )]
    cid = _decode_id(args.get("conversation_id"))
    if cid is None:
        return steps

    preview = _preview_for(sdk, cid)
    prompt = (
        "What do you want to do with this conversation?\n\n"
        f"{preview or ''}"
    ).strip()
    steps.append(FormStep(
        "action",
        prompt,
        True,
        enum=[_LOAD, _DELETE, _CHANGE_CATEGORY, _CHANGE_NOTIF],
        columns=1,
    ))
    if args.get("action") == _CHANGE_CATEGORY:
        choices = _category_choices(overview.get("categories"))
        steps.append(FormStep(
            "target_category",
            "Choose the new category.",
            True,
            enum=choices + [_NEW_CAT],
            columns=1,
        ))
        if args.get("target_category") == _NEW_CAT:
            steps.append(FormStep(
                "custom_category",
                "Enter a name for the new category.",
                True,
                columns=1,
            ))
    if args.get("action") == _CHANGE_NOTIF:
        steps.append(FormStep(
            "mode",
            "Choose how this conversation should notify you while it runs "
            "in the background.",
            True,
            enum=list(_NOTIFICATION_MODES),
            columns=1,
        ))
    return steps


def _category_labels(entries):
    """Labels for the category picker.

    ``conv.list`` answers ``{"category": ..., "count": n}`` per bucket, counted
    across the whole table rather than over the page it sent. Bare values are
    still accepted so this keeps working against an older kernel.
    """
    labels = []
    for entry in entries or []:
        value = entry.get("category") if isinstance(entry, dict) else entry
        label = _MAIN if value in (None, "") else value
        if label not in labels:
            labels.append(label)
    return labels


def _category_choices(values):
    labels = _category_labels(values)
    return labels if _MAIN in labels else [_MAIN] + labels


def _lookup_value(label):
    return "" if label == _MAIN else label


def _resolve_category(args):
    chosen = (args.get("target_category") or "").strip()
    return (
        (args.get("custom_category") or "").strip()
        if chosen == _NEW_CAT
        else chosen
    ) or _MAIN


def _label_for(row):
    title = (row.get("title") or "").strip() or "(untitled)"
    relative = row.get("updated_ago") or ""
    return f"{title}  ({relative})" if relative else title


def _preview_for(sdk, conversation_id):
    data = sdk.conv.read(conversation_id, details=True)
    messages = data.get("messages") or []
    conversation = data.get("conversation") or {}
    title = (conversation.get("title") or "").strip() or "(untitled)"
    agent = data.get("agent_profile") or "(unknown)"
    mode = data.get("notification_mode") or "off"
    snippets = []
    for message in reversed(messages):
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (message.get("content") or "").strip()
        if not content:
            continue
        clean = content.replace("\n", " ").strip()
        snippets.append(
            f"{role}: "
            f"{sdk.text.truncate(clean, 120, suffix='…')}"
        )
        if len(snippets) >= 2:
            break
    snippets.reverse()
    card = sdk.md.card(
        title, [("Agent", agent), ("Notifications", mode)])
    quoted = sdk.md.quote("\n".join(snippets))
    return card + (f"\n\n{quoted}" if snippets else "")


def _session_available(sdk):
    try:
        session = sdk.session.get()
        sdk.conv.list(limit=1)
        return bool(session)
    except sdk.Failed:
        return False


def _decode_id(value):
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
