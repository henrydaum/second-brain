"""Slash command plugin for `/usage` — what model calls have consumed.

Reads the kernel's ``llm_usage`` table through ``usage.read``: one row per
model call, four counts each (input, cache reads, cache writes, output — see
``guest.llm.USAGE_FIELDS``). Kernel rather than a store package because it
reads a kernel table, and a meter that disappears with a package is worse
than none.

The first step *is* the answer. The picker's prompt carries this
conversation's usage, today, the last week and all time, so calling `/usage`
shows everything it reasonably can at once; the buttons are for going deeper.
Every read is SAFE and scoped to the caller's own user, so none of it asks.

Two rules from the table carry through to the display. A count nobody
reported is ``—``, never 0 — ``None`` means the provider did not say. And the
cache counts are *shares* of input, not additions to it, so the cache hit rate
is ``cache_read / input`` and the total is input plus output.
"""

import time

from guest.bases import BaseCommand
from guest.forms import FormStep

MODELS = "models"
DAYS = "days"
CONVERSATIONS = "conversations"
SOURCES = "sources"
VIEWS = (MODELS, DAYS, CONVERSATIONS, SOURCES)

LABELS = {
    MODELS: "All time by model",
    DAYS: "Last 14 days",
    CONVERSATIONS: "Top conversations",
    SOURCES: "By source",
}

TREND_DAYS = 14
DAY = 86400


class UsageCommand(BaseCommand):
    """Show token usage for this conversation and over time."""

    name = "usage"
    description = "Show token usage for this conversation and over time"
    category = "Conversation"
    requests = ["usage.read", "config.read"]

    def form(self, sdk, args):
        """One step: the overview as its prompt, deeper views as buttons.

        Required, for the reason ``/mode``'s picker is: an optional step never
        suspends, so the overview would never be on screen with the buttons
        under it. A view named up front (``/usage days``, ``/usage 42``)
        skips it.
        """
        if args.get("view"):
            return []
        return [FormStep(
            "view", _overview(sdk), True, enum=list(VIEWS),
            enum_labels=[LABELS[view] for view in VIEWS], columns=2)]

    def run(self, sdk, args):
        """Render the view asked for, or the overview when none was."""
        view = str(args.get("view") or "").strip().lower()
        if not view:
            return _overview(sdk)
        if view.lstrip("#").isdigit():
            return _conversation(sdk, int(view.lstrip("#")))
        if view == MODELS:
            return _grouped(sdk, "All time, by model", "model", "Model",
                            since=None)
        if view == DAYS:
            return _days(sdk)
        if view == CONVERSATIONS:
            return _conversations(sdk)
        if view == SOURCES:
            return _grouped(sdk, "All time, by source", "origin", "Source",
                            since=None)
        return (f"Unknown view: {view}. Use one of: " + ", ".join(VIEWS)
                + ", or a conversation id.")


# ── views ─────────────────────────────────────────────────────────────

def _overview(sdk) -> str:
    """Everything that fits on the first screen."""
    parts = []
    here = sdk.usage.read("current", group_by="model")
    if here.get("totals") and here["totals"].get("calls"):
        parts.append(_conversation_block(
            sdk, "This conversation", here))
    elif here.get("conversation_id") is not None:
        parts.append("No model calls in this conversation yet.")

    now = time.time()
    periods = [
        ("Today", _midnight(now)),
        ("Last 7 days", now - 7 * DAY),
        ("All time", None),
    ]
    rows, unreported = [], 0
    for label, since in periods:
        totals = sdk.usage.read(since=since)["totals"] or {}
        if since is None:
            unreported = totals.get("unreported") or 0
        rows.append([label, *_summary_cells(totals)])
    if not any(row[1] != "0" for row in rows):
        parts.append("No model calls recorded yet.")
    else:
        parts.append(sdk.md.table(
            ["Period", "Calls", "Input", "Cache hit", "Output", "Total"],
            rows))
    parts.extend(_footnotes(sdk, unreported))
    return "\n\n".join(parts)


def _conversation(sdk, conversation_id: int) -> str:
    """One conversation's card — the overview's first block, for any id."""
    data = sdk.usage.read(conversation_id, group_by="model")
    if not (data.get("totals") or {}).get("calls"):
        return f"No model calls recorded for conversation {conversation_id}."
    return "\n\n".join([
        _conversation_block(sdk, f"Conversation {conversation_id}", data),
        *_footnotes(sdk, data["totals"].get("unreported") or 0),
    ])


def _conversation_block(sdk, title: str, data: dict) -> str:
    """A card of the totals, then the split by model when there is one."""
    totals = data["totals"]
    latest = data.get("latest") or {}
    calls = f"{totals['calls']:,}"
    if totals.get("failed"):
        calls += f" ({totals['failed']:,} failed)"
    pairs = [
        ["Model calls", calls],
        ["Input", _n(totals.get("input_tokens"))],
        ["Cache hits", _with_pct(totals.get("cache_read_tokens"),
                                 totals.get("input_tokens"))],
        ["Cache writes", _with_pct(totals.get("cache_write_tokens"),
                                   totals.get("input_tokens"))],
        ["Output", _n(totals.get("output_tokens"))],
        ["Total", _n(_total(totals))],
    ]
    if latest.get("input_tokens") is not None:
        pairs.append(["Context now",
                      f"{_n(latest['input_tokens'])} (last call)"])
    pairs.append(["Span", _span(totals)])
    parts = [sdk.md.card(title, pairs)]
    groups = data.get("groups") or []
    if len(groups) > 1:
        parts.append(_group_table(sdk, groups, "Model"))
    return "\n\n".join(parts)


def _grouped(sdk, title: str, group_by: str, column: str, *, since) -> str:
    data = sdk.usage.read(since=since, group_by=group_by)
    groups = data.get("groups") or []
    if not groups:
        return "No model calls recorded yet."
    return "\n\n".join([
        f"**{title}**",
        _group_table(sdk, groups, column),
        *_footnotes(sdk, (data.get("totals") or {}).get("unreported") or 0),
    ])


def _days(sdk) -> str:
    """The trend: one row per day, newest first, with the window's total."""
    since = _midnight(time.time()) - (TREND_DAYS - 1) * DAY
    data = sdk.usage.read(since=since, group_by="day", limit=TREND_DAYS)
    groups = data.get("groups") or []
    if not groups:
        return f"No model calls in the last {TREND_DAYS} days."
    rows = [[g["key"], *_summary_cells(g)] for g in groups]
    rows.append(["**Total**", *_summary_cells(data["totals"])])
    return "\n\n".join([
        f"**Last {TREND_DAYS} days**",
        sdk.md.table(
            ["Day", "Calls", "Input", "Cache hit", "Output", "Total"], rows),
        *_footnotes(sdk, data["totals"].get("unreported") or 0),
    ])


def _conversations(sdk) -> str:
    data = sdk.usage.read(group_by="conversation", limit=15)
    groups = [g for g in data.get("groups") or [] if g.get("key") is not None]
    if not groups:
        return "No conversation has any recorded model calls yet."
    rows = [[g["key"], _title(g), *_summary_cells(g)] for g in groups]
    return "\n\n".join([
        "**Top conversations, by tokens**",
        sdk.md.table(["#", "Title", "Calls", "Input", "Cache hit", "Output",
                      "Total"], rows),
        "`/usage <#>` shows one conversation in full.",
    ])


# ── pieces ────────────────────────────────────────────────────────────

def _group_table(sdk, groups, column: str) -> str:
    rows = [[g.get("key") or "—", *_summary_cells(g)] for g in groups]
    return sdk.md.table(
        [column, "Calls", "Input", "Cache hit", "Output", "Total"], rows)


def _summary_cells(row: dict) -> list:
    """Calls, input, cache hit %, output, total — the columns every table has."""
    return [
        f"{row.get('calls') or 0:,}",
        _n(row.get("input_tokens")),
        _pct(row.get("cache_read_tokens"), row.get("input_tokens")),
        _n(row.get("output_tokens")),
        _n(_total(row)),
    ]


def _footnotes(sdk, unreported: int) -> list:
    """Only what changes how the numbers read, and only when it applies."""
    notes = []
    if unreported:
        noun = "call" if unreported == 1 else "calls"
        notes.append(f"{unreported:,} {noun} did not report token counts; "
                     "they are counted as calls but not as tokens.")
    try:
        days = int(sdk.config.read("data_retention_days") or 0)
    except (sdk.Failed, TypeError, ValueError):
        days = 0
    if days > 0:
        notes.append(f"History is kept for {days} days "
                     "(`data_retention_days`), so all time means that long.")
    return notes


def _total(row: dict):
    """Input plus output; ``None`` only when neither was reported."""
    counts = [row.get("input_tokens"), row.get("output_tokens")]
    if all(count is None for count in counts):
        return None
    return sum(count or 0 for count in counts)


def _n(value) -> str:
    """A count, readable at a glance. ``—`` means nobody reported it."""
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:,}"


def _pct(part, whole) -> str:
    """``part`` as a share of ``whole`` — how the cache hit rate is derived."""
    if part is None or not whole:
        return "—"
    return f"{100 * part / whole:.1f}%"


def _with_pct(part, whole) -> str:
    if part is None:
        return "—"
    pct = _pct(part, whole)
    return _n(part) if pct == "—" else f"{_n(part)} ({pct} of input)"


def _span(row: dict) -> str:
    first, last = row.get("first_ts"), row.get("last_ts")
    if first is None:
        return "—"
    return f"{_when(first)} → {_when(last)}"


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _midnight(now: float) -> float:
    """The start of today, local time."""
    day = time.localtime(now)
    return time.mktime((day.tm_year, day.tm_mon, day.tm_mday,
                        0, 0, 0, 0, 0, -1))


def _title(group: dict) -> str:
    title = (group.get("title") or "").strip() or "(untitled)"
    return title if len(title) <= 40 else title[:39] + "…"
