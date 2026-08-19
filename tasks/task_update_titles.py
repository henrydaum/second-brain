"""One-shot conversation titler.

Fired by the ``update_titles`` cron job, created ``* * * * *`` — every
minute — by ``on_install``; the sweep is one cheap SELECT and exits
immediately when nothing is ripe. A conversation with new messages since
the last sweep is titled once it is *ripe*: the agent has replied AND
either the first agent reply is at least ``title_delay_minutes`` old
(default 10 — the delay buys extra context so similar openers don't
collapse into identical titles) or the conversation already carries
``_EARLY_MESSAGES`` user/agent messages (busy conversations earn their
title early). Only titles that still look kernel-generated ("New
Conversation", "New conversation (Main)", empty, or a "/clear"-stamped
"... (cleared)") are replaced, and only once — a real title (from us or a
user rename) is never overwritten, matching the major chat providers. The
high-water mark advances when a row is processed or already titled (so
nothing replays every tick), but *not* while a row merely isn't ripe yet
— it must come back next minute. Each write emits
``conversation_changed`` with ``action='retitled'`` so frontends refresh
sidebars/banners live.

Migrated to the SDK, and three things changed shape:

- The candidate query reads ``my_conversations``, the kernel-owned virtual
  table name that expands to the current user's rows. Reading
  ``conversations`` directly is refused, which is the point — a sweep that
  silently retitled everybody's conversations would be a worse bug than a
  refusal.
- ``title_update_llm_profile`` now names an **LLM profile** rather than an
  agent profile. ``sdk.agent.complete(profile=...)`` resolves a model by name
  because a box cannot hold one; the extra indirection through
  ``agent_profiles[name]["llm"]`` bought nothing here, since this task has no
  agent, no tools and no scope — only a model. Empty or "default" follows the
  default profile.
- The high-water mark is still ``conversations.last_title_check_message_count``,
  written with ``sdk.db.write``. Kernel table *rows* are writable and schemas
  are not; this column is bookkeeping with no ``conv.*`` verb behind it, which
  is precisely the case that line exists for.
"""

dependencies_files = ['llm/llm_litellm.py']
dependencies_pip = []

import json
import re
import time

from guest.bases import BaseTask

UPDATE_TITLES = "update_titles"
"""Plugin-owned event channel for periodic conversation retitling."""

CONVERSATION_CHANGED = "conversation_changed"
"""The kernel's channel for conversation list changes.

Spelled out rather than imported: ``events.event_channels`` is a kernel module
and a plugin importing one cannot load in a subprocess. Channel names are
deliberately not validated against that file — plugins own their own channels —
so this string has to match by agreement, and it is the same one the kernel
emits created/deleted/recategorized on.
"""

_MAX_LEN = 80
# A conversation with this many user/agent messages is titled without
# waiting out the delay — it already carries enough context.
_EARLY_MESSAGES = 6

# Titles that still look kernel-generated and therefore may be replaced.
# Covers "New Conversation", "New conversation (Main)" and friends.
_DEFAULT_TITLE = re.compile(r"^new conversation\b", re.IGNORECASE)


def _needs_title(title) -> bool:
    """True while the conversation has never been given a real title."""
    text = str(title or "").strip()
    return not text or bool(_DEFAULT_TITLE.match(text)) or text.endswith("(cleared)")

_SYSTEM_PROMPT = (
    "You label conversations with short, concrete titles. "
    "You output only the title — never a sentence, greeting, or explanation."
)

_USER_TEMPLATE = (
    "<conversation>\n"
    "{transcript}\n"
    "</conversation>\n\n"
    "Write a 2-6 word title summarizing what the conversation is about.\n"
    "Rules:\n"
    "- Output only the title, no preamble, no quotes, no markdown\n"
    "- Be concrete and specific, not generic\n"
    "- Use title case\n\n"
    "Examples:\n"
    "Conversation about Rolls-Royce Cullinan pricing -> Cullinan Price\n"
    "Conversation planning a Virginia holiday -> Virginia Holiday Getaway\n"
    "Conversation debugging a SQLite migration -> SQLite Migration Bug\n\n"
    "Title:"
)


class UpdateTitles(BaseTask):
    """Update titles."""

    name = "update_titles"
    description = "Give new conversations a real title once they are ripe."
    trigger = "event"
    # Spelled out, not ``[UPDATE_TITLES]``: declarations are read by AST, so a
    # name reads as nothing and the task would register subscribed to no
    # channel at all — loading cleanly and never firing.
    trigger_channels = ["update_titles"]
    writes = []
    timeout = 600
    event_payload_schema = {"type": "object", "properties": {}, "required": []}
    #: The schedule this task wants to exist, created by ``on_install`` and by
    #: nothing else. Every minute is affordable because the sweep is one cheap
    #: SELECT that exits immediately when nothing is ripe.
    job = {"channel": "update_titles", "cron": "* * * * *", "payload": {}}

    requests = ["db.query", "db.write", "conv.read", "conv.set_title",
                "agent.complete", "event.emit", "config.read", "service.call"]

    config_settings = [
        ("Title Update LLM Profile", "title_update_llm_profile",
         "LLM profile used to generate conversation titles. 'default' follows "
         "the default profile. A small, cheap model is the right choice — the "
         "job is six words.",
         "default", {"type": "text"}),

        ("Title Delay (minutes)", "title_delay_minutes",
         "How long after the agent's first reply a conversation waits before "
         "being titled — the wait accumulates context so similar openers get "
         "distinct titles. Busy conversations are titled as soon as they "
         "reach 6 user/agent messages. 0 titles right after the first reply.",
         10, {"type": "slider", "range": (0, 60, 60), "is_float": False}),
    ]

    # New-message gate (vs the high-water mark) lives in SQL so the
    # every-minute sweep is one indexed SELECT that usually returns nothing.
    #
    # ``my_conversations`` rather than ``conversations``: the kernel expands it
    # to a subquery filtered to this execution's user, and refuses the bare
    # name. ``conversation_messages`` carries no owner column, so it is read
    # directly — the join through ``c.id`` is what scopes it.
    _CANDIDATES_SQL = """
        SELECT c.id    AS id,
               c.title AS title,
               (SELECT COUNT(*) FROM conversation_messages m
                  WHERE m.conversation_id = c.id) AS total_count,
               (SELECT COUNT(*) FROM conversation_messages m
                  WHERE m.conversation_id = c.id
                    AND m.role IN ('user', 'assistant')) AS content_count,
               (SELECT MIN(m.timestamp) FROM conversation_messages m
                  WHERE m.conversation_id = c.id
                    AND m.role = 'assistant') AS first_agent_ts
        FROM my_conversations c
        WHERE (SELECT COUNT(*) FROM conversation_messages m
                 WHERE m.conversation_id = c.id)
              > COALESCE(c.last_title_check_message_count, 0)
        ORDER BY c.updated_at DESC
    """

    _MARK_SQL = """
        UPDATE conversations
           SET last_title_check_message_count = ?
         WHERE id = ?
    """

    def on_install(self, sdk):
        """Create this task's schedule, once, when the package is installed.

        This was a ``default_jobs`` declaration, and the orchestrator seeded it
        at **every registration** — boot, install, hot-reload — skipping only a
        job that existed at that moment. A job the user had deleted did not
        exist, which is indistinguishable from one that was never installed, so
        it came back at the next restart and wrote config to announce itself.
        There was no way to say no; the base class even claimed the timekeeper
        tombstoned removals, which it has never done.

        ``on_install`` runs when somebody installs or updates this package and
        at no other time, so a deletion lasts until the user asks for this
        package again — which is the one moment re-creating it is what they
        meant.

        Read-then-skip, so an existing job keeps whatever cron it has since
        been given. Raising is reported by ``/packages`` and does not undo the
        install, which is the right way round: the task is installed and can be
        scheduled by hand, and a silent failure here would leave a task that
        simply never runs.
        """
        try:
            if sdk.services.call("timekeeper", "get_job", self.name) is None:
                sdk.services.call("timekeeper", "create_job",
                                  self.name, self.job)
                sdk.log(f"scheduled job {self.name} created")
        except sdk.Failed as error:
            raise RuntimeError(
                f"schedule {self.name!r} was not created ({error}) — this task "
                f"will not run until one is added in /schedule") from error

    def on_uninstall(self, sdk):
        """Take the schedule with it.

        A job whose task is gone fires into nothing forever, and is the kind of
        leftover only somebody reading ``/schedule`` would ever find.
        """
        try:
            sdk.services.call("timekeeper", "remove_job", self.name)
        except sdk.Failed as error:
            sdk.log(f"could not remove the {self.name} schedule: {error}",
                    level="warning")

    def _candidates(self, sdk) -> list:
        """Conversations with messages the sweep hasn't seen yet, as dicts.

        ``sdk.db.query`` answers with a list of dicts already — the native
        ``db.query`` returned ``{columns, rows}`` and needed zipping, and the
        Request does that work host-side because dicts are what crosses.
        """
        return list(sdk.db.query(self._CANDIDATES_SQL, max_rows=100) or [])

    def _mark(self, sdk, conversation_id, message_count: int) -> None:
        """Advance the high-water mark, best-effort.

        Swallowed on failure because the mark is an optimization, not the
        product: a sweep that cannot record where it got to repeats work next
        minute, which is cheap. One that dies here retitles nothing at all.
        """
        try:
            sdk.db.write(self._MARK_SQL, [int(message_count), conversation_id])
        except sdk.Failed:
            sdk.log(f"could not advance the title mark for {conversation_id}",
                    level="warning")

    def run_event(self, sdk, payload):
        """Sweep once: find ripe conversations and title them."""
        profile = str(
            sdk.config.read("title_update_llm_profile") or "default").strip()
        # "default" is the setting's way of saying "don't choose"; the Request
        # spells that as an absent profile.
        if profile == "default":
            profile = ""

        try:
            candidates = self._candidates(sdk)
        except sdk.Failed as exc:
            return sdk.fail(f"Failed to list conversations for title check: {exc}")

        if not candidates:
            return sdk.ok({"processed": 0, "candidates": 0})

        try:
            delay_minutes = float(sdk.config.read("title_delay_minutes") or 10)
        except (TypeError, ValueError):
            delay_minutes = 10.0
        now = time.time()

        updated = 0
        for row in candidates:
            conversation_id = row.get("id")
            message_count = int(row.get("total_count") or 0)
            if not _needs_title(row.get("title")):
                # Titled once (by us or by the user) — never overwrite.
                # Advance the mark so the row leaves the candidate list.
                self._mark(sdk, conversation_id, message_count)
                continue
            first_agent_ts = row.get("first_agent_ts")
            if not first_agent_ts:
                continue  # agent hasn't replied yet; revisit next tick
            ripe = (now - float(first_agent_ts)) >= delay_minutes * 60 \
                or int(row.get("content_count") or 0) >= _EARLY_MESSAGES
            if not ripe:
                continue  # don't advance the mark — it must come back
            try:
                if self._retitle(sdk, profile, conversation_id, message_count):
                    updated += 1
            except sdk.Failed as exc:
                sdk.log(f"title update failed for conversation "
                        f"{conversation_id}: {exc}", level="warning")
                # Still advance the high-water mark so a permanently bad
                # conversation doesn't block the sweep next tick.
                self._mark(sdk, conversation_id, message_count)

        sdk.log(f"title update sweep: processed {updated}/{len(candidates)} "
                f"conversations.")
        return sdk.ok({"processed": updated, "candidates": len(candidates)})

    def _retitle(self, sdk, profile, conversation_id, message_count: int) -> bool:
        """Title one conversation. Returns whether a title was written."""
        wrote = False
        try:
            # ``since_id=0`` is "walk forwards from the beginning", so this
            # asks for the twelve rows ``_transcript`` was already slicing to.
            # It has to be explicit: ``conv.read`` answers with the *newest*
            # page by default, and taking ``[:12]`` of that would title every
            # conversation from wherever it happened to have got to.
            messages = (sdk.conv.read(conversation_id, since_id=0, limit=12)
                        or {}).get("messages") or []
            transcript = _transcript(messages)
            if not transcript:
                return False
            response = sdk.agent.complete(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",
                     "content": _USER_TEMPLATE.format(transcript=transcript)},
                ],
                profile=profile,
            ) or {}
            title = _sanitize(response.get("content") or "")
            if title:
                sdk.conv.set_title(conversation_id, title)
                # Frontends (sidebars, pinned banners) refresh off this;
                # the kernel only emits created/deleted/recategorized.
                sdk.events.emit(CONVERSATION_CHANGED,
                                {"action": "retitled",
                                 "conversation_id": conversation_id})
                sdk.log(f"updated conversation {conversation_id} title to "
                        f"'{title}'.")
                wrote = True
        finally:
            # Always advance the high-water mark — even if we skipped or failed
            # — so an empty / un-titleable conversation does not replay.
            self._mark(sdk, conversation_id, message_count)
        return wrote


# ======================================================================
# Pure helpers
# ======================================================================

# The kernel's ``runtime.token_stripper`` does this and more, and a plugin
# cannot import it — ``runtime`` is on the kernel side of the boundary. Copied
# rather than turned into a Request: it is four patterns and a function, it is
# pure, and growing the Request vocabulary for a string transform would be the
# wrong direction. Only the batch half is here; the streaming filter has no
# caller in a task that sees whole responses.
_THINKING_PATTERN = re.compile(
    r"<(?:think|thinking)>(.*?)</(?:think|thinking)>", re.DOTALL)
_STRUCTURAL_PATTERN = re.compile(
    r"<invoke.*?>.*?</invoke>|<tool_call.*?>.*?</tool_call>|"
    r"<(?:/)?minimax:tool_call>|<\|im_end\|>|<\|eot_id\|>", re.DOTALL)
_THINKING_TAG_PATTERN = re.compile(r"</?(?:think|thinking)>")


def _strip_model_tokens(text: str) -> str:
    """Reasoning blocks, tool-call XML and leaked EOS tokens removed.

    The opening tag is *required* on a thinking block. Making it optional means
    the non-greedy body matches "anything up to the next closer", so two blocks
    around a title eat the title — a bug the kernel version documents at
    length. A six-word answer has no room to lose any of itself.
    """
    clean = _THINKING_PATTERN.sub("", text or "")
    clean = _STRUCTURAL_PATTERN.sub("", clean)
    return _THINKING_TAG_PATTERN.sub("", clean).strip()


def _transcript(messages: list) -> str:
    """The first few turns, flattened to ``ROLE: text`` lines."""
    lines = []
    for msg in messages[:12]:
        role = (msg.get("role") or "").upper()
        if role == "TOOL":
            continue
        content = msg.get("content") or ""
        if role == "ASSISTANT":
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    content = parsed.get("content") or ""
            except (TypeError, ValueError):
                pass
        content = " ".join(content.split()).strip()
        if not content:
            continue
        if len(content) > 300:
            content = content[:300].rstrip() + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _sanitize(text: str) -> str:
    """One clean title line, or "" if the model produced nothing usable."""
    title = _strip_model_tokens(text)
    if not title:
        return ""
    title = title.splitlines()[0].strip()
    title = title.strip().strip("\"'`*#-: ")
    title = " ".join(title.split())
    title = title[:_MAX_LEN].strip()
    generic = {"new conversation", "conversation", "chat", "untitled", "title"}
    if not title or title.casefold() in generic:
        return ""
    return title
