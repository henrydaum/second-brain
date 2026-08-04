"""Curate memory notes when a conversation goes quiet.

A conversation ending is an episodic boundary: it is one self-contained piece
of work, which makes it a far better unit to reflect on than a calendar day.
So when a conversation stops being the active one, a subagent reads what
happened and decides what — if anything — is worth keeping.

**The trigger is ``session_conversation_ended``, and the watermark is what
makes it safe to trust.** The event names the conversation being left — on a
switch, on a session closing, on a delete — which is exactly the unit of work
worth reflecting on. What it cannot promise is delivery: a crash emits nothing.
So every conversation also carries the id of the last message already reflected
on, and the hourly sweep asks the same question with no event at all. Between
them: the event makes reflection prompt, the watermark makes it idempotent
under someone flicking through their history, and the sweep makes a lost event
recoverable rather than silently dropped. The watermark is also what gives
"reflect only on what is new" for free when an old conversation is reopened and
left again.

**The subagent is a curator, not a recorder.** It may write several notes, or
none, and "none" is the common case. Where something similar already exists it
writes a merged note that declares what it supersedes — it never rewrites an
existing note in place, because iterative rewriting is precisely what erodes a
corpus over time: each pass quietly drops the details the last one thought
were minor.
"""

import time

from guest.bases import BaseTask

#: Conversations reflected on in one run. A backlog is drained over several
#: runs rather than risking the task's own deadline: each subagent is a real
#: agent turn, and the sweep comes round again in an hour.
MAX_PER_RUN = 3

#: Messages pulled into the curator's prompt, newest-last.
MAX_MESSAGES = 400

#: Characters per message. Enough to see what happened, not enough for one
#: pasted file to crowd out the rest of the conversation.
MAX_MESSAGE_CHARS = 800

#: Total transcript budget.
MAX_TRANSCRIPT_CHARS = 24_000

_PROMPT = """You are curating the memory folder at:
{root}

Below are the new messages from conversation {cid} ("{title}"). They are the
part nobody has reflected on yet.

Decide what, if anything, is worth remembering. Good candidates: a durable
fact about the user or their systems, a preference they stated, a reusable
procedure that worked, a lesson from something that went wrong, a decision and
its reasoning.

**Most conversations deserve nothing. Writing no notes is a correct and common
outcome.** Do not record what happened for its own sake, anything true only
inside this conversation, or anything already obvious from the code or the
repository.

Before writing, search the memory folder for what is already there. If a note
covers this ground, write a merged note and list the slugs it replaces in
`supersedes:`. Never edit or delete an existing note in place — superseding
leaves a trail, rewriting destroys one.

Write each note as its own file under the folder above, named by function
(`skill_*.md`, `fact_*.md`, `pref_*.md`, or whatever fits). One idea per file;
several files is fine. Start every file with frontmatter:

---
name: kebab-case-slug
type: skill | fact | preference | summary
description: one line — what this note holds
when: the situation that should bring this note back
keywords: [a, few, search, terms]
created: {today}
updated: {today}
source: conversation {cid}
supersedes: []
---

`when` is the most important field. It is matched against what the user says
in some future conversation, so write it as the *situation* — "a PDF yields no
text", "the user asks about deploying" — not as a topic label.

Transcript:

{transcript}

When you are done, reply with one line saying what you wrote, or "nothing
worth keeping" if that is the answer."""


class MemoryReflect(BaseTask):
    """Reflect on conversations that have gone quiet."""

    name = "memory_reflect"
    description = "Curate memory notes from conversations that have gone quiet."

    trigger = "event"
    trigger_channels = ["session_conversation_ended"]
    reads = []
    writes = ["memory_reflections"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS memory_reflections (
            conversation_id INTEGER PRIMARY KEY,
            last_message_id INTEGER NOT NULL,
            reflected_at REAL
        )
    """
    timeout = 600

    default_jobs = {
        "memory_reflect_sweep": {
            "channel": "session_conversation_ended",
            "cron": "0 * * * *",
            "payload": {},
        },
    }

    requests = ["db.query", "agent.spawn", "paths.get", "config.read",
                "conv.read", "session.list", "session.get"]
    dependencies_files = []
    dependencies_pip = []

    config_settings = [
        ("Memory reflection floor", "memory_reflect_min_messages",
         "How many new messages a conversation needs before it is reflected on. "
         "Keeps browsing your history from spawning subagents.",
         4, {"type": "slider", "range": (2, 40, 38), "is_float": False}),
        ("Memory reflection window", "memory_reflect_max_age_hours",
         "How recently a conversation must have been active to be reflected on. "
         "Stops a fresh install from reflecting on your entire history.",
         24, {"type": "slider", "range": (1, 168, 167), "is_float": False}),
    ]

    agent_prompt = (
        "When a conversation goes quiet, a background agent reviews it and "
        "writes anything durable into your memory folder. You do not need to "
        "record things as you go for that reason alone."
    )

    def run_event(self, sdk, payload):
        """Reflect on whatever has gone quiet, oldest first.

        Two shapes of payload arrive here and both are handled by the same
        query. ``session_conversation_ended`` names one conversation, so the
        run narrows to it; the hourly sweep names nothing, so the run takes
        whatever the watermark has been left holding.
        """
        if self._is_own_child(payload):
            return sdk.ok([])

        floor = self._floor(sdk)
        busy = self._busy_conversations(sdk)
        candidates = [row for row in self._candidates(sdk, floor, self._cutoff(sdk))
                      if row["cid"] not in busy]
        if ended := (payload or {}).get("conversation_id"):
            candidates = [row for row in candidates
                          if row["cid"] == int(ended)]
        if not candidates:
            return sdk.ok([])

        root = sdk.path.join(sdk.paths.get("workspace"), "memory")
        today = time.strftime("%Y-%m-%d")
        done = []
        for row in candidates[:MAX_PER_RUN]:
            if self._reflect(sdk, row, root, today):
                done.append({
                    "conversation_id": row["cid"],
                    "last_message_id": row["max_id"],
                    "reflected_at": time.time(),
                })
        return sdk.ok(done, llm_summary=f"Reflected on {len(done)} conversation(s).")

    # ── deciding what to reflect on ──────────────────────────────────

    def _is_own_child(self, payload):
        """Whether this event is the curator we just spawned, finishing.

        A subagent gets its own conversation and closes its session when it is
        done, so every curator completion emits the very channel that started
        it. Left unfiltered that is a feedback loop: the curator's own
        transcript looks like a conversation that has gone quiet, so we reflect
        on it, which spawns another curator, which ends, which... It terminates
        only by luck — when a curator's transcript happens to fall under the
        message floor — which is how one conversation ending produced four
        runs.

        The session key is the tell and it costs nothing to read. The category
        filter in the candidate query is the other half, for conversations the
        sweep meets later with no event to inspect.
        """
        return str((payload or {}).get("session_key") or "").startswith(
            "spawn_subagent:")

    def _floor(self, sdk):
        """How many new messages earn a reflection."""
        try:
            return max(1, int(sdk.config.read("memory_reflect_min_messages") or 4))
        except (sdk.Failed, TypeError, ValueError):
            return 4

    def _cutoff(self, sdk):
        """How far back a conversation may have been active and still count.

        Without this, installing the bundle reflects on *everything you have
        ever said*: the watermark defaults to zero for a conversation nobody
        has looked at, so every message in the archive reads as new and the
        whole history queues up three at a time. That is expensive, spawns an
        agent per conversation, and produces notes about work from months ago
        as though it had just happened.

        A window is the right shape rather than a one-off backfill guard,
        because it keeps being true: a conversation abandoned last spring
        should not become a candidate the day somebody opens it to read.
        Reopening one and adding a message makes it recent again, which is
        exactly when it *is* worth reflecting on.
        """
        try:
            hours = max(1, int(sdk.config.read("memory_reflect_max_age_hours") or 24))
        except (sdk.Failed, TypeError, ValueError):
            hours = 24
        return time.time() - (hours * 3600)

    def _busy_conversations(self, sdk):
        """Conversations somebody is currently sitting in.

        The ended event is emitted as the session lets go, so the conversation
        it names is already gone from every live session and this never
        excludes the thing we were called about. What it does catch is the
        sweep finding a conversation somebody is mid-sentence in — reflecting
        there would spawn an agent to summarise a turn still being typed.
        """
        busy = set()
        try:
            keys = sdk.session.list() or []
        except sdk.Failed:
            return busy
        for key in keys:
            try:
                info = sdk.session.get(str(key)) or {}
            except sdk.Failed:
                continue
            if cid := info.get("conversation_id"):
                busy.add(int(cid))
        return busy

    def _candidates(self, sdk, floor, cutoff):
        """Recently-active conversations with unreflected messages, oldest first.

        Subagent conversations are excluded here rather than trusted to the
        session-key check: that one only sees an event, and the sweep meets a
        finished curator's conversation with no event to inspect. A child is
        created with ``kind = 'user'``, so the category is what separates it.

        ``conversation_messages`` is not user-scoped and is read directly.
        ``conversations`` cannot be, so this joins ``my_conversations``, which
        resolves to user 1 inside a task — reflection is therefore scoped to
        the base user, which is right for a single-user install and something
        to revisit before anyone else's conversations need reflecting on.
        """
        sql = """
            SELECT m.conversation_id AS cid,
                   MAX(m.id)         AS max_id,
                   COUNT(*)          AS new_count
              FROM conversation_messages m
              JOIN my_conversations c
                     ON c.id = m.conversation_id
              LEFT JOIN memory_reflections r
                     ON r.conversation_id = m.conversation_id
             WHERE m.id > COALESCE(r.last_message_id, 0)
               AND LOWER(m.role) IN ('user', 'assistant')
               AND COALESCE(m.content, '') <> ''
               AND COALESCE(c.category, '') <> 'Subagent'
             GROUP BY m.conversation_id
            HAVING new_count >= ?
               AND MAX(COALESCE(m.timestamp, 0)) >= ?
             ORDER BY max_id ASC
        """
        try:
            return sdk.db.query(sql, [floor, cutoff], max_rows=100) or []
        except sdk.Failed as error:
            sdk.log(f"could not look for conversations to reflect on: {error}",
                    level="warning")
            return []

    # ── the curator ──────────────────────────────────────────────────

    def _reflect(self, sdk, row, root, today):
        """Run one curator subagent. True when the watermark may advance."""
        cid = int(row["cid"])
        transcript = self._transcript(sdk, cid, int(row["max_id"]),
                                      int(row["new_count"]))
        if not transcript:
            # Nothing readable to reflect on, but the messages have been
            # considered — advancing stops us reconsidering them hourly.
            return True
        prompt = _PROMPT.format(root=root, cid=cid, title=self._title(sdk, cid),
                                today=today, transcript=transcript)
        try:
            report = sdk.agent.spawn(prompt, title=f"Memory: conversation {cid}",
                                     wait=True)
        except sdk.Failed as error:
            sdk.log(f"memory curator failed for conversation {cid}: {error}",
                    level="warning")
            return False
        if not (report or {}).get("ok"):
            sdk.log(f"memory curator did not finish for conversation {cid}: "
                    f"{(report or {}).get('error') or 'no report'}",
                    level="warning")
            return False
        return True

    def _title(self, sdk, cid):
        """The conversation's title, for the curator's orientation only."""
        try:
            record = sdk.conv.read(cid) or {}
        except sdk.Failed:
            return "untitled"
        return str((record.get("conversation") or {}).get("title") or "untitled")

    def _transcript(self, sdk, cid, max_id, new_count):
        """The new messages, oldest first, capped."""
        sql = """
            SELECT role, content
              FROM conversation_messages
             WHERE conversation_id = ?
               AND id <= ?
               AND LOWER(role) IN ('user', 'assistant')
               AND COALESCE(content, '') <> ''
             ORDER BY id DESC
             LIMIT ?
        """
        limit = min(max(new_count, 1), MAX_MESSAGES)
        try:
            rows = sdk.db.query(sql, [cid, max_id, limit], max_rows=MAX_MESSAGES)
        except sdk.Failed as error:
            sdk.log(f"could not read conversation {cid}: {error}", level="warning")
            return ""
        parts = []
        for row in reversed(rows or []):
            role = str(row.get("role") or "?").upper()
            body = " ".join(str(row.get("content") or "").split())
            if body:
                parts.append(f"{role}: {body[:MAX_MESSAGE_CHARS]}")
        return "\n\n".join(parts)[:MAX_TRANSCRIPT_CHARS]
