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

**The corpus is reinforcement learning without weights.** A note is a
situation, an action, and the result that followed — so retrieval is what makes
a past result bear on a present decision. Actions that produced good results
should be repeated, actions that produced bad ones avoided, and a neutral
result changes no action and is therefore not worth a file. That is the whole
model, and everything below follows from it: a note with no action in it cannot
change anything, so it is not a note.

**The curator has two jobs, and the transcript says which one applies.** If the
agent reached for memory and solved the problem with it, the note that helped
earns a rewrite — it worked, so make it work harder: sharpen the situation that
should retrieve it, tighten the action, record what the result actually was. If
the agent solved the problem some novel way without consulting memory, that
solution is not yet in the corpus and wants writing down. Both jobs answer the
same question, which is whether the next agent in this situation will do better
than this one did.

Rewriting in place is the mechanism rather than a hazard here, because it is
*targeted*: only notes the conversation actually exercised, edited against
evidence of how they performed. That is not the wholesale re-summarisation that
erodes a corpus.

**Facts do not belong here.** Anything simply true — a name, which machine
something runs on, a stated preference with no action attached — belongs in
``MEMORY.md``, which the agent maintains itself and this task never touches.
This folder is for actions and sequences of actions only.
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

#: Characters per tool result. Tighter than a message on purpose: a file read
#: or a search result is the longest thing in a transcript and the least
#: informative per character, but *which* tool ran is load-bearing evidence.
MAX_TOOL_CHARS = 300

#: Total transcript budget.
MAX_TRANSCRIPT_CHARS = 24_000

_PROMPT = """You curate the memory folder at:
{root}

It holds **actions**: what to do, or not do, in a situation that has come up
before. Each note is one situation, one action, and the result that followed.
Nothing else goes here — facts, names and preferences with no action attached
belong in MEMORY.md, which you must not touch.

Below is conversation {cid} ("{title}"), including the tool calls. Read it and
work out what the next agent in this situation should do differently.

## Which job is yours

**Did the agent read files from the memory folder and use them?**

- **Yes** — those notes earned their place. Improve them. Open each one that
  was used and rewrite it against what actually happened: sharpen `when` so it
  fires on this situation too, tighten `do`/`avoid` if the advice was vague or
  partly wrong, and record the real result in `because`. Editing in place is
  correct here — you have evidence, which is the one thing that makes a rewrite
  an improvement rather than a guess.

- **No** — the agent solved this without memory, so whatever worked is not in
  the corpus yet. Write it down.

Both can apply. Search the folder before writing anything new: if a note
already covers the situation, improve that note instead of adding a second one.

## What earns a note

One test: **will this change what an agent does?** If you cannot name the
action, there is nothing to write.

- An action that produced a good result -> a note saying to do it.
- An action that produced a bad result, a trap, a correction the user made,
  something that broke -> a note saying to avoid it, and what to do instead.
  These are worth more than successes; they are what nobody writes down.
- A neutral result changes no action. Write nothing.

**Most conversations deserve nothing, and writing nothing is a correct
outcome.** Never record what happened for its own sake, anything true only
inside this conversation, or anything obvious from reading the code.

## The format

One file per situation, named for what it is about (`retry_failed_uploads.md`).
Keep it short — a few lines of body is normal.

---
when: the situation that should bring this back
do: the action to take
because: what happened when it was done
updated: {today}
source: conversation {cid}
---

Use `avoid:` in place of `do:` when the lesson is not to do something; say what
to do instead in the same field.

`when` is the field that decides whether this note is ever seen again. It is
matched against what a user says in some future conversation, so write it as
the *situation* — "a PDF yields no text", "about to commit to a repo Henry
owns" — never as a topic label.

Transcript:

{transcript}

Reply with one line: what you wrote or improved, or "nothing worth keeping"."""


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
        """The new messages, oldest first, capped — tool calls included.

        The tool rows are not optional detail here, they are the evidence the
        curator branches on: whether the agent reached for memory and used what
        it found is visible only in what it called. A user/assistant-only
        transcript also hides where the work actually happened, which is most
        of what a conversation with real work in it consists of.
        """
        sql = """
            SELECT role, content, tool_name
              FROM conversation_messages
             WHERE conversation_id = ?
               AND id <= ?
               AND LOWER(COALESCE(role, '')) <> 'system'
               AND COALESCE(content, '') <> ''
             ORDER BY id DESC
             LIMIT ?
        """
        # Tool rows inflate the count well past the message floor, so the
        # window is the cap rather than what the floor happened to count.
        limit = min(max(new_count * 4, MAX_MESSAGES // 4), MAX_MESSAGES)
        try:
            rows = sdk.db.query(sql, [cid, max_id, limit], max_rows=MAX_MESSAGES)
        except sdk.Failed as error:
            sdk.log(f"could not read conversation {cid}: {error}", level="warning")
            return ""
        parts = []
        for row in reversed(rows or []):
            if line := self._line(row):
                parts.append(line)
        return "\n\n".join(parts)[:MAX_TRANSCRIPT_CHARS]

    def _line(self, row):
        """Render one message, naming the tool when there is one."""
        role = str(row.get("role") or "?").upper()
        body = " ".join(str(row.get("content") or "").split())
        if not body:
            return ""
        if tool := str(row.get("tool_name") or "").strip():
            # A tool result is the longest thing in most transcripts and the
            # least informative per character: which tool ran and roughly what
            # came back is the whole signal.
            return f"{role} ({tool}): {body[:MAX_TOOL_CHARS]}"
        return f"{role}: {body[:MAX_MESSAGE_CHARS]}"
