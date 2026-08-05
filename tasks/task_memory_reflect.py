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

**The curator has two jobs, and a retrieved-then-read pair says which.** A note
the agent was shown and went on to open earns a rewrite — it worked, so make it
work harder: sharpen the situation that retrieves it, tighten the action,
record what the result actually was. Anything the agent worked out for itself
is not in the corpus yet and wants writing down. Both answer one question,
which is whether the next agent here does better than this one did.

Neither half of that pair is available alone. The offer lives in the system
prompt, which is stored nowhere, so ``service_memory`` records it; the open is
a ``read_file`` call, which is in the transcript because the agent had to name
the path to make it. This is also the reason the prompt carries situations and
paths rather than the notes themselves — inline the advice and there is no
reason to open anything, and the signal disappears.

Note what the pair is *not*: a score. That a note was opened says the situation
looked close, not that the advice was good. Only reading what happened next
settles that, which is why a model does it and not a counter.

Rewriting in place is the mechanism rather than a hazard here, because it is
*targeted*: only notes the conversation actually exercised, edited against
evidence of how they performed. That is not the wholesale re-summarisation that
erodes a corpus.

**Facts do not belong here.** Anything simply true — a name, which machine
something runs on, a stated preference with no action attached — belongs in
``MEMORY.md``, which the agent maintains itself and this task never touches.
This folder is for actions and sequences of actions only.
"""

import json
import time

from guest.bases import BaseTask

#: Conversations reflected on in one run. A backlog is drained over several
#: runs rather than risking the task's own deadline: each subagent is a real
#: agent turn, and the sweep comes round again in an hour.
MAX_PER_RUN = 3

#: What this task titles its own children. Filtering on it is exact rather than
#: fragile, because the task owns both ends of the string: it sets the title
#: when it spawns and matches the same constant when it queries. Needed once
#: subagent conversations become reflectable at all, since the curator's own
#: conversation is a subagent conversation like any other.
CURATOR_TITLE = "Memory curation:"

#: Notes live in this subfolder of the memory root, and only notes do. It is
#: what the retrieval service searches, so membership is a path rather than a
#: property of the file's contents — which is the one thing a writer cannot get
#: subtly wrong. Must match ``NOTES_DIRNAME`` in ``service_memory.py``; the two
#: are pinned equal by ``tests/test_store_memory_bundle.py``.
NOTES_DIRNAME = "actions"

#: Last-resort budget for ``MEMORY.md``, used only when the setting cannot be
#: read at all. The real number is the kernel's ``memory_index_cap``: past it
#: the index is truncated out of the prompt, so a curator told a different
#: figure would write facts nobody ever sees. A plugin cannot import
#: ``agent/system_prompt.py`` to ask, which is exactly why the budget is a
#: setting and not a constant on either side.
FALLBACK_INDEX_BUDGET = 4000

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

_PROMPT = """You curate the memory notes at:
{notes}

They are **actions**: what to do, or not do, in a situation that has come up
before. Each note is one situation, one action, and the result that followed.

**Every note goes in that folder and only that folder.** It is the only place
searched, so a note written anywhere else will never be found again. The rest
of `{root}` is for things that are not notes.

Below is conversation {cid} ("{title}"), including the tool calls. Read it and
work out what the next agent in this situation should do differently.

## Job one: improve the notes that were used

These notes were surfaced to the agent and it opened them:

{used}

Each one earned its place, so make it work harder. Open it and rewrite it
against what actually happened:

- sharpen `when` so it also fires on the situation that just came up
- tighten `do`/`avoid` if the advice was vague, incomplete, or partly wrong
- record the real result in `because`

Editing in place is correct here. You have evidence of how the note performed,
which is the one thing that makes a rewrite an improvement rather than a guess.
If a note was opened and turned out to be wrong or useless, say so in it — a
note that records its own failure is worth more than one quietly left standing.

## Job two: write down what is missing

Whatever the agent worked out for itself, without memory, is not in the corpus
yet. Go through the conversation for actions worth repeating or avoiding that
no note covers, and write those.

Search the folder before writing anything new. If a note already covers the
situation, improve that one rather than adding a second — two notes about one
situation is how a corpus stops being useful.

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

One file per situation, in `{notes}`, named for what it is about
(`retry_failed_uploads.md`). Keep it short — a few lines of body is normal.

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

{facts}
Transcript:

{transcript}

Reply with one line: what you wrote or improved, or "nothing worth keeping"."""

_FACTS_JOB = """## Job three: keep MEMORY.md current

`{root}/MEMORY.md` holds **facts** — things that are simply true and carry no
action. Names, paths, which machine something runs on, how the user likes to be
addressed, a preference with nothing to do about it. It is inlined into the
agent's prompt in full, every turn, so it is the one place a fact is guaranteed
to be seen without anybody going to look.

Add what this conversation established. A fact belongs there when it will still
be true next month and is not discoverable by reading the code.

**It has a budget of about {budget} characters.** Past that the kernel
truncates it and the tail is simply not in the prompt — so this job is as much
pruning as adding. Every time you touch it: delete what has become false,
merge duplicates, cut anything that turned out to be obvious or one-off, and
tighten wording. If it is near the budget, something has to go before anything
is added; choose what to lose deliberately rather than letting the truncation
choose for you.

Keep the division clean. Anything with an action in it is a note, not a fact,
and belongs in the folder — MEMORY.md is not a place for advice.

"""


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

    requests = ["db.query", "db.write", "agent.spawn", "paths.get",
                "config.read", "conv.read", "session.list", "session.get"]
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
        ("Curate facts into MEMORY.md", "memory_reflect_curate_facts",
         "Let the curator also add facts learned in a conversation to "
         "MEMORY.md, and prune it to stay inside its prompt budget. Turn off "
         "to keep that file yours alone.",
         True, {"type": "bool"}),
        ("Reflect on subagents", "memory_reflect_include_subagents",
         "Also curate memories from subagents you spawned. Their whole purpose "
         "is often to go and find something out, so the lesson is real — but "
         "nobody was steering, and every one costs another curator run. "
         "Scheduled subagents are never included: they reuse one conversation "
         "forever, so there is no ending to reflect on.",
         False, {"type": "bool"}),
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
        subagents = self._include_subagents(sdk)
        if not subagents and self._from_a_child(payload):
            return sdk.ok([])

        floor = self._floor(sdk)
        cutoff = self._cutoff(sdk)
        self._prune_retrievals(sdk, cutoff)
        busy = self._busy_conversations(sdk)
        candidates = [row for row in self._candidates(sdk, floor, cutoff, subagents)
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

    def _facts_job(self, sdk, root):
        """The third job, or nothing at all.

        Appended rather than woven in, so that turning it off leaves a prompt
        with no trace of a job the curator is not allowed to do — a rule
        stated and then contradicted is worse than one never stated.
        """
        try:
            wanted = sdk.config.read("memory_reflect_curate_facts")
        except sdk.Failed:
            wanted = True
        if wanted is None:
            wanted = True
        if not wanted:
            return ""
        return _FACTS_JOB.format(root=root, budget=self._index_budget(sdk))

    def _index_budget(self, sdk):
        """How much of MEMORY.md the kernel will actually inline."""
        try:
            return max(1, int(sdk.config.read("memory_index_cap")
                              or FALLBACK_INDEX_BUDGET))
        except (sdk.Failed, TypeError, ValueError):
            return FALLBACK_INDEX_BUDGET

    def _include_subagents(self, sdk):
        """Whether subagent conversations are worth curating here.

        Off by default and deliberately so. A subagent's whole purpose is often
        to go and find something out, which makes the lesson real — but nobody
        was steering it, so there is no user correcting a wrong turn, and every
        included conversation costs another curator run on top of the work it
        is reflecting on.
        """
        try:
            return bool(sdk.config.read("memory_reflect_include_subagents"))
        except sdk.Failed:
            return False

    def _from_a_child(self, payload):
        """Whether this event is a subagent's session closing.

        Every subagent gets its own conversation and closes its session when it
        finishes, so a curator completion emits the very channel that started
        it. Left unfiltered that is a feedback loop: the curator's transcript
        looks like a conversation that has gone quiet, so we reflect on it,
        spawning another curator, which ends, which... It terminated only by
        luck — when a transcript happened to fall under the message floor —
        which is how one conversation ending once produced four runs.

        Skipping every child is the blunt form of that guard and it is the
        right one while subagents are not being curated at all. Once they are,
        this cannot fire, and what keeps the loop closed instead is the title
        filter in the candidate query: exact, since the task sets that title
        itself, and it also covers the conversations the sweep meets later with
        no event to inspect.
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

    def _candidates(self, sdk, floor, cutoff, subagents):
        """Recently-active conversations with unreflected messages, oldest first.

        **At least one assistant message is required**, and the total floor
        does not cover it: the corpus records what the *agent* did, so a
        conversation the agent never spoke in has no action to learn from and
        nothing a note could say. Someone can reach the floor without that
        happening — a few messages typed at a turn that failed or was
        cancelled, or a conversation opened only to run slash commands — and
        the curator would then be handed a transcript with no agent in it and
        asked what should be done differently next time.

        **Three kinds of conversation are separated by category**, which the
        kernel already sets and which is the only one of these facts that
        survives into the database. An interactive ``sdk.agent.spawn`` files
        its child under ``Subagent``; a scheduled one under ``Scheduled`` or
        ``Scheduled (one-time)``. So the setting can include the first without
        the second — and the second must always be excluded, because a
        scheduled job pins its conversation and reuses it forever, which means
        it never really ends and reflecting on it would re-read the same
        growing transcript every hour.

        The curator's own children are excluded by title in both modes. With
        subagents off the session-key guard already covers the event path, but
        the sweep has no event to inspect; with subagents on the guard cannot
        fire at all, and this is the only thing standing between the curator
        and its own output.

        ``conversation_messages`` is not user-scoped and is read directly.
        ``conversations`` cannot be, so this joins ``my_conversations``, which
        resolves to user 1 inside a task — reflection is therefore scoped to
        the base user, which is right for a single-user install and something
        to revisit before anyone else's conversations need reflecting on.
        """
        # A literal fragment chosen between two constants — never anything the
        # guest or the database supplied, which is why it can be formatted in
        # while every value stays a bound parameter.
        exclude = "" if subagents else "\n               AND COALESCE(c.category, '') <> 'Subagent'"
        sql = f"""
            SELECT m.conversation_id AS cid,
                   MAX(m.id)         AS max_id,
                   COUNT(*)          AS new_count,
                   COALESCE(MAX(r.last_message_id), 0) AS previous_id
              FROM conversation_messages m
              JOIN my_conversations c
                     ON c.id = m.conversation_id
              LEFT JOIN memory_reflections r
                     ON r.conversation_id = m.conversation_id
             WHERE m.id > COALESCE(r.last_message_id, 0)
               AND LOWER(m.role) IN ('user', 'assistant')
               AND COALESCE(m.content, '') <> ''
               AND COALESCE(c.category, '') NOT LIKE 'Scheduled%'
               AND COALESCE(c.title, '') NOT LIKE ?{exclude}
             GROUP BY m.conversation_id
            HAVING new_count >= ?
               AND MAX(COALESCE(m.timestamp, 0)) >= ?
               AND SUM(CASE WHEN LOWER(m.role) = 'assistant'
                            THEN 1 ELSE 0 END) >= 1
             ORDER BY max_id ASC
        """
        try:
            return sdk.db.query(sql, [f"{CURATOR_TITLE}%", floor, cutoff],
                                max_rows=100) or []
        except sdk.Failed as error:
            sdk.log(f"could not look for conversations to reflect on: {error}",
                    level="warning")
            return []

    # ── the curator ──────────────────────────────────────────────────

    def _reflect(self, sdk, row, root, today):
        """Run one curator subagent. True when the watermark may advance."""
        cid = int(row["cid"])
        previous_id = int(row.get("previous_id") or 0)
        transcript = self._transcript(
            sdk, cid, previous_id, int(row["max_id"]), int(row["new_count"]))
        if not transcript:
            # Nothing readable to reflect on, but the messages have been
            # considered — advancing stops us reconsidering them hourly.
            return True
        prompt = _PROMPT.format(root=root,
                                notes=sdk.path.join(root, NOTES_DIRNAME),
                                cid=cid, title=self._title(sdk, cid),
                                today=today, transcript=transcript,
                                used=self._used_notes(
                                    sdk, cid, previous_id, int(row["max_id"])),
                                facts=self._facts_job(sdk, root))
        try:
            report = sdk.agent.spawn(
                prompt, title=f"{CURATOR_TITLE} conversation {cid}", wait=True)
        except sdk.Failed as error:
            sdk.log(f"memory curator failed for conversation {cid}: {error}",
                    level="warning")
            return False
        if not (report or {}).get("ok"):
            sdk.log(f"memory curator did not finish for conversation {cid}: "
                    f"{(report or {}).get('error') or 'no report'}",
                    level="warning")
            return False
        self._forget_retrievals(sdk, cid)
        return True

    def _used_notes(self, sdk, cid, previous_id, max_id):
        """Notes that were surfaced to the agent and that it then opened.

        Both halves are needed and neither is available alone. The offer lives
        only in the prompt, which is not stored anywhere, so the service
        records it in ``memory_retrievals_v2``. The open is a ``read_file`` call,
        which appears in the transcript because the agent had to name the path
        to make it. Nothing else puts that string there: the retrieval block
        itself never enters the conversation.

        The stored assistant row is JSON whose ``arguments`` value is itself a
        JSON string. Both layers are decoded, and only a call named
        ``read_file`` whose normalized path exactly equals the offered path is
        accepted. A filename in prose or in another tool call is not a read.

        Matched against the *messages*, not against the transcript built for
        the prompt. That transcript is capped at ``MAX_TRANSCRIPT_CHARS`` and
        trimmed per message, so a read early in a long conversation falls off
        the front of it — and the curator would conclude the note was never
        used and write a duplicate instead of improving the one that helped.
        The evidence has to be searched at full length even though only a
        window of it is shown. The read must also follow the particular user
        message that caused the offer; finding both events somewhere in the
        same conversation is not enough.

        This decides which job the curator has, and nothing more. It is not a
        score — that a note was opened says the situation looked close, not
        that the advice was any good, and only reading what happened next can
        settle that.
        """
        try:
            rows = sdk.db.query(
                "SELECT path, offered_message_id FROM memory_retrievals_v2"
                " WHERE conversation_id = ?"
                "   AND offered_message_id > ? AND offered_message_id <= ?",
                [int(cid), int(previous_id), int(max_id)], max_rows=500) or []
        except sdk.Failed:
            return ""
        if not rows:
            return ("None. Memory was either not surfaced or not opened, so "
                    "nothing here has been shown to help yet.")
        calls = self._read_file_calls(
            sdk, cid,
            min(int(row.get("offered_message_id") or 0) for row in rows),
            max_id,
        )
        used = []
        project = sdk.paths.get("project")
        for row in rows:
            path = str(row.get("path") or "")
            offered_id = int(row.get("offered_message_id") or 0)
            wanted = (sdk.path.normalize(
                sdk.path.absolute(path, base=project)) if path else "")
            if wanted and any(message_id > offered_id and opened == wanted
                              for message_id, opened in calls):
                used.append(path)
        if not used:
            return ("None. Memory was either not surfaced or not opened, so "
                    "nothing here has been shown to help yet.")
        return "\n".join(f"- {path}" for path in dict.fromkeys(used))

    def _read_file_calls(self, sdk, cid, after_id, max_id):
        """Exact ``(message id, normalized path)`` read-file calls."""
        try:
            rows = sdk.db.query(
                "SELECT id, content FROM conversation_messages"
                " WHERE conversation_id = ? AND id > ? AND id <= ?"
                "   AND LOWER(role) = 'assistant'"
                "   AND COALESCE(content, '') <> ''",
                [int(cid), int(after_id), int(max_id)], max_rows=500) or []
        except sdk.Failed:
            return []
        found = []
        project = sdk.paths.get("project")
        for row in rows:
            try:
                packed = json.loads(str(row.get("content") or ""))
            except (TypeError, ValueError):
                continue
            calls = packed.get("tool_calls") if isinstance(packed, dict) else None
            for call in calls if isinstance(calls, list) else []:
                function = call.get("function") if isinstance(call, dict) else None
                function = function if isinstance(function, dict) else {}
                name = call.get("name") or function.get("name")
                if str(name or "").strip() != "read_file":
                    continue
                arguments = call.get("arguments")
                if arguments is None:
                    arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (TypeError, ValueError):
                        continue
                if not isinstance(arguments, dict):
                    continue
                raw_path = str(arguments.get("path") or "").strip()
                if not raw_path:
                    continue
                absolute = sdk.path.absolute(raw_path, base=project)
                found.append((int(row["id"]), sdk.path.normalize(absolute)))
        return found

    def _prune_retrievals(self, sdk, cutoff):
        """Drop retrieval rows too old to answer anything.

        The log is cleared per conversation once that conversation has been
        reflected on — but a conversation that never qualifies (under the
        message floor, outside the window, no assistant message) is never
        reflected on and so never cleared. Those rows would accumulate for the
        life of the install; ``prune_expired`` only knows about kernel tables,
        so nothing else would ever remove them. Past the reflection window they
        cannot be read again by anybody, which makes the window the natural
        retention line.
        """
        try:
            sdk.db.write("DELETE FROM memory_retrievals_v2 WHERE offered_at < ?",
                         [float(cutoff)])
        except sdk.Failed:
            pass

    def _forget_retrievals(self, sdk, cid):
        """Drop the retrieval log for a conversation now reflected on.

        The log answers one question once. Left to accumulate it is an
        unbounded table nothing prunes, since ``prune_expired`` only knows
        about the kernel's own.
        """
        try:
            sdk.db.write("DELETE FROM memory_retrievals_v2 WHERE conversation_id = ?",
                         [int(cid)])
        except sdk.Failed:
            pass

    def _title(self, sdk, cid):
        """The conversation's title, for the curator's orientation only."""
        try:
            record = sdk.conv.read(cid) or {}
        except sdk.Failed:
            return "untitled"
        return str((record.get("conversation") or {}).get("title") or "untitled")

    def _transcript(self, sdk, cid, previous_id, max_id, new_count):
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
               AND id > ?
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
            rows = sdk.db.query(
                sql, [cid, previous_id, max_id, limit], max_rows=MAX_MESSAGES)
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
