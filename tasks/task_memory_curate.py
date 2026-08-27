"""Curate memory entries when a conversation goes quiet.

A conversation ending is an episodic boundary: it is one self-contained piece
of work, which makes it a far better unit to reflect on than a calendar day.
So when a conversation stops being the active one, a subagent reads what
happened and decides what — if anything — is worth keeping.

**The trigger is ``session_conversation_ended`` and nothing else.** The event
names the conversation being left — on a switch, on a session closing, on a
delete — which is exactly the unit of work worth reflecting on, and it fires
for every one of them. The watermark (the id of the last message already
reflected on) is what makes that safe: someone flicking through their history
emits the channel repeatedly for conversations that have not changed, and the
watermark is what turns those into no-ops.

There was an hourly sweep alongside it, publishing the same channel with an
empty payload, and it is gone. Its stated job was recovering an event lost to a
crash, but the event covers every ordinary ending, so in practice it woke up
twenty-four times a day to find nothing — while being a *task that spawns
subagents*, which is the most expensive possible thing to run speculatively.
The gap it covered is real and is now accepted: a conversation whose ending
event was lost to a crash or a shutdown stays eligible in the watermark table
forever and nothing asks about it again. That is one lost reflection against a
standing hourly cost, and losing it silently is what the corpus is designed to
tolerate — an entry that never gets written is a lesson not learned, not a
corruption.

**The corpus is reinforcement learning without weights.** An entry is a
situation, an action, and the result that followed — so retrieval is what makes
a past result bear on a present decision. Actions that produced good results
should be repeated, actions that produced bad ones avoided, and a neutral
result changes no action and is therefore not worth a file. That is the whole
model, and everything below follows from it: an entry with no action in it
cannot change anything, so it is not an entry.

**The curator has two jobs, and one column says which.** ``memory_usage``
records every entry the retrieval service offered and every one ``tool_memory``
then opened. An entry that was *taken* earns a rewrite — it worked, so make it
work harder. Anything the agent worked out for itself is
not in the corpus yet and wants writing down. Offered-but-not-taken is neither,
and saying so is cheap now that it is a recorded fact rather than something
reconstructed by parsing tool calls out of the transcript.

Note what "taken" is *not*: a score. That an entry was opened says the
situation looked close, not that the advice was good. Only reading what
happened next settles that, which is why a model does it and not a counter.

**The curator writes through ``memory`` and has no other way to write.**
It is spawned under the ``memory_curator`` agent profile, whose whitelist does
not include ``edit_file``; the ``memory`` tool addresses entries by name and
derives every path itself. So a background agent nobody is watching cannot
touch anything outside the memory folder — including ``MEMORY.md``, which holds
facts, is inlined into every prompt by the kernel, and belongs to the agent the
user actually talks to.

**It reads widely on purpose**, and the two halves are not in tension. The
profile grants the filesystem read-only (``read_file``, ``grep``, ``glob``),
all three search tools, and ``sql_query`` for the conversation record itself —
because a lesson is usually only visible in what happened *around* the
transcript it was handed, and a curator that can only see one conversation
writes entries about one conversation. Reading everything is dangerous next to
a way to send it somewhere, and the curator has neither egress nor a writable
path out of the memory folder.
"""

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

#: The restricted agent profile a curator runs under. ``service_memory_retrieve``
#: seeds it from ``on_install`` — it cannot be seeded from here, because a
#: task's chain is unattended and writing a kernel setting is refused rather
#: than asked. The kernel refuses to spawn under a profile that does not exist,
#: which is the behaviour that matters: a curator that cannot be restricted
#: must not run unrestricted.
CURATOR_PROFILE = "memory_curator"

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

_PROMPT = """You curate this agent's memory. Read conversation {cid} ("{title}")
below, including its tool calls, and work out what the next agent in this
situation should do differently.

Everything you write goes through the `memory` tool. You address entries
by name and it handles paths, frontmatter and dates; you cannot write anywhere
else, and you must not try.

You can read far more than you can write. The transcript below is the starting
point, not the limit: `grep`, `glob` and `read_file` reach the codebase,
`lexical_search` and `semantic_search` reach the indexed corpus, and `sql_query`
reaches the conversation record — earlier conversations included. Use them when
a lesson only makes sense with what happened around it: whether this went wrong
before, what the code actually does, whether the user has corrected this twice.
Do not go looking when the transcript already settles it.

## Job one: improve what was used

These entries were surfaced to the agent and it opened them:

{used}

Each one earned its place, so make it work harder. Read it with
`memory read`, then `memory update` it against what actually happened:

- sharpen the description so it also fires on the situation that just came up
- tighten the advice if it was vague, incomplete, or partly wrong
- record the real result

Omit `description` on an update to keep the one that is already there; supply
one to replace it. Sharpening it is usually the most valuable part of a
revision, so supply one whenever the situation has widened.

You have evidence of how these performed, which is the one thing that makes a
rewrite an improvement rather than a guess. If one turned out to be wrong or
useless, say so in it — an entry that records its own failure is worth more
than one quietly left standing.

## Job two: write down what is missing

Whatever the agent worked out for itself is not in the corpus yet. Go through
the conversation for actions worth repeating or avoiding that no entry covers.

Here is everything you already have:

{corpus}

If one of those already covers the situation, update it rather than adding a
second — two entries about one situation is how a corpus stops being useful.

## What earns an entry

One test: **will this change what an agent does?** If you cannot name the
action, there is nothing to write.

- An action that produced a good result -> an entry saying to do it.
- An action that produced a bad result, a trap, a correction the user made,
  something that broke -> an entry saying to avoid it, and what to do instead.
  These are worth more than successes; they are what nobody writes down.
- A neutral result changes no action. Write nothing.

**Most conversations deserve nothing, and writing nothing is a correct
outcome.** Never record what happened for its own sake, anything true only
inside this conversation, or anything obvious from reading the code.

## Note or skill

Most of what you write is a **note**: one situation and its lesson. Use a
**skill** when the lesson is a repeatable procedure somebody would follow step
by step — it gets its own folder, so you can add `references/` beside it later.

The `description` is the field that decides whether an entry is ever seen
again. It is matched against what a user says in some future conversation, so
write it as the *situation* — "a PDF yields no text", "about to commit to a
repo Henry owns" — never as a topic label.

Transcript:

{transcript}

Reply with one line: what you wrote or improved, or "nothing worth keeping"."""


class MemoryCurate(BaseTask):
    """Curate memory entries from conversations that have gone quiet."""

    name = "memory_curate"
    description = "Curate memory entries from conversations that have gone quiet."

    trigger = "event"
    trigger_channels = ["session_conversation_ended"]
    reads = []
    writes = ["memory_curations"]
    output_schema = """
        CREATE TABLE IF NOT EXISTS memory_curations (
            conversation_id INTEGER PRIMARY KEY,
            last_message_id INTEGER NOT NULL,
            curated_at REAL
        )
    """
    timeout = 600

    # No ``default_jobs``. This task is driven by the channel above and by
    # nothing on a clock — see the module docstring for what the retired hourly
    # sweep was for and why its absence is affordable.

    requests = ["db.query", "db.write", "agent.spawn", "tool.call",
                "config.read", "conv.read", "session.list", "session.get"]
    # Declared for installation, never imported: the corpus listing goes
    # through ``tool.call``, so the curator and this task read the folder
    # through exactly one implementation.
    dependencies_files = ["tools/tool_memory.py"]
    dependencies_pip = []

    config_settings = [
        ("Memory curation floor", "memory_curate_min_messages",
         "How many new messages a conversation needs before it is curated. "
         "Keeps browsing your history from spawning subagents.",
         4, {"type": "slider", "range": (2, 40, 38), "is_float": False}),
        ("Memory curation window", "memory_curate_max_age_hours",
         "How recently a conversation must have been active to be curated. "
         "Stops a fresh install from curating your entire history.",
         24, {"type": "slider", "range": (1, 168, 167), "is_float": False}),
        ("Curate subagents", "memory_curate_include_subagents",
         "Also curate memories from subagents you spawned. Their whole purpose "
         "is often to go and find something out, so the lesson is real — but "
         "nobody was steering, and every one costs another curator run. "
         "Scheduled subagents are never included: they reuse one conversation "
         "forever, so there is no ending to reflect on.",
         False, {"type": "bool"}),
    ]

    agent_prompt = (
        "When a conversation goes quiet, a background agent reviews it and "
        "writes anything durable into your memory folder."
    )

    def run_event(self, sdk, payload):
        """Reflect on whatever has gone quiet, oldest first.

        ``session_conversation_ended`` names one conversation, so the run
        narrows to it. The unnarrowed path — take whatever the watermark has
        been left holding — is kept because it costs one line and it is what
        anything else publishing this channel would get: a caller with a
        conversation in mind says so, and one without means "whatever is
        due". It was the hourly sweep's path, and the sweep is gone.
        """
        subagents = self._include_subagents(sdk)
        if not subagents and self._from_a_child(payload):
            return sdk.ok([])

        floor = self._floor(sdk)
        cutoff = self._cutoff(sdk)
        self._prune_usage(sdk, cutoff)
        busy = self._busy_conversations(sdk)
        candidates = [row for row in self._candidates(sdk, floor, cutoff, subagents)
                      if row["cid"] not in busy]
        if ended := (payload or {}).get("conversation_id"):
            candidates = [row for row in candidates
                          if row["cid"] == int(ended)]
        if not candidates:
            return sdk.ok([])

        corpus = self._corpus(sdk)
        done = []
        for row in candidates[:MAX_PER_RUN]:
            if self._reflect(sdk, row, corpus):
                done.append({
                    "conversation_id": row["cid"],
                    "last_message_id": row["max_id"],
                    "curated_at": time.time(),
                })
        return sdk.ok(done, llm_summary=f"Reflected on {len(done)} conversation(s).")

    # ── deciding what to reflect on ──────────────────────────────────

    def _include_subagents(self, sdk):
        """Whether subagent conversations are worth curating here.

        Off by default and deliberately so. A subagent's whole purpose is often
        to go and find something out, which makes the lesson real — but nobody
        was steering it, so there is no user correcting a wrong turn, and every
        included conversation costs another curator run on top of the work it
        is reflecting on.
        """
        try:
            return bool(sdk.config.read("memory_curate_include_subagents"))
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
            return max(1, int(sdk.config.read("memory_curate_min_messages") or 4))
        except (sdk.Failed, TypeError, ValueError):
            return 4

    def _cutoff(self, sdk):
        """How far back a conversation may have been active and still count.

        Without this, installing the bundle reflects on *everything you have
        ever said*: the watermark defaults to zero for a conversation nobody
        has looked at, so every message in the archive reads as new and the
        whole history queues up three at a time. That is expensive, spawns an
        agent per conversation, and produces entries about work from months ago
        as though it had just happened.

        A window is the right shape rather than a one-off backfill guard,
        because it keeps being true: a conversation abandoned last spring
        should not become a candidate the day somebody opens it to read.
        Reopening one and adding a message makes it recent again, which is
        exactly when it *is* worth reflecting on.
        """
        try:
            hours = max(1, int(sdk.config.read("memory_curate_max_age_hours") or 24))
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
        nothing an entry could say. Someone can reach the floor without that
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
              LEFT JOIN memory_curations r
                     ON r.conversation_id = m.conversation_id
             WHERE m.id > COALESCE(r.last_message_id, 0)
               AND LOWER(m.role) IN ('user', 'assistant')
               AND COALESCE(m.author, '') = ''
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

    def _reflect(self, sdk, row, corpus):
        """Run one curator subagent. True when the watermark may advance."""
        cid = int(row["cid"])
        previous_id = int(row.get("previous_id") or 0)
        transcript = self._transcript(
            sdk, cid, previous_id, int(row["max_id"]), int(row["new_count"]))
        if not transcript:
            # Nothing readable to reflect on, but the messages have been
            # considered — advancing stops us reconsidering them hourly.
            return True
        prompt = _PROMPT.format(cid=cid, title=self._title(sdk, cid),
                                transcript=transcript, corpus=corpus,
                                used=self._used_entries(sdk, cid))
        try:
            report = sdk.agent.spawn(
                prompt, title=f"{CURATOR_TITLE} conversation {cid}",
                profile=CURATOR_PROFILE, wait=True)
        except sdk.Failed as error:
            # Includes the profile not existing, which is the one failure worth
            # not working around: the curator writes unattended, and the
            # profile is what keeps it inside the memory folder. Returning
            # False leaves the watermark where it is, so the next sweep retries
            # once the profile exists — which normally happened at install, and
            # otherwise takes one line of /config.
            sdk.log(f"memory curator failed for conversation {cid}: {error}",
                    level="warning")
            return False
        if not (report or {}).get("ok"):
            sdk.log(f"memory curator did not finish for conversation {cid}: "
                    f"{(report or {}).get('error') or 'no report'}",
                    level="warning")
            return False
        return True

    def _used_entries(self, sdk, cid):
        """Entries this conversation was offered and actually opened.

        One column, because ``tool_memory`` wrote the fact down when it
        happened. This used to be reconstructed: read every offer out of one
        table, then scan every assistant message for a ``read_file`` tool call,
        decode two layers of JSON, absolutize and normalize the path it named,
        and compare — all to infer something the agent could simply have told
        us. A dedicated recall tool is what made it tell us.

        An entry recalled *without* being offered counts too, and is recorded
        the same way. The agent went looking for it, which is at least as
        strong a signal as taking one it was handed.

        This decides which job the curator has, and nothing more. It is not a
        score — that an entry was opened says the situation looked close, not
        that the advice was any good, and only reading what happened next can
        settle that.
        """
        try:
            rows = sdk.db.query(
                "SELECT DISTINCT name FROM memory_usage"
                " WHERE conversation_id = ? AND recalled_at IS NOT NULL"
                " ORDER BY name", [int(cid)], max_rows=100) or []
        except sdk.Failed:
            rows = []
        names = [str(row.get("name") or "") for row in rows if row.get("name")]
        if not names:
            return ("None. Memory was either not surfaced or not opened, so "
                    "nothing here has been shown to help yet — skip job one.")
        return "\n".join(f"- {name}" for name in names)

    def _corpus(self, sdk):
        """Every entry that exists, as name and description.

        Handed to the curator whole rather than left to a search, because the
        decision it drives — improve this one, or add another — is exactly the
        decision a search can get wrong by missing something. A corpus small
        enough to be useful is small enough to list.
        """
        try:
            entries = sdk.tools.call("memory", action="list") or []
        except sdk.Failed as error:
            sdk.log(f"could not list memory entries: {error}", level="info")
            return "(could not be listed)"
        lines = [f"- {row.get('name')}"
                 f"{' (skill)' if row.get('kind') == 'skill' else ''}"
                 f" — {row.get('description') or '(no description)'}"
                 for row in entries if row.get("name")]
        return "\n".join(lines) if lines else "(nothing yet — the corpus is empty)"

    def _prune_usage(self, sdk, cutoff):
        """Drop offers nobody ever took, once they are too old to answer anything.

        **Recalls are kept.** They are the record of which entries earn their
        place, which is the input to a future pass over what nobody has used in
        months, and clearing the table per conversation — as this did — throws
        that away for a table that was never the problem. Offers are the
        volume: five a turn, forever, and past the reflection window no
        conversation can be reflected on again so nothing can ever read them.
        Recalls are rare, small, and the data.
        """
        try:
            sdk.db.write("DELETE FROM memory_usage"
                         " WHERE recalled_at IS NULL AND offered_at < ?",
                         [float(cutoff)])
        except sdk.Failed:
            pass

    def _title(self, sdk, cid):
        """The conversation's title, for the curator's orientation only."""
        try:
            # ``limit=0`` because this wants a title, and the transcript it
            # was pulling to get one reached 20 MB on a long conversation.
            record = sdk.conv.read(cid, limit=0) or {}
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

        Rows with an ``author`` are excluded, and that is a different exclusion
        from ``role <> 'system'`` beside it. A state marker is bookkeeping the
        transcript never held; an authored row is the *kernel* wearing the
        person's role — a cancel notice, a doorman's note, a compaction bridge.
        Without this the curator read all of them as things the user said and
        wrote them into MEMORY.md as the user's own words.
        """
        sql = """
            SELECT role, content, tool_name
              FROM conversation_messages
             WHERE conversation_id = ?
               AND id > ?
               AND id <= ?
               AND LOWER(COALESCE(role, '')) <> 'system'
               AND COALESCE(author, '') = ''
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
