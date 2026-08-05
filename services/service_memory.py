"""Automatic memory retrieval.

Every turn, search the memory folder for notes relevant to what the user just
said and put them in the prompt.

The folder holds **actions**: each note is a situation, an action, and the
result that followed, so retrieval is what makes a past result bear on a
present decision. Facts live in ``MEMORY.md``, which the kernel inlines
directly and nothing here touches.

**The prompt gets situations, never the notes themselves.** The only decision
to make from the prompt is whether a past situation is the present one, and the
situation alone answers that; what was tried and how it turned out is what the
file is for. Injecting the advice as well was tried and had to come out — it
grew with the corpus, truncated long notes into advice stripped of its context,
and destroyed the one observable signal in the system. With the content already
in the prompt there is no reason to open the file, so nothing downstream can
tell which notes were used, and the curator needs exactly that to know whether
it is improving an old note or writing a new one. So this service also records
what it surfaced (``memory_retrievals_v2``); the read is the other half. Each
offer is tied to the user-message id that caused it, so a later reader can
prove that the file was opened after it was offered.

The retrieval itself is one ``hybrid_search`` call at ``turn_start``, with the
user's own message as the query. That is deliberate on both counts. The hook
runs synchronously on the drive thread and sets the latency floor for every
reply, so there is no model call and no subagent here. And the user's words
*are* the retrieval cue — rewriting them into "better" search terms costs a
round trip to guess at something the person already said.

Writing is not this service's job. Notes are written by ``task_memory_reflect``
when a conversation goes quiet, and by the agent itself with ``edit_file``;
the folder is inside the workspace, so neither needs approval.
"""

import time

from guest.bases import BaseService

#: How much of a note to read when building its line. Only the frontmatter is
#: wanted, and that is at the top.
HEAD_CHARS = 2000


#: Where situation/action/result notes live, under the memory root. Everything
#: else in ``memory/`` — ``MEMORY.md``, the README, drafts, anything the agent
#: leaves lying around — is outside it and therefore outside retrieval.
NOTES_DIRNAME = "actions"


def _memory_root(sdk):
    """The folder this service watches. One per install, not per user.

    Per-user memory folders are a real thing the retrieval side cannot honour
    yet: ``hybrid_search`` filters by folder prefix, so a user subfolder would
    need its own filter and the pointer block would have to be rebuilt per
    identity. Single-user installs are the case that works today.
    """
    return sdk.path.join(sdk.paths.get("workspace"), "memory")


def _notes_root(sdk):
    """The only folder retrieval searches.

    Membership is location, not content, and that is the point. Requiring a
    ``when`` in the frontmatter made *being a note* a property the writer had
    to restate correctly in every file, and getting it subtly wrong — no
    fences, the key in the body, a capital in the wrong place — made the note
    silently unreachable. A path cannot be subtly wrong.

    It also retires an exception list. Filtering the memory root meant naming
    every file that was not a note (``MEMORY.md``, the README), which is a list
    that grows and whose omissions are invisible. And it moves the filter into
    the search: a draft outside this folder never ranks, rather than ranking
    and being read and discarded on every turn.
    """
    return sdk.path.join(_memory_root(sdk), NOTES_DIRNAME)


def _frontmatter(text):
    """Parse the leading ``---`` block into a dict, tolerating anything.

    Deliberately not a YAML parser: a note is written by a language model and
    the failure mode that matters is a malformed block taking the whole turn
    down. Unparseable lines are skipped, and a note with no frontmatter at all
    is still a usable pointer — it just falls back to its excerpt.
    """
    body = text.lstrip()
    if not body.startswith("---"):
        return {}
    end = body.find("\n---", 3)
    if end == -1:
        return {}
    fields = {}
    for line in body[3:end].splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            fields[key.strip().lower()] = _unquote(value.strip())
    return fields


def _unquote(value):
    """Drop wrapping quotes, and only wrapping ones.

    ``strip("'\\"")`` eats a trailing quote whether or not anything opened it,
    so a situation ending in a quoted word — ``a task with trigger = "event"``
    — silently lost its last character in the prompt.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


class Memory(BaseService):
    """Surface relevant memory notes at the start of every turn."""

    name = "memory"
    description = "Finds memory notes relevant to the current message and points the agent at them."

    exports = []
    hooks = {"turn_start": "on_turn_start"}
    requests = ["paths.get", "config.read", "config.write",
                "tool.call", "fs.read", "fs.list", "fs.write",
                "db.define", "db.query", "db.write",
                "session.add_prompt_extra"]
    dependencies_files = ["tools/tool_hybrid_search.py"]
    dependencies_pip = []

    config_settings = [
        ("Memory pointers", "memory_max_pointers",
         "How many memory notes to surface at the start of each turn. 0 disables retrieval.",
         5, {"type": "slider", "range": (0, 15, 15), "is_float": False}),
    ]

    agent_prompt = (
        "## Memory\n"
        "`memory/actions/` in your workspace holds **actions**: what to do, or "
        "not do, in a situation that has come up before. Each note is one "
        "situation, the action taken, and the result that followed. Only that "
        "folder is searched, so a note written anywhere else is never found — "
        "the rest of `memory/` is for things that are not notes.\n"
        "Situations matching the current message are listed above under "
        "'Situations you have been in before'. That list gives you the "
        "situation and a path, never the advice — read the file when a "
        "situation looks close enough to yours to be worth learning from. It "
        "is advice from a past case, not an instruction: weigh whether this "
        "case really is that one.\n"
        "Retrieval happens once, on the user's message. When something comes "
        "up mid-turn that feels familiar, search the folder yourself with "
        "`hybrid_search` scoped to it — cheap, and often faster than working "
        "it out again.\n"
        "Write a note when you learn an action worth repeating or avoiding "
        "and you want it to outlive this conversation; no approval is needed "
        "inside your workspace. Keep the format: `when:` the situation that "
        "should bring it back, written as a situation rather than a topic, "
        "since that is what retrieval matches; `do:` or `avoid:` the action; "
        "`because:` what actually happened. A note with no action in it "
        "cannot change anything, so do not write one.\n"
        "Facts, names and preferences with no action attached are not memory "
        "notes — those go in MEMORY.md, which is yours to maintain and is "
        "already in this prompt.\n"
        "Anything you do not write down is reviewed after the conversation "
        "ends, so there is no need to record as you go."
    )

    def start(self, sdk):
        """Make the folder and the log. Indexing waits for somebody to ask.

        ``sync_directories`` is a kernel setting, so writing it is UNSAFE — and
        a service loading at boot has no session, which makes the chain
        unattended, which means the write is **refused outright rather than
        asked about** (``sandbox/approval.py``, step 3). Seeding here therefore
        did nothing at all except log a warning, and the symptom was the worst
        available: the folder was never indexed, so retrieval returned nothing,
        forever, while everything looked healthy. It happens on the first
        attended turn instead, where a person is present to answer.
        """
        self._seeded = False
        self._ensure_folder(sdk, _memory_root(sdk))
        self._ensure_log(sdk)

    def _ensure_log(self, sdk):
        """The retrieval log this service owns.

        Defined here rather than declared by the reflect task, because the
        writer should own the schema: the task only reads it, and a task that
        is not installed must not take retrieval logging down with it.
        """
        try:
            sdk.db.define(
                "CREATE TABLE IF NOT EXISTS memory_retrievals_v2 ("
                " conversation_id INTEGER,"
                " path TEXT,"
                " offered_message_id INTEGER,"
                " offered_at REAL,"
                " PRIMARY KEY (conversation_id, path, offered_message_id))")
        except sdk.Failed as error:
            sdk.log(f"could not create the retrieval log: {error}",
                    level="warning")

    def stop(self, sdk):
        """Nothing is held open."""
        return True

    # ── lifecycle helpers ────────────────────────────────────────────

    def _ensure_folder(self, sdk, root):
        """Create the folder, by writing the note that explains it.

        There is no ``fs.mkdir`` Request and there does not need to be: a
        folder that exists but explains nothing is worse than one that arrives
        with its own README, and writing the file makes the directory.

        ``actions/`` is deliberately *not* pre-created. A placeholder inside it
        would be searched like any other file there, found to have no ``when``,
        and reported as a broken note every single turn. Writing a file creates
        its parents, so the first note makes the folder, and searching a folder
        that does not exist yet correctly finds nothing.
        """
        try:
            sdk.fs.list(root)
            return
        except sdk.Failed:
            pass
        try:
            sdk.fs.write(sdk.path.join(root, "README.md"), _README)
        except sdk.Failed as error:
            sdk.log(f"could not create the memory folder: {error}",
                    level="warning")

    def _maybe_seed(self, sdk, ctx, root):
        """Ask, once, for the memory folder to be indexed.

        Only from an attended turn: an unattended one cannot draw a dialog, so
        the Request would be refused and a naive once-only flag would spend the
        single attempt on a subagent's turn and never try again. The attempt is
        marked spent whatever the answer, including a refusal — asking again
        every turn would be worse than not asking.

        This does block the turn until it is answered, which is why it happens
        once and says plainly what it is for.
        """
        if self._seeded or not getattr(ctx, "attended", False):
            return
        self._seeded = True
        self._ensure_indexed(sdk, root)

    def _ensure_indexed(self, sdk, root):
        """Add the folder to ``sync_directories`` so the pipeline indexes it."""
        try:
            existing = sdk.config.read("sync_directories") or []
        except sdk.Failed as error:
            sdk.log(f"could not read sync_directories: {error}", level="warning")
            return
        if not isinstance(existing, list):
            existing = [existing]
        wanted = sdk.path.normalize(root)
        if any(sdk.path.normalize(str(entry)) == wanted for entry in existing):
            return
        try:
            sdk.config.write("sync_directories", list(existing) + [root])
            sdk.log(f"memory folder added to sync_directories: {root}")
        except sdk.Denied:
            sdk.log("memory folder is not indexed — retrieval will stay empty "
                    "until sync_directories includes it", level="warning")
        except sdk.Failed as error:
            sdk.log(f"could not add the memory folder to sync_directories: {error}",
                    level="warning")

    # ── the hook ─────────────────────────────────────────────────────

    def on_turn_start(self, sdk, ctx, payload):
        """Search the memory folder and point the agent at what matched.

        Every failure path abstains rather than raising. ``hybrid_search`` is
        absent until the indexing packages are installed, an empty index is the
        normal state of a fresh install, and neither is a reason for a turn to
        fail.
        """
        self._maybe_seed(sdk, ctx, _memory_root(sdk))
        try:
            limit = int(sdk.config.read("memory_max_pointers") or 0)
        except (sdk.Failed, TypeError, ValueError):
            limit = 5
        if limit <= 0:
            return None

        user_message = self._latest_user_message(sdk, ctx)
        if not user_message:
            return None
        message_id, query = user_message

        hits = self._search(sdk, query, limit)
        if not hits:
            return None

        offered = []
        block = self._render(sdk, hits, offered)
        if not block:
            return None
        try:
            sdk.session.add_prompt(block, key=ctx.session_key, slot="memory")
        except sdk.Failed as error:
            # Nothing was shown, so nothing was offered. Recording it anyway
            # would tell the curator a note had been surfaced and ignored when
            # the agent never saw it — a false negative in the one pair its
            # branch is chosen by.
            sdk.log(f"could not inject memory pointers: {error}", level="warning")
            return None
        self._log_offered(sdk, ctx, offered, message_id)
        return None

    def _log_offered(self, sdk, ctx, offered, message_id):
        """Record which notes were surfaced in this conversation.

        Half of a pair: the curator later checks which of these the agent went
        on to open, and that is what tells it whether to improve an existing
        note or write a new one. The prompt is not stored anywhere, so this is
        the only place the offer is knowable — without it the curator can see
        the read but never what prompted it.

        Keyed on (conversation, path, user message). The message id lets the
        curator require a later exact ``read_file`` call for this offer.
        """
        cid = getattr(ctx, "conversation_id", None)
        if not (cid and offered):
            return
        now = time.time()
        for path in offered:
            try:
                sdk.db.write(
                    "INSERT OR REPLACE INTO memory_retrievals_v2 "
                    "(conversation_id, path, offered_message_id, offered_at) "
                    "VALUES (?, ?, ?, ?)",
                    [int(cid), str(path), int(message_id), now])
            except sdk.Failed as error:
                sdk.log(f"could not record a memory retrieval: {error}",
                        level="warning")
                return

    def _latest_user_message(self, sdk, ctx):
        """The id and text of the message the turn is about.

        One row. ``sdk.conv.read`` would have been the obvious call and is the
        wrong one here: it answers with the *entire* conversation, so a long
        one crossed the wire in full, every turn, on the thread that sets the
        latency floor for every reply — to find a single string at the end of
        it.

        Read rather than held between calls: a cached "last message" would go
        stale the moment two sessions were live.
        """
        cid = getattr(ctx, "conversation_id", None)
        if not cid:
            return ""
        try:
            rows = sdk.db.query(
                "SELECT id, content FROM conversation_messages"
                " WHERE conversation_id = ? AND LOWER(role) = 'user'"
                "   AND COALESCE(content, '') <> ''"
                " ORDER BY id DESC LIMIT 1", [int(cid)], max_rows=1)
        except sdk.Failed:
            return ""
        if not rows:
            return ""
        text = str(rows[0].get("content") or "").strip()[:2000]
        message_id = rows[0].get("id")
        if not (message_id and text):
            return None
        return int(message_id), text

    def _search(self, sdk, query, limit):
        """One hybrid search, scoped to the memory folder."""
        try:
            results = sdk.tools.call("hybrid_search", query=query,
                                     folder=_notes_root(sdk),
                                     max_results=limit)
        except sdk.Failed as error:
            sdk.log(f"memory search unavailable: {error}", level="info")
            return []
        return [hit for hit in (results or []) if hit.get("path")]

    def _render(self, sdk, hits, offered):
        """Build the block, and record what was offered.

        ``offered`` collects the paths so the caller can log them: what the
        curator needs later is the pair "surfaced, then opened", and only this
        half is knowable here.
        """
        entries = []
        malformed = []
        for hit in hits:
            path = str(hit["path"])
            if entry := self._entry(sdk, path):
                entries.append(entry)
                offered.append(path)
            else:
                malformed.append(path)
        if malformed:
            # Everything here is in the notes folder, so a file with no
            # ``when`` is a *broken note* rather than an unrelated file — a
            # louder thing than it was when this folder held everything, and
            # worth naming, since the symptom is a note that ranks well and is
            # never once offered.
            sdk.log("memory notes with no 'when' were skipped: "
                    + ", ".join(sdk.path.name(p) for p in malformed),
                    level="warning")
        if not entries:
            return ""
        return ("## Situations you have been in before\n"
                "These past situations matched this message. Each is a note "
                "recording what was tried and what came of it — read one when "
                "the situation looks close enough to yours to be worth "
                "learning from.\n\n"
                + "\n\n".join(entries))

    def _entry(self, sdk, path):
        """One note's line, or "" for a file that is not a note.

        The situation and nothing else, because the only decision to make from
        the prompt is whether this past situation is the present one. What was
        attempted and how it turned out is what the agent opens the file for.

        Injecting the action too was tried and had to come out. It cost tokens
        proportional to the corpus, it truncated long notes into advice with no
        context, and — worst — it destroyed the one observable signal in the
        system: with the content already in the prompt there is no reason to
        open the file, so nothing downstream can tell which notes were used.
        The read is what tells the curator whether its job this time is to
        improve an existing note or write a new one.

        Whether a file *is* a note is decided by the folder it sits in, not by
        this — see :func:`_notes_root`. What is left for the frontmatter is
        whether the note can be *rendered*, and a note with no ``when`` cannot:
        there is no situation to show, and falling back to the matched chunk
        would put a fragment with no context into a list that promises
        situations. That is a broken note rather than a foreign file, so the
        caller says so out loud.
        """
        try:
            situation = _frontmatter(sdk.fs.read(path)[:HEAD_CHARS]).get("when")
        except sdk.Failed:
            return ""
        return f"- {situation}\n  ({path})" if situation else ""


_README = """# Memory

    actions/     one file per situation — searched, and the only thing that is
    MEMORY.md    facts, inlined into the agent's prompt in full
    (anything else lives here and is ignored)

`actions/` holds what to do, or not do, in a situation that has come up before.
Each note is a situation, an action, and the result that followed. A note
outside that folder is never found, which is also what makes the folder safe to
keep drafts and scratch files in.

```
---
when: A PDF yields no text, or extraction produces an empty document
do: Check the parser is installed before assuming the file is broken
because: Spent an hour on a corrupt-PDF theory when parser-pdf was not installed
updated: 2026-01-01
source: conversation 1
---
```

Use `avoid:` in place of `do:` when the lesson is not to do something, and say
what to do instead in the same field.

`when` is the field that does the work. It is matched against what the user
says in some future conversation, so write it as the situation that should
bring the note back, never as a topic label.

A note with no action in it cannot change what anyone does, so it is not a note.
That is the test for whether something belongs in `actions/` at all.

Two things deliberately do not go there. **Facts** — names, paths, which machine
something runs on, a preference with no action attached — belong in MEMORY.md,
which is inlined into the prompt directly. And **records of what happened**:
this is not a journal, and a note that does not change a future decision is
noise that makes the rest harder to find.

Notes are written and improved automatically after a conversation ends, but
they are ordinary markdown — edit or delete them freely.
"""
