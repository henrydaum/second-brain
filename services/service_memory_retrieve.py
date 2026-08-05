"""Automatic memory retrieval — the half nobody has to ask for.

Every turn, search the memory folder for entries relevant to what the user just
said and put their names in the prompt.

One of four, in two pairs, each with an automatic half and an agent-invoked
one: ``memory_retrieve`` and ``memory_recall`` read, ``memory_reflect`` and
``memory_curate`` write.

The folder holds two kinds of entry and retrieval does not distinguish between
them. A **note** (``notes/<name>.md``) is one situation and what to do about
it. A **skill** (``skills/<name>/SKILL.md``) is a repeatable procedure that may
carry its own references and scripts. Both carry agentskills.io frontmatter —
``name`` and ``description`` — which is the whole reason they can rank in one
list and render as one shape. Facts live in ``MEMORY.md``, which the kernel
inlines directly and nothing in this suite touches.

**The prompt gets names and descriptions, never bodies.** The only decision to
make from the prompt is whether a past situation is the present one, and the
description answers exactly that; what to do about it is what the entry is for.
Injecting the body as well was tried and had to come out — it grew with the
corpus, truncated long entries into advice stripped of its context, and
destroyed the one observable signal in the system. With the content already in
the prompt there is no reason to open anything, and the curator needs to know
which entries were opened to tell whether it is improving one or writing a new
one. So this service also records what it surfaced (``memory_usage``);
``tool_memory_recall`` records the other half.

The retrieval itself is one ``hybrid_search`` call at ``turn_start``, with the
user's own message as the query. That is deliberate on both counts. The hook
runs synchronously on the drive thread and sets the latency floor for every
reply, so there is no model call and no subagent here. And the user's words
*are* the retrieval cue — rewriting them into "better" search terms costs a
round trip to guess at something the person already said.

Writing is not this service's job and cannot be: only ``tool_memory_curate``
writes entries, whether the caller is the agent mid-conversation or the curator
subagent ``task_memory_reflect`` spawns after one ends.
"""

import time

from guest.bases import BaseService

#: How much of an entry to read when building its line. Only the frontmatter is
#: wanted, and that is at the top.
HEAD_CHARS = 2000

#: How long a description may be in the prompt block. Long enough to recognise
#: a situation, short enough that fifteen of them are not the prompt.
MAX_DESCRIPTION_CHARS = 200

#: Where entries live, under the memory root, and the only two places searched.
#: Everything else in ``memory/`` — ``MEMORY.md``, the README, drafts, anything
#: the agent leaves lying around — is outside them and therefore outside
#: retrieval. Must match the constants in ``tool_memory_recall`` and
#: ``tool_memory_curate``.
MEMORY_DIRNAME = "memory"
NOTES_DIRNAME = "notes"
SKILLS_DIRNAME = "skills"

#: The agent profile a curator subagent runs under. Seeded here because this is
#: the only part of the suite that ever runs attended, and writing a kernel
#: setting from an unattended chain is refused rather than asked. Must match
#: ``CURATOR_PROFILE`` in ``task_memory_reflect``.
CURATOR_PROFILE = "memory_curator"

#: What that profile may do. Four tools, and the absence of ``edit_file`` is
#: the point: a curator writes through ``memory_curate``, which cannot address
#: a file outside the memory folder. ``read_file`` is for a skill's own
#: references; ``hybrid_search`` pulls its two sub-searches in on its own,
#: because a whitelist is closed over declared tool dependencies.
CURATOR_TOOLS = ["memory_recall", "memory_curate", "hybrid_search", "read_file"]


def _memory_root(sdk):
    """The folder this service watches. One per install, not per user.

    Per-user memory folders are a real thing the retrieval side cannot honour
    yet: ``hybrid_search`` filters by folder prefix, so a user subfolder would
    need its own filter and the pointer block would have to be rebuilt per
    identity. Single-user installs are the case that works today.
    """
    return sdk.path.join(sdk.paths.get("workspace"), MEMORY_DIRNAME)


def _entry_dirs(sdk):
    """The two folders that hold entries.

    Membership is location, not content, and that is the point. Requiring a
    field in the frontmatter made *being an entry* a property the writer had to
    restate correctly in every file, and getting it subtly wrong — no fences,
    the key in the body, a capital in the wrong place — made the entry silently
    unreachable. A path cannot be subtly wrong.

    It also retires an exception list. Filtering the memory root meant naming
    every file that was not an entry (``MEMORY.md``, the README), which is a
    list that grows and whose omissions are invisible.
    """
    root = _memory_root(sdk)
    return (sdk.path.join(root, NOTES_DIRNAME),
            sdk.path.join(root, SKILLS_DIRNAME))


def _frontmatter(text):
    """Parse the leading ``---`` block into a dict, tolerating anything.

    Deliberately not a YAML parser: an entry is written by a language model and
    the failure mode that matters is a malformed block taking the whole turn
    down. Unparseable lines are skipped.
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
    so a description ending in a quoted word — ``a task with trigger =
    "event"`` — silently lost its last character in the prompt.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


class MemoryRetrieve(BaseService):
    """Surface relevant memory entries at the start of every turn."""

    name = "memory_retrieve"
    description = "Finds memory notes and skills relevant to the current message and points the agent at them."

    exports = []
    hooks = {"turn_start": "on_turn_start"}
    requests = ["paths.get", "config.read", "config.write",
                "tool.call", "fs.read", "fs.list", "fs.write", "fs.delete",
                "db.define", "db.query", "db.write",
                "session.add_prompt_extra"]
    dependencies_files = ["tools/tool_hybrid_search.py",
                          "tools/tool_memory_recall.py",
                          "tools/tool_memory_curate.py"]
    dependencies_pip = []

    config_settings = [
        ("Memory pointers", "memory_max_pointers",
         "How many memory entries to surface at the start of each turn. 0 disables retrieval.",
         5, {"type": "slider", "range": (0, 15, 15), "is_float": False}),
    ]

    agent_prompt = (
        "## Memory\n"
        "`memory/` in your workspace holds what you have learned: `notes/` for "
        "a situation and what to do about it, `skills/` for a repeatable "
        "procedure with its own references. Only those two folders are "
        "searched, so the rest of `memory/` is free for drafts and scratch "
        "files.\n"
        "Entries matching the current message are listed above under 'Things "
        "you have done before'. That list gives you a name and a description, "
        "never the entry itself — `memory_recall` one when its situation looks "
        "close enough to yours to be worth learning from. It is advice from a "
        "past case, not an instruction: weigh whether this case really is that "
        "one.\n"
        "Facts, names and preferences with no action attached are not entries "
        "— those go in MEMORY.md, which is yours to maintain and is already in "
        "this prompt.\n"
        "Anything you do not write down is reviewed after the conversation "
        "ends, so there is no need to record as you go."
    )

    def start(self, sdk):
        """Make the folders and the usage table. Indexing waits to be asked.

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
        self._ensure_usage_table(sdk)

    def _ensure_usage_table(self, sdk):
        """The one table that records the life of a memory: offered, then taken.

        Defined here because this service is the first writer — it inserts a
        row per offer. ``tool_memory_recall`` fills ``recalled_at`` and
        ``task_memory_reflect`` only reads. One table rather than three answers
        every question anyone has asked of it so far: which entries this
        conversation actually used, and — later, when there is a pruning pass
        — which entries nobody has used in months.
        """
        try:
            sdk.db.define(
                "CREATE TABLE IF NOT EXISTS memory_usage ("
                " id INTEGER PRIMARY KEY,"
                " name TEXT NOT NULL,"
                " conversation_id INTEGER,"
                " offered_at REAL,"
                " recalled_at REAL)")
            # Every recall looks a pending offer up by (conversation, name),
            # against a table that gains a row per offer forever and is pruned
            # only of offers nobody took. Without this that is a scan, and it
            # gets slower precisely because the design is working — the whole
            # point of keeping recalls is to have a history to query.
            sdk.db.define(
                "CREATE INDEX IF NOT EXISTS memory_usage_lookup"
                " ON memory_usage (conversation_id, name, recalled_at)")
        except sdk.Failed as error:
            sdk.log(f"could not create the memory usage table: {error}",
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

        ``notes/`` and ``skills/`` are deliberately *not* pre-created. A
        placeholder inside either would be searched like any other file there,
        found to have no description, and reported as broken every single turn.
        Writing a file creates its parents, so the first entry makes its folder,
        and searching a folder that does not exist yet correctly finds nothing.
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

    def _maybe_seed(self, sdk, ctx):
        """Ask, once, for the two things this suite needs a person to allow.

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
        self._ensure_indexed(sdk, _memory_root(sdk))
        self._ensure_curator_profile(sdk)

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

    def _ensure_curator_profile(self, sdk):
        """Make sure the restricted profile the curator spawns under exists.

        ``task_memory_reflect`` names it and the kernel refuses to spawn under
        a profile that is not configured — deliberately, since substituting
        ``default`` would run a background agent with every installed tool
        while everything looked confined. So somebody has to write it, and this
        is the only part of the suite that ever runs where a person can answer.

        Never overwritten. Once it exists it is the user's, and narrowing or
        widening it is their business.
        """
        try:
            profiles = sdk.config.read("agent_profiles") or {}
        except sdk.Failed as error:
            sdk.log(f"could not read agent_profiles: {error}", level="warning")
            return
        if not isinstance(profiles, dict) or CURATOR_PROFILE in profiles:
            return
        updated = dict(profiles)
        updated[CURATOR_PROFILE] = {
            "llm": "default",
            "prompt_suffix": "",
            "whitelist_or_blacklist_tools": "whitelist",
            "tools_list": list(CURATOR_TOOLS),
        }
        try:
            sdk.config.write("agent_profiles", updated)
            sdk.log(f"added the {CURATOR_PROFILE!r} agent profile")
        except sdk.Denied:
            sdk.log(f"the {CURATOR_PROFILE!r} agent profile was not created — "
                    "memory curation will not run until it exists",
                    level="warning")
        except sdk.Failed as error:
            sdk.log(f"could not create the curator profile: {error}",
                    level="warning")

    # ── the hook ─────────────────────────────────────────────────────

    def on_turn_start(self, sdk, ctx, payload):
        """Search the memory folder and point the agent at what matched.

        Every failure path abstains rather than raising. ``hybrid_search`` is
        absent until the indexing packages are installed, an empty index is the
        normal state of a fresh install, and neither is a reason for a turn to
        fail.
        """
        self._maybe_seed(sdk, ctx)
        try:
            limit = int(sdk.config.read("memory_max_pointers") or 0)
        except (sdk.Failed, TypeError, ValueError):
            limit = 5
        if limit <= 0:
            return None

        query = self._latest_user_message(sdk, ctx)
        if not query:
            return None

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
            # would tell the curator an entry had been surfaced and ignored
            # when the agent never saw it — a false negative in the one pair
            # its branch is chosen by.
            sdk.log(f"could not inject memory pointers: {error}", level="warning")
            return None
        self._log_offered(sdk, ctx, offered)
        return None

    def _log_offered(self, sdk, ctx, offered):
        """Record which entries were surfaced in this conversation.

        Half of a pair: ``tool_memory_recall`` fills in ``recalled_at`` if the
        agent goes on to open one, and that is what tells the curator whether
        to improve an existing entry or write a new one. The prompt is not
        stored anywhere, so this is the only place the offer is knowable —
        without it a recall can be seen but never what prompted it.
        """
        cid = getattr(ctx, "conversation_id", None)
        if not (cid and offered):
            return
        now = time.time()
        for name in offered:
            try:
                sdk.db.write(
                    "INSERT INTO memory_usage"
                    " (name, conversation_id, offered_at, recalled_at)"
                    " VALUES (?, ?, ?, NULL)", [str(name), int(cid), now])
            except sdk.Failed as error:
                sdk.log(f"could not record a memory offer: {error}",
                        level="warning")
                return

    def _latest_user_message(self, sdk, ctx):
        """The text of the message the turn is about.

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
                "SELECT content FROM conversation_messages"
                " WHERE conversation_id = ? AND LOWER(role) = 'user'"
                "   AND COALESCE(content, '') <> ''"
                " ORDER BY id DESC LIMIT 1", [int(cid)], max_rows=1)
        except sdk.Failed:
            return ""
        if not rows:
            return ""
        return str(rows[0].get("content") or "").strip()[:2000]

    def _search(self, sdk, query, limit):
        """One hybrid search, scoped to the memory folder.

        Scoped to the *root* rather than to each entry folder in turn, because
        ``hybrid_search`` takes one prefix and two calls would mean ranking
        notes against notes and skills against skills — which is exactly the
        separation this design exists to remove. The non-entry files are
        dropped afterwards instead, and the over-fetch is what stops
        ``MEMORY.md`` and the README crowding real results out of the limit.
        """
        try:
            results = sdk.tools.call("hybrid_search", query=query,
                                     folder=_memory_root(sdk),
                                     max_results=limit * 3)
        except sdk.Failed as error:
            sdk.log(f"memory search unavailable: {error}", level="info")
            return []
        notes_dir, skills_dir = _entry_dirs(sdk)
        notes = sdk.path.normalize(notes_dir)
        skills = sdk.path.normalize(skills_dir)
        kept, seen = [], set()
        for hit in results or []:
            path = str(hit.get("path") or "")
            if not path:
                continue
            normalized = sdk.path.normalize(path)
            if normalized.startswith(notes):
                entry = (sdk.path.stem(path), "note", path)
            elif normalized.startswith(skills):
                entry = self._skill_of(sdk, skills_dir, path)
            else:
                continue  # MEMORY.md, the README, a draft — not an entry
            if entry is None or entry[0] in seen:
                continue
            seen.add(entry[0])
            kept.append(entry)
            if len(kept) >= limit:
                break
        return kept

    def _skill_of(self, sdk, skills_dir, path):
        """Which skill a hit inside ``skills/`` belongs to.

        A skill is a folder, so its references and scripts are indexed as
        documents in their own right and any of them can be what matched. The
        agent should be pointed at the skill either way — its ``SKILL.md`` is
        what says when to open the rest — so every hit under a skill folder
        collapses onto the skill itself, and the dedupe above keeps one line.
        """
        relative = sdk.path.normalize(path)[len(sdk.path.normalize(skills_dir)):]
        parts = [part for part in relative.replace("\\", "/").split("/") if part]
        if not parts:
            return None
        name = parts[0]
        return name, "skill", sdk.path.join(skills_dir, name, "SKILL.md")

    def _render(self, sdk, hits, offered):
        """Build the block, and record what was offered.

        ``offered`` collects the names so the caller can log them: what the
        curator needs later is the pair "surfaced, then recalled", and only
        this half is knowable here.
        """
        entries = []
        malformed = []
        for name, kind, path in hits:
            if description := self._description(sdk, path):
                label = f"{name} (skill)" if kind == "skill" else name
                entries.append(f"- {label} — {description}")
                offered.append(name)
            else:
                malformed.append(name)
        if malformed:
            # Everything here is inside an entry folder, so a file with no
            # description is a *broken entry* rather than an unrelated file,
            # and worth naming: the symptom is otherwise an entry that ranks
            # well and is never once offered.
            sdk.log("memory entries with no description were skipped: "
                    + ", ".join(malformed), level="warning")
        if not entries:
            return ""
        return ("## Things you have done before\n"
                "These matched what was just said. Each is a note or skill you "
                "wrote earlier; the description is all you get here, so "
                "`memory_recall` the name when one looks close enough to your "
                "situation to be worth learning from.\n\n"
                + "\n".join(entries)
                + self._more(sdk, len(entries)))

    def _more(self, sdk, shown):
        """Say that the corpus is bigger than the list, when it is.

        A block of five with no total reads as "this is what you have", and an
        agent that believes it has five memories does not go looking for the
        sixty-two others. One count turns the block from an inventory into a
        sample.
        """
        total = self._corpus_size(sdk)
        if total <= shown:
            return ""
        return (f"\n\nShowing {shown} of {total}. `memory_recall list` shows "
                "them all, or search the folder with `hybrid_search`.")

    def _corpus_size(self, sdk):
        """How many entries exist at all. Two listings, no reads."""
        notes_dir, skills_dir = _entry_dirs(sdk)
        total = 0
        try:
            total += len([entry for entry
                          in sdk.fs.list(notes_dir, pattern="*.md", details=True) or []
                          if not entry.get("is_dir")])
        except sdk.Failed:
            pass
        try:
            total += len([entry for entry
                          in sdk.fs.list(skills_dir, details=True) or []
                          if entry.get("is_dir")])
        except sdk.Failed:
            pass
        return total

    def _description(self, sdk, path):
        """One entry's description, or "" for a file that is not an entry.

        The description and nothing else, because the only decision to make
        from the prompt is whether this past situation is the present one. What
        to do about it is what the agent opens the entry for.

        Injecting the body too was tried and had to come out. It cost tokens
        proportional to the corpus, it truncated long entries into advice with
        no context, and — worst — it destroyed the one observable signal in the
        system: with the content already in the prompt there is no reason to
        recall anything, and the recall is what tells the curator whether its
        job this time is to improve an existing entry or write a new one.
        """
        try:
            description = _frontmatter(
                sdk.fs.read(path)[:HEAD_CHARS]).get("description")
        except sdk.Failed:
            return ""
        if not description:
            return ""
        description = " ".join(description.split())
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[:MAX_DESCRIPTION_CHARS].rstrip() + "…"
        return description


_README = """# Memory

    notes/       one file per situation — searched
    skills/      one folder per procedure — searched
    MEMORY.md    facts, inlined into the agent's prompt in full
    (anything else lives here and is ignored)

`notes/` holds what to do, or not do, in a situation that has come up before.
`skills/` holds repeatable procedures, each in its own folder, following the
[agentskills.io](https://agentskills.io) layout — a `SKILL.md` plus optional
`references/`, `scripts/` and `assets/`.

Both are searched together and both are found the same way: by their
`description`, which is matched against whatever the user says in some future
conversation. Write it as the situation that should bring the entry back, never
as a topic label.

```
---
name: pdf-yields-no-text
description: A PDF yields no text, or extraction produces an empty document
updated: 2026-01-01
source: conversation 1
---

Do: Check the parser is installed before assuming the file is broken.
Because: Spent an hour on a corrupt-PDF theory when parser-pdf was not
installed.
```

An entry outside those two folders is never found, which is also what makes the
rest of this folder safe for drafts and scratch files.

An entry with no action in it cannot change what anyone does, so it is not an
entry. That is the test for whether something belongs here at all. Two things
deliberately do not: **facts** — names, paths, which machine something runs on,
a preference with no action attached — belong in MEMORY.md, which is inlined
into the prompt directly; and **records of what happened**, because this is not
a journal and an entry that does not change a future decision is noise that
makes the rest harder to find.

Entries are written and improved automatically after a conversation ends, but
they are ordinary markdown — edit or delete them freely.
"""
