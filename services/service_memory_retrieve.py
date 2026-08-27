"""Automatic memory retrieval — the half nobody has to ask for.

Every turn, search the memory folder for entries relevant to what the user just
said and put their names in the prompt.

One of three. This service ranks the corpus and puts names in the prompt,
``tool_memory`` is the only thing that touches the files in either direction,
and ``task_memory_curate`` decides when a curator runs.

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
``tool_memory`` records the other half.

The retrieval itself is one ``hybrid_search`` call at ``turn_start``, with the
user's own message as the query. That is deliberate on both counts. The hook
runs synchronously on the drive thread and sets the latency floor for every
reply, so there is no model call and no subagent here. And the user's words
*are* the retrieval cue — rewriting them into "better" search terms costs a
round trip to guess at something the person already said.

Writing is not this service's job and cannot be: only ``tool_memory``
writes entries, whether the caller is the agent mid-conversation or the curator
subagent ``task_memory_curate`` spawns after one ends.
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
#: retrieval. Must match the constants in ``tool_memory``.
MEMORY_DIRNAME = "memory"
NOTES_DIRNAME = "notes"
SKILLS_DIRNAME = "skills"

#: The agent profile a curator subagent runs under. Seeded by ``on_install``,
#: which is the one moment a plugin can write a kernel setting: attended, so
#: the write can be asked about, and caused by somebody deliberately installing
#: this. Must match ``CURATOR_PROFILE`` in ``task_memory_curate``.
CURATOR_PROFILE = "memory_curator"

#: What that profile may do: read widely, write in exactly one place.
#:
#: The absence of ``edit_file`` is the point and survives every addition here.
#: A curator writes through ``memory``, which addresses entries by name
#: and derives every path itself, so a background agent nobody is watching
#: cannot touch a file outside the memory folder however much it can *read*.
#: That asymmetry is what makes a broad read list safe to grant: reading
#: everything is only dangerous next to a way to send it somewhere, and egress
#: is gated separately.
#:
#: The search tools are listed rather than left to the dependency closure.
#: ``scoped_registry`` distinguishes *visible* from *callable*: closing over
#: ``hybrid_search``'s ``dependencies_tools`` made ``lexical_search`` and
#: ``semantic_search`` callable, but they never appeared in the curator's
#: catalogue, so it could not reach for one deliberately — a keyword search
#: for an exact phrase, or a semantic one for a situation it cannot name.
#:
#: ``sql_query`` is how the curator reads the conversation record itself
#: rather than the one transcript it was handed. Its writes are bounded by the
#: kernel rather than by this list: ``db.write``/``db.define`` refuse every
#: kernel table outright, so what remains reaches plugin-owned tables only —
#: ``memory_usage`` among them. A curator that corrupted its own bookkeeping
#: would be a nuisance, not a breach, and nothing here can reach conversations,
#: users or the ledger.
CURATOR_TOOLS = [
    "memory",                                  # its own one
    "read_file", "grep", "glob",               # the filesystem, read-only
    "hybrid_search", "lexical_search", "semantic_search",  # the index
    "sql_query",                               # the record
]

#: Tool names this suite used to ship, dropped from an existing profile on a
#: top-up. ``memory_recall`` and ``memory_curate`` were the read half and the
#: write half before they became the one ``memory`` tool above; a profile seeded
#: before that merge names both and would keep naming them forever, since
#: ``_top_up_curator_tools`` is otherwise purely additive.
RETIRED_CURATOR_TOOLS = ("memory_recall", "memory_curate")


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
    # ``config.write`` is here for ``on_install`` and nowhere else. That is not
    # a convention this file keeps on its honour: everywhere else in this
    # plugin the chain is a service's own, which is unattended, and an unsafe
    # write on an unattended chain is refused outright rather than asked.
    requests = ["paths.get", "config.read", "config.write",
                "tool.call", "fs.read", "fs.list", "fs.write",
                "db.define", "db.query", "db.write",
                "session.add_prompt_extra"]
    # The first three are what retrieval and the curator *call*. The last four
    # are what ``on_install`` writes into the ``memory_curator`` profile, and
    # naming a tool that is not installed grants nothing and says nothing — the
    # name is simply dropped, and the curator runs quietly narrower than the
    # profile claims. So this is the same relationship as the others: files
    # this plugin needs present to work as described.
    dependencies_files = ["tools/tool_hybrid_search.py",
                          "tools/tool_memory.py",
                          "tools/tool_read_file.py",
                          "tools/tool_grep.py",
                          "tools/tool_glob.py",
                          "tools/tool_sql_query.py"]
    dependencies_pip = []

    config_settings = [
        ("Memory pointers", "memory_max_pointers",
         "How many memory entries to surface at the start of each turn. 0 disables retrieval.",
         5, {"type": "slider", "range": (0, 15, 15), "is_float": False}),
    ]

    # Says only what this service is the authority on: which folders are
    # searched, and what the list it injects is. How to open an entry, and what
    # earns one, belong to the ``memory`` tool's own block — stating them here
    # too is how note-vs-skill came to be explained in three files. Nor does
    # this claim that anything reviews the conversation afterwards: that is
    # ``task_memory_curate``'s block, and if the task is uninstalled while this
    # service is not, a copy here would be a promise nothing keeps.
    agent_prompt = (
        "## Memory\n"
        "`memory/` in your workspace holds what you have learned, as `notes/` "
        "and `skills/`. Only those two folders are searched, so the rest of "
        "`memory/` is free for drafts and scratch files.\n"
    )

    def on_install(self, sdk):
        """Arrange the two kernel settings this suite cannot work without.

        ``sync_directories`` must contain the memory folder or nothing is ever
        indexed and retrieval stays empty forever; ``agent_profiles`` must hold
        ``memory_curator`` or ``task_memory_curate`` cannot spawn a confined
        curator and refuses to spawn an unconfined one.

        Both were attempted from ``start`` and then, when that failed, from the
        ``turn_start`` hook. Neither works, and neither should: a service has
        no session, so its chain is unattended and an unsafe write is refused
        rather than asked — and making the hook attended, which was tried and
        reverted, only moved the question to the moment *furthest* from
        anything the user deliberately did. Typing a message is consent to a
        reply, not to a config change. Installing this package is.

        Read-then-skip rather than write-then-hope, on both. This runs again on
        every update that changes this file, and a value the user has since
        edited is theirs.

        The two are attempted independently and neither aborts the other. Each
        is its own dialog, so letting the first refusal skip the second would
        make one "no" answer a question that was never asked — and the two
        settings fail in unrelated ways, one leaving retrieval empty and the
        other stopping the curator.
        """
        problems = [note for note in (self._seed_sync_directory(sdk),
                                      self._seed_curator_profile(sdk)) if note]
        if problems:
            raise RuntimeError("; ".join(problems))

    def _seed_sync_directory(self, sdk):
        """Put the memory folder on the sync list. Answers with what went wrong."""
        root = _memory_root(sdk)
        current = sdk.config.read("sync_directories") or []
        if root in current:
            return ""
        try:
            sdk.config.write("sync_directories", [*current, root])
        except sdk.Failed as error:
            return f"memory folder not synced ({error}) — nothing will be indexed"
        sdk.log(f"memory folder added to sync_directories: {root}")
        return ""

    def _seed_curator_profile(self, sdk):
        """Define the confined profile the curator runs under. Same shape."""
        profiles = sdk.config.read("agent_profiles") or {}
        if CURATOR_PROFILE in profiles:
            return self._top_up_curator_tools(sdk, profiles)
        try:
            sdk.config.write("agent_profiles", {**profiles, CURATOR_PROFILE: {
                "llm": "default",
                "prompt_suffix": "",
                "whitelist_or_blacklist_tools": "whitelist",
                "tools_list": list(CURATOR_TOOLS),
            }})
        except sdk.Failed as error:
            return f"{CURATOR_PROFILE} profile not created ({error}) — no curator can spawn"
        sdk.log(f"agent profile {CURATOR_PROFILE} created")
        return ""

    def _top_up_curator_tools(self, sdk, profiles):
        """Add tools a newer version needs to a profile that predates them.

        ``CURATOR_TOOLS`` grows as the suite learns what a curator needs, so an
        install that predates the growth holds a profile that is *correct for a
        version that is gone*. Doing nothing there was tried and is worse than
        it sounds: the curator silently does less than the prompt tells it to,
        the only symptom is work not happening, and updating the package — the
        one act that means "give me the new version" — changed nothing at all.

        Additive, and only from a package operation. Every other field is left
        exactly as the user set it, an unrecognised name they added stays, and
        this cannot run from a boot or a turn, so it is not something that
        fights an edit every morning — it happens when somebody installs or
        updates this package, which is when they asked for the new version.

        A blacklist profile is left alone entirely: nothing is being kept out,
        so there is nothing to top up, and rewriting it into a whitelist would
        be a narrowing dressed as a repair.

        Names this suite *retired* are the one exception to leaving a list
        alone, and the distinction is authorship: an unrecognised name the user
        added is theirs and stays, but one this package published and then
        stopped shipping is ours to clean up. Dead whitelist entries grant
        nothing — ``scoped_registry`` matches against what is actually
        registered — so this is tidiness rather than safety, and without it the
        two names the merge retired would sit in the user's ``/config``
        forever.
        """
        profile = profiles[CURATOR_PROFILE]
        if str(profile.get("whitelist_or_blacklist_tools") or "") != "whitelist":
            return ""
        listed = [name for name in (profile.get("tools_list") or [])
                  if name not in RETIRED_CURATOR_TOOLS]
        missing = [name for name in CURATOR_TOOLS if name not in listed]
        if not missing and listed == list(profile.get("tools_list") or []):
            return ""
        updated = {**profiles,
                   CURATOR_PROFILE: {**profile, "tools_list": listed + missing}}
        try:
            sdk.config.write("agent_profiles", updated)
        except sdk.Failed as error:
            return (f"{CURATOR_PROFILE} still missing {', '.join(missing)} "
                    f"({error}) — the curator will run without them")
        sdk.log(f"{CURATOR_PROFILE} gained {', '.join(missing)}" if missing
                else f"{CURATOR_PROFILE} dropped tools this suite retired")
        return ""

    def on_uninstall(self, sdk):
        """Drop the usage table. Leave the two settings alone.

        The asymmetry with ``on_install`` is deliberate. ``memory_usage`` is
        unambiguously this plugin's — nothing else writes it and nothing else
        can read anything out of it. A folder the user has been syncing for
        months and a profile they may have edited are theirs now, whoever put
        them there first, and an uninstall quietly narrowing what the machine
        indexes is a worse surprise than a leftover config line. Both are
        visible and removable in ``/config``; a dropped table is not
        recoverable at all.

        The notes and skills themselves are never touched. They are the user's
        writing, in the user's workspace.
        """
        try:
            sdk.db.define("DROP TABLE IF EXISTS memory_usage")
        except sdk.Failed as error:
            sdk.log(f"could not drop the memory usage table: {error}",
                    level="warning")

    def start(self, sdk):
        """Make the folder and the usage table. Nothing else, on purpose.

        The two kernel settings this needs are ``on_install``'s job, because a
        service's own chain is unattended and cannot be asked about one. If
        they are missing — an install predating that hook, or a declined
        dialog — both fail loudly rather than silently: an unindexed folder
        logs on the first search, and a missing profile stops the curator with
        a message naming it.
        """
        # Standing misconfigurations already reported. See ``_say_once``.
        self._said = set()
        self._ensure_folder(sdk, _memory_root(sdk))
        self._ensure_usage_table(sdk)

    def _ensure_usage_table(self, sdk):
        """The one table that records the life of a memory: offered, then taken.

        Defined here because this service is the first writer — it inserts a
        row per offer. ``tool_memory`` fills ``recalled_at`` and
        ``task_memory_curate`` only reads. One table rather than three answers
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

    # ── the hook ─────────────────────────────────────────────────────

    def on_turn_start(self, sdk, ctx, payload):
        """Search the memory folder and point the agent at what matched.

        Every failure path abstains rather than raising. ``hybrid_search`` is
        absent until the indexing packages are installed, an empty index is the
        normal state of a fresh install, and neither is a reason for a turn to
        fail.
        """
        try:
            limit = int(sdk.config.read("memory_max_pointers") or 0)
        except (sdk.Failed, TypeError, ValueError):
            limit = 5
        if limit <= 0:
            sdk.log("memory retrieval is disabled (memory_max_pointers is 0)",
                    level="debug")
            return None

        query = self._latest_user_message(sdk, ctx)
        if not query:
            # Ordinary at the very start of a conversation, and the one case
            # where an empty result says nothing about the corpus at all.
            sdk.log("memory: no user message to search on yet", level="debug")
            return None

        hits = self._search(sdk, query, limit)
        if not hits:
            return None  # _search said why

        offered = []
        block = self._render(sdk, hits, offered)
        if not block:
            # Entries matched but every one rendered empty, which means their
            # frontmatter has no description — invisible otherwise, since the
            # symptom is an absent block either way.
            self._say_once(
                sdk, "no-descriptions",
                f"memory: {len(hits)} entry(ies) matched but none could be "
                f"described — an entry needs a 'description:' in its "
                f"frontmatter to be offered.")
            return None
        try:
            # No ``key``: naming a session makes this "inject into *that*
            # session", which is unsafe unless the chain already names it —
            # and a hook's chain roots at ``service:memory_retrieve``, so it
            # never does. It was refused outright rather than asked, every
            # turn, in silence. Omitting it means "my own session", and the
            # kernel lends a doorway its session, so the handler lands where
            # this hook is actually standing.
            sdk.session.add_prompt(block, slot="memory")
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

        Half of a pair: ``tool_memory`` fills in ``recalled_at`` if the
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
            # ``author`` is the test, not ``role`` alone. The kernel writes
            # user-role rows the person never typed — a cancel notice, a
            # doorman's note, the ``reveal_user_commands`` note — so the newest
            # one after any ``/cancel`` was "[The user cancelled the previous
            # turn…]", and retrieval keyed off that instead of the question.
            rows = sdk.db.query(
                "SELECT content FROM conversation_messages"
                " WHERE conversation_id = ? AND LOWER(role) = 'user'"
                "   AND COALESCE(author, '') = ''"
                "   AND COALESCE(content, '') <> ''"
                " ORDER BY id DESC LIMIT 1", [int(cid)], max_rows=1)
        except sdk.Failed as error:
            # Abstain, but not in silence: a database this cannot read is a
            # different world from a conversation with nothing in it, and both
            # used to come back as the same empty string.
            sdk.log(f"memory: could not read conversation {cid}: {error}",
                    level="warning")
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

        Every outcome says something, because "no memories were offered" used
        to be reachable four different ways and looked identical from outside
        every time. The levels split on one question: did the search *run*.
        Nothing matching is the ordinary state of a fresh install and is
        ``debug``; a search that could not run, or one whose hits all fell
        outside the entry folders, is a misconfiguration nobody would otherwise
        find, and those are said out loud.
        """
        root = _memory_root(sdk)
        try:
            results = sdk.tools.call("hybrid_search", query=query,
                                     folder=root, max_results=limit * 3)
        except sdk.Failed as error:
            # Not necessarily a bug: hybrid_search ships with the indexing
            # packages, and without them there is nothing to search. Named
            # rather than shrugged at, because the two look the same from here.
            # Deliberately does not name *which* piece is missing: hybrid_search
            # may be absent, or present with both its retrievers down, and it
            # reports which itself. Repeating a guess over its answer would be
            # the wrong half of the message winning.
            self._say_once(
                sdk, "no-search",
                f"memory retrieval is off — the search it runs on failed: "
                f"{error}. Install the search packages, or set "
                f"memory_max_pointers to 0 to stop trying.")
            return []

        results = list(results or [])
        if not results:
            sdk.log(f"memory: nothing indexed under {root} matched "
                    f"{query[:60]!r}", level="debug")
            return []

        notes_dir, skills_dir = _entry_dirs(sdk)
        notes = sdk.path.normalize(notes_dir)
        skills = sdk.path.normalize(skills_dir)
        kept, seen, outside = [], set(), []
        for hit in results:
            path = str(hit.get("path") or "")
            if not path:
                continue
            normalized = sdk.path.normalize(path)
            if normalized.startswith(notes):
                entry = (sdk.path.stem(path), "note", path)
            elif normalized.startswith(skills):
                entry = self._skill_of(sdk, skills_dir, path)
            else:
                outside.append(path)  # MEMORY.md, the README, a draft
                continue
            if entry is None or entry[0] in seen:
                continue
            seen.add(entry[0])
            kept.append(entry)
            if len(kept) >= limit:
                break

        if not kept:
            # The failure worth naming. The folder is indexed and matching, so
            # everything upstream is working — the entries are simply not in
            # notes/ or skills/, which is the one thing this service cannot
            # infer and the user cannot see. A sample, because the whole list
            # is every draft in the folder.
            self._say_once(
                sdk, "no-entries",
                f"memory: {len(results)} indexed file(s) under {root} matched, "
                f"but none are entries — an entry lives in {NOTES_DIRNAME}/ or "
                f"{SKILLS_DIRNAME}/. Matched instead: "
                + ", ".join(outside[:3]) + ("…" if len(outside) > 3 else ""))
            return []

        sdk.log(f"memory: {len(kept)} of {len(results)} matches are entries",
                level="debug")
        return kept

    def _say_once(self, sdk, topic, message):
        """Log a standing misconfiguration the first time only.

        This runs at the top of every turn, so a warning that repeats is a
        warning that gets filtered out — and the conditions here are *states*
        rather than events: nothing is installed, nothing is in the entry
        folders. Both stay true until somebody fixes them, and neither is worth
        saying twice. Reset per process, so a restart says it again, which is
        the natural moment for a person to be reading the log anyway.

        Self-initialising rather than trusting ``start`` to have run first: a
        diagnostic that raises ``AttributeError`` on the path it exists to
        explain would be the worst possible version of this.
        """
        said = getattr(self, "_said", None)
        if said is None:
            said = self._said = set()
        if topic in said:
            sdk.log(message, level="debug")
            return
        said.add(topic)
        sdk.log(message, level="warning")

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
                "`memory read` the name when one looks close enough to your "
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
        return (f"\n\nShowing {shown} of {total}. `memory list` shows "
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
