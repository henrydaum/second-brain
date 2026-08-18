"""Read, list and write memory entries.

**The only thing in the system that touches the memory folder, in either
direction.** ``service_memory_retrieve`` ranks the corpus at the start of every
turn and puts *names and descriptions* in the prompt; this is what turns one of
those names into the thing itself, and what writes the thing. The main agent
uses it mid-conversation when it learns something worth keeping; the curator
subagent that ``task_memory_curate`` spawns after a conversation ends uses the
same tool for the same job. One tool means one format and one set of rules
about what belongs where.

**It takes a name, never a path.** The path is derived from the name after the
name is checked against the agentskills.io character set, so there is no
argument that reaches outside ``workspace/memory`` and none that names
``MEMORY.md`` — which is the agent's own file of facts, inlined into every
prompt by the kernel, and not something a background subagent should be able to
rewrite. That confinement is why the curator can be given this tool and not
``edit_file``.

Reading and writing being one tool does not widen any of that, because there is
exactly one place a path is derived and it is not from an argument: every
``sdk.fs`` call takes a value from ``_paths_for``, which produces two paths per
legal name and nothing else. This was once two tools whose *declarations* were
disjoint — the writer had no ``fs.read`` — and that bought nothing it could not
get one tool over, since the curator also holds ``read_file`` and ``grep``. The
character set is the boundary; the split was only bookkeeping, and it cost a
revision the ability to keep a description it had just read.

**The tool writes the frontmatter.** The model supplies a name, a description
and a body; ``updated`` and ``source`` are stamped here. A model asked to
produce its own ``---`` fences gets them subtly wrong often enough to matter,
and a broken block makes an entry unrankable with no symptom at all.

Two kinds, one shape. A **note** is one situation and what to do about it, at
``notes/<name>.md``. A **skill** is a repeatable procedure that may carry
``references/``, ``scripts/`` and ``assets/`` beside it, at
``skills/<name>/SKILL.md``. Both carry ``name`` and ``description``, which is
what lets retrieval rank and render them identically — the distinction is for
the reader, not for the search.

**Calling ``read`` is the used-signal**, and it is the reason ``db.write`` is
here. Whether a memory helped is only knowable as a pair — it was offered, and
then it was opened — and neither half is available alone. The offer lives in the
system prompt, which is stored nowhere, so the service records it. The open used
to be reconstructed by parsing every assistant message for a ``read_file`` call
and normalizing the path it named, which is a lot of machinery to infer
something the agent could simply have told us. It tells us now: one row, written
here, at the moment it happens. Recording an *unprompted* recall matters as much
as filling in a prompted one — an entry the agent went looking for without being
shown it is being used, and the curator should revise it on the same evidence.
"""

dependencies_files = []
dependencies_pip = []
requests = ["paths.get", "fs.read", "fs.list", "fs.write", "fs.delete",
            "session.get", "session.push", "db.query", "db.write",
            "config.read"]

import time

from guest.bases import BaseTool

#: The two folders that hold entries, and the only two that are searched,
#: read or written. Everything else under ``memory/`` — ``MEMORY.md``, the
#: README, drafts, whatever the agent leaves lying around — is deliberately
#: unreachable from here. Must match the constants in
#: ``service_memory_retrieve``; the two are pinned equal by
#: ``tests/test_store_memory_bundle.py``. The folder is what makes an entry an
#: entry — a fact a writer cannot get subtly wrong, unlike a field it has to
#: remember to include.
MEMORY_DIRNAME = "memory"
NOTES_DIRNAME = "notes"
SKILLS_DIRNAME = "skills"

#: A skill is a folder; these are the subfolders the spec gives it. Named
#: rather than listed from disk so the answer is stable and ordered.
SKILL_RESOURCE_DIRS = ("references", "scripts", "assets")

#: Every action but ``list`` addresses one entry, so the name is checked once,
#: before dispatch, rather than four times inside it. Module-level so it sits
#: with the other constants and cannot be mistaken for a declaration the
#: validator reads.
NAMED_ACTIONS = ("read", "create", "update", "delete")

#: Cap on one entry. A skill is meant to stay under ~5,000 tokens by its own
#: spec, so anything near this is a sign the entry should have been split.
MAX_READ_CHARS = 20_000

#: How much of a file to read when only its frontmatter is wanted.
HEAD_CHARS = 2000

#: The spec's own ceiling on a description, and the reason to have one here is
#: that this field goes in every prompt where the entry ranks.
MAX_DESCRIPTION = 1024

#: What each action is called in that note. Past tense, because the note is
#: about something that already happened. Reads are absent on purpose: they are
#: not changes, and a read that announced itself would interrupt the user every
#: time the agent opened a memory.
NOTIFY_VERB = {"create": "created", "update": "updated", "delete": "deleted"}


def _memory_root(sdk):
    """The folder this tool may touch, and the only one."""
    return sdk.path.join(sdk.paths.get("workspace"), MEMORY_DIRNAME)


def _valid_name(name):
    """Whether this is a legal entry name, per the agentskills.io spec.

    1-64 characters, lowercase alphanumerics and hyphens, no leading, trailing
    or doubled hyphen. This is a security check rather than tidiness, and it
    guards the writer as much as the reader: the name becomes a path component,
    and the character set is what makes ``..``, a separator, a drive letter and
    ``MEMORY.md`` all impossible to express. Checked with plain string
    operations rather than a regex because the rule is short and the failure is
    worth being exact about.
    """
    if not name or len(name) > 64:
        return False
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False
    return all(char.isdigit() or ("a" <= char <= "z") or char == "-"
               for char in name)


def _note_path(sdk, name):
    """Where a note by this name lives."""
    return sdk.path.join(_memory_root(sdk), NOTES_DIRNAME, name + ".md")


def _skill_path(sdk, name):
    """Where a skill by this name keeps its instructions."""
    return sdk.path.join(_memory_root(sdk), SKILLS_DIRNAME, name, "SKILL.md")


def _paths_for(sdk, name):
    """The note path and the skill path a name could mean, in that order.

    The only two paths this tool can produce, for reading or for writing, which
    is what makes ``MEMORY.md`` unreachable even before the character set gets
    to it. Everything that resolves an entry comes through here, so the write
    side cannot grow a third template without this function growing one.
    """
    return [(_note_path(sdk, name), "note"),
            (_skill_path(sdk, name), "skill")]


def _exists(sdk, path):
    """Whether a file is there. ``fs.list`` answers for a file too."""
    try:
        sdk.fs.list(path)
        return True
    except sdk.Failed:
        return False


def _existing(sdk, name):
    """``(path, kind)`` for an entry that is already there, or ``(None, "")``."""
    for path, kind in _paths_for(sdk, name):
        if _exists(sdk, path):
            return path, kind
    return None, ""


def _frontmatter(text):
    """The leading ``---`` block as a dict, tolerating anything.

    Deliberately not a YAML parser. Entries are written by language models and
    the failure that matters is a malformed block taking a turn down, so
    unparseable lines are skipped and a file with no block at all still reads.
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

    ``strip("'\\"")`` eats a trailing quote whether or not anything opened one,
    so a description ending in a quoted word silently lost its last character.
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _document(name, description, body, source):
    """One entry, frontmatter and all.

    ``updated`` is stamped rather than accepted, because a model asked for
    today's date will confidently supply the date it was trained. ``source``
    records the conversation the lesson came from, which is the only way back
    to the evidence once the transcript has scrolled away.
    """
    lines = ["---", f"name: {name}", f"description: {description}",
             f"updated: {time.strftime('%Y-%m-%d')}"]
    if source:
        lines.append(f"source: conversation {source}")
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.strip() + "\n"


class Memory(BaseTool):
    """Read, list and write memory notes and skills."""

    name = "memory"
    config_settings = [
        ("Announce memory changes", "notify_on_memory_change",
         "Post a line to your chat whenever a background agent writes to "
         "memory. Only fires for work you are not watching — a change you "
         "asked for in conversation is already visible in the reply.",
         True, {"type": "bool"}),
    ]
    description = (
        "Read, list and write your memory — notes (one situation and what to "
        "do about it) and skills (repeatable procedures). Names come from the "
        "memory block in your prompt, which gives descriptions only; read an "
        "entry before acting on it. Entries outlive the conversation and are "
        "searched at the start of every future turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "list", "create", "update", "delete"],
                "description": (
                    "read: return one entry's full text. "
                    "list: every entry's name and description. "
                    "create: write a new entry. "
                    "update: replace an entry's body, and its description if "
                    "you give one. delete: remove it."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "The entry's name, as shown in your memory block — e.g. "
                    "'retry-failed-uploads'. Lowercase letters, digits and "
                    "hyphens. Name it for the situation, not for this "
                    "conversation. Not a path. Required for everything but "
                    "list."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["note", "skill"],
                "description": (
                    "create only. note (default): one situation and its "
                    "lesson. skill: a repeatable procedure, which gets its "
                    "own folder for references and scripts."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "One or two sentences: what this is for and when it "
                    "applies. This is matched against what a user says in "
                    "some future conversation and is all that is shown before "
                    "the entry is opened, so write the situation — 'a PDF "
                    "yields no text' — not a topic label. Required to create; "
                    "on update, omit it to keep the one already there."
                ),
            },
            "body": {
                "type": "string",
                "description": (
                    "The entry itself, in markdown. For a note: what to do or "
                    "avoid, and what actually happened when it was tried. For "
                    "a skill: the procedure. Do not write frontmatter — it is "
                    "added for you."
                ),
            },
            "narration": {
                "type": "string",
                "description": (
                    "A few words on what you are looking up or recording and "
                    "why, shown to the user beside the call. E.g. 'checking "
                    "what happened last time an upload failed'."
                ),
            },
        },
        "required": ["action"],
    }
    requires_services = []

    agent_prompt = (
        "## The memory tool\n"
        "`memory` reads and writes the folder: `read` one entry by name, "
        "`list` everything, `create`, `update`, `delete`. The block above gives "
        "descriptions only, and a description is a map — `memory read` an entry "
        "before acting on it. When something feels familiar mid-turn and was "
        "not listed, `memory list` is cheap.\n"
        "Write when you learn something that will change what you do next "
        "time: a trap, a correction the user made, something that broke, or "
        "the words this user uses for things — their own vocabulary, "
        "preferences and recurring phrasings are worth recording, so they do "
        "not have to keep explaining them. If you cannot name what it would "
        "change, there is nothing to write. `list` first and `update` an entry "
        "that already covers the situation rather than adding a second.\n"
        "A **note** is one situation and its lesson. A **skill** is a "
        "repeatable procedure worth its own folder, for `references/` and "
        "`scripts/` you add with your file tools afterwards. Nothing here "
        "needs approval, and nothing here can touch MEMORY.md."
    )

    def run(self, sdk, **kwargs):
        """Read, list, create, update or delete one memory entry."""
        action = (kwargs.get("action") or "read").strip().lower()
        if action == "list":
            return self._list(sdk)
        if action not in NAMED_ACTIONS:
            return sdk.fail(f"Unknown action: {action!r}. Use read, list, "
                            "create, update or delete.")

        name = (kwargs.get("name") or "").strip().lower()
        if not name:
            return sdk.fail(f"{action!r} needs a name. Use action='list' to "
                            "see what is there.")
        if not _valid_name(name):
            return sdk.fail(
                f"{name!r} is not a usable entry name. Use 1-64 lowercase "
                "letters, digits and hyphens, with no leading, trailing or "
                "doubled hyphen — and a name, never a path.")

        if action == "read":
            return self._read(sdk, name)
        if action == "create":
            return self._create(sdk, name, kwargs)
        if action == "update":
            return self._update(sdk, name, kwargs)
        return self._delete(sdk, name)

    # ── reading ──────────────────────────────────────────────────────

    def _read(self, sdk, name):
        """One entry's full text, and the fact that it was opened.

        Tries the read rather than probing with ``_existing`` first: the two
        candidates are the answer either way, and a probe would cost an extra
        Request on every recall.
        """
        for path, kind in _paths_for(sdk, name):
            try:
                text = sdk.fs.read(path)
            except sdk.Failed:
                continue
            self._record(sdk, name)
            return self._rendered(sdk, name, kind, text)

        known = self._names(sdk)
        near = [other for other in known if name in other or other in name]
        suggestion = ", ".join(near[:5] or known[:8]) or "(none yet)"
        return sdk.fail(f"No memory entry named {name!r}. Did you mean: {suggestion}")

    def _rendered(self, sdk, name, kind, text):
        """One entry, plus what else is in its folder when it is a skill."""
        if len(text) > MAX_READ_CHARS:
            text = text[:MAX_READ_CHARS] + "\n\n... (truncated)"
        if kind == "skill":
            if resources := self._resources(sdk, name):
                text += ("\n\n---\nFiles bundled with this skill, to read with "
                         "`read_file` when you need them:\n"
                         + "\n".join(f"- {item}" for item in resources))
        return sdk.ok(None, llm_summary=text)

    def _resources(self, sdk, name):
        """A skill's bundled files, as paths the agent can hand to read_file.

        This is progressive disclosure done by the tool rather than by prose:
        the skill body says *when* to open a reference, and this says where it
        is, so the agent never has to guess at a path or list a directory.
        """
        folder = sdk.path.join(_memory_root(sdk), SKILLS_DIRNAME, name)
        found = []
        for sub in SKILL_RESOURCE_DIRS:
            directory = sdk.path.join(folder, sub)
            try:
                entries = sdk.fs.list(directory, details=True)
            except sdk.Failed:
                continue
            for entry in entries or []:
                if not entry.get("is_dir"):
                    found.append(sdk.path.join(directory, entry["name"]))
        return sorted(found)

    # ── listing ──────────────────────────────────────────────────────

    def _list(self, sdk):
        """Every entry, both kinds together, as name and description."""
        entries = self._entries(sdk)
        if not entries:
            return sdk.ok([], llm_summary="No memory entries yet.")
        lines = [f"- {row['name']}{' (skill)' if row['kind'] == 'skill' else ''}"
                 f" — {row['description'] or '(no description)'}"
                 for row in entries]
        return sdk.ok(entries,
                      llm_summary=f"{len(entries)} memory entries:\n"
                                  + "\n".join(lines))

    def _entries(self, sdk):
        """Name, kind and description for everything in the corpus."""
        root = _memory_root(sdk)
        found = []
        try:
            notes = sdk.fs.list(sdk.path.join(root, NOTES_DIRNAME),
                                pattern="*.md", details=True) or []
        except sdk.Failed:
            notes = []
        for entry in notes:
            if entry.get("is_dir"):
                continue
            name = sdk.path.stem(entry["name"])
            found.append({"name": name, "kind": "note",
                          "description": self._stored_description(
                              sdk, sdk.path.join(root, NOTES_DIRNAME,
                                                 entry["name"]))})
        try:
            skills = sdk.fs.list(sdk.path.join(root, SKILLS_DIRNAME),
                                 details=True) or []
        except sdk.Failed:
            skills = []
        for entry in skills:
            if not entry.get("is_dir"):
                continue
            name = entry["name"]
            found.append({"name": name, "kind": "skill",
                          "description": self._stored_description(
                              sdk, sdk.path.join(root, SKILLS_DIRNAME, name,
                                                 "SKILL.md"))})
        return sorted(found, key=lambda row: row["name"])

    def _stored_description(self, sdk, path):
        """One entry's description as it stands on disk, or "" for none.

        Named for the direction it travels, against ``_supplied_description``
        below. Two functions called ``_description`` in one file is how a
        model-supplied string ends up somewhere only a stored one belongs.
        """
        try:
            return _frontmatter(sdk.fs.read(path)[:HEAD_CHARS]).get(
                "description", "")
        except sdk.Failed:
            return ""

    def _names(self, sdk):
        """Just the names, for a failed lookup to suggest from."""
        return [row["name"] for row in self._entries(sdk)]

    # ── writing ──────────────────────────────────────────────────────

    def _create(self, sdk, name, kwargs):
        """Write a new entry, refusing to shadow one that exists."""
        existing, kind = _existing(sdk, name)
        if existing:
            return sdk.fail(
                f"A {kind} named {name!r} already exists. Read it with "
                "action='read' and update it — two entries about one "
                "situation is how a corpus stops being useful.")

        description = self._supplied_description(kwargs)
        body = (kwargs.get("body") or "").strip()
        if not description:
            return sdk.fail(
                "'create' needs a description: what this is for and when it "
                "applies. It is what retrieval matches on.")
        if not body:
            return sdk.fail("'create' needs a body.")

        kind = (kwargs.get("kind") or "note").strip().lower()
        if kind not in ("note", "skill"):
            return sdk.fail(f"Unknown kind: {kind!r}. Use note or skill.")
        path = _skill_path(sdk, name) if kind == "skill" else _note_path(sdk, name)

        sdk.fs.write(path, _document(name, description, body,
                                     self._conversation(sdk)))
        self._notify(sdk, "create", name)
        where = (" Add references/ and scripts/ beside it with your file tools."
                 if kind == "skill" else "")
        return sdk.ok(None, llm_summary=f"Wrote {kind} '{name}'.{where}")

    def _update(self, sdk, name, kwargs):
        """Replace an entry's body, keeping its description unless given one.

        A full replacement of the *body* rather than a patch, because the caller
        read the entry before deciding to change it — and a partial edit against
        a file the model has not seen is how a corpus acquires contradictions.

        The description is the exception, and it is why this is the one write
        that reads. Sharpening the field the entry is retrieved by is usually
        the most valuable part of a revision, so supplying one replaces it; but
        a revision that only fixes the body should not have to restate a
        sentence it just read, and requiring it was a workaround for this tool
        once being unable to read at all. Omitting it keeps what is on disk.

        An entry with no recoverable description is still refused: it would
        rank and never once be offered, which is indistinguishable from having
        no memory of the situation at all.
        """
        path, kind = _existing(sdk, name)
        if not path:
            return sdk.fail(
                f"No memory entry named {name!r} to update. Use "
                "action='list' to see what is there, or create it.")

        body = (kwargs.get("body") or "").strip()
        if not body:
            return sdk.fail("'update' needs a body — it replaces the old one.")
        description = (self._supplied_description(kwargs)
                       or self._stored_description(sdk, path))
        if not description:
            return sdk.fail(
                "'update' needs a description: this entry has none on disk to "
                "keep. Write the situation it should fire on — that is what "
                "retrieval matches, and an entry without one is never offered.")

        sdk.fs.write(path, _document(name, description, body,
                                     self._conversation(sdk)))
        self._notify(sdk, "update", name)
        return sdk.ok(None, llm_summary=f"Updated {kind} '{name}'.")

    def _delete(self, sdk, name):
        """Remove an entry, and a skill's whole folder with it."""
        path, kind = _existing(sdk, name)
        if not path:
            return sdk.fail(f"No memory entry named {name!r} to delete.")
        target = (sdk.path.join(_memory_root(sdk), SKILLS_DIRNAME, name)
                  if kind == "skill" else path)
        sdk.fs.delete(target)
        self._notify(sdk, "delete", name)
        return sdk.ok(None, llm_summary=f"Deleted {kind} '{name}'.")

    # ── helpers ──────────────────────────────────────────────────────

    def _supplied_description(self, kwargs):
        """The description the model handed us, collapsed and capped.

        Newlines are flattened because this is rendered as a single bullet in
        every prompt where the entry ranks, and a description that breaks the
        line breaks the block.
        """
        raw = " ".join((kwargs.get("description") or "").split())
        return raw[:MAX_DESCRIPTION]

    def _conversation(self, sdk):
        """Which conversation this is happening in.

        The recall's, for the usage row; and the lesson's, for ``source``.
        """
        try:
            return (sdk.session.get() or {}).get("conversation_id")
        except sdk.Failed:
            return None

    # ── the used-signal ──────────────────────────────────────────────

    def _record(self, sdk, name):
        """Mark this entry recalled, in the conversation that recalled it.

        Fills the pending offer row when the service surfaced this entry, and
        inserts a bare row when it did not — an entry the agent went looking
        for on its own is being used just as much as one it was handed, and
        the curator revises both on the same evidence.

        Entirely best-effort. The table belongs to ``service_memory_retrieve``,
        and a half-installed suite must degrade to a memory that still reads
        and merely stops learning which entries earn their place.
        """
        cid = self._conversation(sdk)
        if not cid:
            return
        now = time.time()
        try:
            pending = sdk.db.query(
                "SELECT id FROM memory_usage"
                " WHERE conversation_id = ? AND name = ? AND recalled_at IS NULL"
                " ORDER BY id DESC LIMIT 1", [int(cid), name], max_rows=1)
        except sdk.Failed as error:
            sdk.log(f"memory usage is not being recorded: {error}", level="debug")
            return
        try:
            if pending:
                sdk.db.write("UPDATE memory_usage SET recalled_at = ? WHERE id = ?",
                             [now, int(pending[0]["id"])])
            else:
                sdk.db.write(
                    "INSERT INTO memory_usage"
                    " (name, conversation_id, offered_at, recalled_at)"
                    " VALUES (?, ?, NULL, ?)", [name, int(cid), now])
        except sdk.Failed as error:
            sdk.log(f"could not record a memory recall: {error}", level="debug")

    # ── telling the user something happened ──────────────────────────

    def _notify(self, sdk, action, name):
        """Announce one write, when it happened somewhere nobody could see it.

        Called from the three writes and from neither read. A read that
        announced itself would put a line in the user's chat every time the
        agent opened a memory, which is the noise that would make the setting
        below get turned off.

        The curator subagent writes to memory after a conversation ends, on a
        session with no person attached to it. That is the whole point of it —
        but it means the corpus the agent is retrieved *from* can change with
        no trace anywhere the user looks, and a memory system quietly editing
        itself is exactly the thing to be out of the loop about.

        **``attended`` is the condition, not "is this a subagent".** The kernel
        already owns that question (``runtime.is_attended``), and it is the
        right one: a change the user asked for mid-conversation is already
        visible in the reply, and announcing it again would be the tool talking
        over itself. Anything else — a curator, a scheduled job, a background
        drive — is invisible by construction. Asking the kernel also means this
        stays correct for a concurrent multi-user frontend, which owns its own
        attendance and would defeat any guess made from a session key.

        **A notification rather than a chat message**, which is what it always
        wanted to be. This used to emit ``chat_message_pushed`` by literal
        channel name — reaching around ``session.push`` to a bus channel this
        tool does not own, in order to get the one thing that channel offered
        and the Request did not: a payload with a ``source`` on it. Nothing
        read that field, so the note arrived as an ordinary line of chat and
        the attribution was decoration.

        ``notify=True`` is the supported spelling of the same intent, and it is
        better in the way that matters: the kernel stamps ``source`` off the
        provenance chain, so it says ``tool_memory`` because that is what
        actually ran, not because this file claimed it. A frontend with a
        notification area puts it there; one without shows it in the chat
        exactly as before, so nothing regresses for the REPL or Telegram.

        No ``session_key``, deliberately, and that is unchanged: the write
        happened where nobody was watching, so there is no session to reply
        *to* and the note goes to whatever surface the user is actually at.

        Failing closed on both readings, and the direction matters in opposite
        ways. An unreadable ``attended`` means *do not send*, since a spurious
        notification in the middle of somebody's conversation is worse than a
        missing one. An unreadable setting means *do send*, because the default
        is on and the user has not said otherwise.

        Never raises. Announcing a write must not be able to fail the write —
        the entry is already on disk by the time this runs, so a failure here
        would report an error for something that fully succeeded.
        """
        try:
            if not self._unattended(sdk) or not self._announcing(sdk):
                return
            verb = NOTIFY_VERB.get(action, action)
            sdk.session.push(name, title=f"Memory {verb}", notify=True,
                             level="success")
        except Exception:                      # noqa: BLE001 - see docstring
            sdk.log(f"could not announce the memory {action} of {name!r}",
                    level="debug")

    def _unattended(self, sdk):
        """Whether this write is happening where nobody can see it."""
        try:
            attended = (sdk.session.get() or {}).get("attended")
        except sdk.Failed:
            return False
        return attended is False

    def _announcing(self, sdk):
        """Whether the user wants to hear about it. Default on."""
        try:
            return bool(sdk.config.read("notify_on_memory_change") is not False)
        except sdk.Failed:
            return True
