"""Open one memory entry by name.

The agent-invoked half of retrieval. ``service_memory`` ranks the corpus at the
start of every turn and puts *names and descriptions* in the prompt; this is
what turns one of those names into the thing itself.

**It takes a name, never a path.** The path is derived from the name after the
name is checked against the agentskills.io character set, so there is no
argument that reaches outside ``workspace/memory`` and none that names
``MEMORY.md``. That is what makes this safe to hand to a subagent, and it is
also simply easier for the model: a name is what the prompt showed it.

**Calling this is the used-signal, and that is the reason it exists as its own
tool.** Whether a memory helped is only knowable as a pair — it was offered,
and then it was opened — and neither half is available alone. The offer lives
in the system prompt, which is stored nowhere, so the service records it. The
open used to be reconstructed by parsing every assistant message for a
``read_file`` call and normalizing the path it named, which is a lot of
machinery to infer something the agent could simply have told us. It tells us
now: one row, written here, at the moment it happens.

Recording an unprompted recall matters as much as filling in a prompted one.
An entry the agent went looking for without being shown it is being used, and
the curator should revise it on the same evidence.
"""

dependencies_files = []
dependencies_pip = []
requests = ["paths.get", "fs.read", "fs.list", "session.get",
            "db.query", "db.write"]

import time

from guest.bases import BaseTool

#: The two folders that hold entries, and the only two that are searched or
#: read. Everything else under ``memory/`` — ``MEMORY.md``, the README, drafts,
#: whatever the agent leaves lying around — is deliberately unreachable from
#: here. Must match the constants in ``service_memory`` and
#: ``tool_memory_curate``; the three are pinned equal by
#: ``tests/test_store_memory_bundle.py``.
MEMORY_DIRNAME = "memory"
NOTES_DIRNAME = "notes"
SKILLS_DIRNAME = "skills"

#: A skill is a folder; these are the subfolders the spec gives it. Named
#: rather than listed from disk so the answer is stable and ordered.
SKILL_RESOURCE_DIRS = ("references", "scripts", "assets")

#: Cap on one entry. A skill is meant to stay under ~5,000 tokens by its own
#: spec, so anything near this is a sign the entry should have been split.
MAX_READ_CHARS = 20_000

#: How much of a file to read when only its frontmatter is wanted.
HEAD_CHARS = 2000


def _memory_root(sdk):
    """The folder this tool may touch, and the only one."""
    return sdk.path.join(sdk.paths.get("workspace"), MEMORY_DIRNAME)


def _valid_name(name):
    """Whether this is a legal entry name, per the agentskills.io spec.

    1-64 characters, lowercase alphanumerics and hyphens, no leading, trailing
    or doubled hyphen. Checked with plain string operations rather than a
    regex because the rule is short and the failure is worth being exact about:
    this is what stops a name being a path. ``..``, a separator and a drive
    letter all fail on the character set alone.
    """
    if not name or len(name) > 64:
        return False
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False
    return all(char.isdigit() or ("a" <= char <= "z") or char == "-"
               for char in name)


def _paths_for(sdk, name):
    """The note path and the skill path a name could mean, in that order."""
    root = _memory_root(sdk)
    return [
        (sdk.path.join(root, NOTES_DIRNAME, name + ".md"), "note"),
        (sdk.path.join(root, SKILLS_DIRNAME, name, "SKILL.md"), "skill"),
    ]


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


class MemoryRecall(BaseTool):
    """Read a memory note or skill by name."""

    name = "memory_recall"
    description = (
        "Open one memory entry — a note or a skill — by its name, or list "
        "every entry you have. Names come from the memory block in your "
        "prompt. The block gives you descriptions only; read the entry before "
        "acting on it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "list"],
                "description": (
                    "read: return one entry's full text. "
                    "list: every entry's name and description."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "The entry's name, as shown in your memory block — e.g. "
                    "'retry-failed-uploads'. Not a path. Required for read."
                ),
            },
            "narration": {
                "type": "string",
                "description": (
                    "A few words on what you are looking up and why, shown to "
                    "the user beside the call. E.g. 'checking what happened "
                    "last time an upload failed'."
                ),
            },
        },
        "required": ["action"],
    }
    requires_services = []

    agent_prompt = (
        "## Recalling memory\n"
        "Entries matching the current message are listed above under 'Things "
        "you have done before', as a name and a description. The description "
        "is a map, not the content — `memory_recall` the name when one looks "
        "close enough to your situation to be worth learning from, and do not "
        "answer from a description alone.\n"
        "A **note** is one situation and what to do about it. A **skill** is a "
        "repeatable procedure, and reading one also lists its `references/` "
        "files — open those with `read_file` only when you get to the part "
        "that needs them.\n"
        "`memory_recall list` shows everything you have. When something comes "
        "up mid-turn that feels familiar and was not in the block, that or a "
        "`hybrid_search` scoped to the memory folder will find it — both are "
        "cheap, and usually cheaper than working it out again."
    )

    def run(self, sdk, **kwargs):
        """Read an entry, or list them all."""
        action = (kwargs.get("action") or "read").strip().lower()
        if action == "list":
            return self._list(sdk)
        if action != "read":
            return sdk.fail(f"Unknown action: {action!r}. Use read or list.")

        name = (kwargs.get("name") or "").strip().lower()
        if not name:
            return sdk.fail("'read' needs a name. Use action='list' to see them.")
        if not _valid_name(name):
            return sdk.fail(
                f"{name!r} is not a memory entry name. Names are lowercase "
                "letters, digits and hyphens — and a name, never a path.")

        for path, kind in _paths_for(sdk, name):
            try:
                text = sdk.fs.read(path)
            except sdk.Failed:
                continue
            self._record(sdk, name)
            return self._rendered(sdk, name, kind, path, text)

        known = self._names(sdk)
        near = [other for other in known if name in other or other in name]
        suggestion = ", ".join(near[:5] or known[:8]) or "(none yet)"
        return sdk.fail(f"No memory entry named {name!r}. Did you mean: {suggestion}")

    # ── reading ──────────────────────────────────────────────────────

    def _rendered(self, sdk, name, kind, path, text):
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
                          "description": self._description(
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
                          "description": self._description(
                              sdk, sdk.path.join(root, SKILLS_DIRNAME, name,
                                                 "SKILL.md"))})
        return sorted(found, key=lambda row: row["name"])

    def _description(self, sdk, path):
        """One entry's description, or "" for a file that has none."""
        try:
            return _frontmatter(sdk.fs.read(path)[:HEAD_CHARS]).get(
                "description", "")
        except sdk.Failed:
            return ""

    def _names(self, sdk):
        """Just the names, for a failed lookup to suggest from."""
        return [row["name"] for row in self._entries(sdk)]

    # ── the used-signal ──────────────────────────────────────────────

    def _record(self, sdk, name):
        """Mark this entry recalled, in the conversation that recalled it.

        Fills the pending offer row when the service surfaced this entry, and
        inserts a bare row when it did not — an entry the agent went looking
        for on its own is being used just as much as one it was handed, and
        the curator revises both on the same evidence.

        Entirely best-effort. The table belongs to ``service_memory``, and a
        half-installed suite must degrade to a memory that still reads and
        merely stops learning which entries earn their place.
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

    def _conversation(self, sdk):
        """Which conversation is doing the recalling."""
        try:
            return (sdk.session.get() or {}).get("conversation_id")
        except sdk.Failed:
            return None
