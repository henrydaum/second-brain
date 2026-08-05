"""Create, update and delete memory entries.

The agent-invoked half of curation, and **the only thing in the system that
writes to the memory folder.** The main agent uses it mid-conversation when it
learns something worth keeping; the curator subagent that
``task_memory_reflect`` spawns after a conversation ends uses the same tool for
the same job. One writer means one format and one set of rules about what
belongs where.

**It takes a name, never a path**, and the path is derived from the name after
the name is checked against the agentskills.io character set. There is no
argument that reaches outside ``workspace/memory``, and none that names
``MEMORY.md`` — which is the agent's own file of facts, inlined into every
prompt by the kernel, and not something a background subagent should be able to
rewrite. That confinement is why the curator can be given this tool and not
``edit_file``.

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
"""

dependencies_files = []
dependencies_pip = []
requests = ["paths.get", "fs.list", "fs.write", "fs.delete", "session.get"]

import time

from guest.bases import BaseTool

#: Must match the constants in ``service_memory`` and ``tool_memory_recall``;
#: the three are pinned equal by ``tests/test_store_memory_bundle.py``. The
#: folder is what makes an entry an entry — a fact a writer cannot get subtly
#: wrong, unlike a field it has to remember to include.
MEMORY_DIRNAME = "memory"
NOTES_DIRNAME = "notes"
SKILLS_DIRNAME = "skills"

#: The spec's own ceiling on a description, and the reason to have one here is
#: that this field goes in every prompt where the entry ranks.
MAX_DESCRIPTION = 1024


def _memory_root(sdk):
    """The folder this tool may touch, and the only one."""
    return sdk.path.join(sdk.paths.get("workspace"), MEMORY_DIRNAME)


def _valid_name(name):
    """Whether this is a legal entry name, per the agentskills.io spec.

    1-64 characters, lowercase alphanumerics and hyphens, no leading, trailing
    or doubled hyphen. This is a security check rather than tidiness: the name
    becomes a path component, and the character set is what makes ``..``, a
    separator and a drive letter all impossible to express.
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


def _exists(sdk, path):
    """Whether a file is there. ``fs.list`` answers for a file too."""
    try:
        sdk.fs.list(path)
        return True
    except sdk.Failed:
        return False


def _existing(sdk, name):
    """``(path, kind)`` for an entry that is already there, or ``(None, "")``."""
    for path, kind in ((_note_path(sdk, name), "note"),
                       (_skill_path(sdk, name), "skill")):
        if _exists(sdk, path):
            return path, kind
    return None, ""


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


class MemoryCurate(BaseTool):
    """Write, revise and remove memory entries."""

    name = "memory_curate"
    description = (
        "Create, update or delete a memory entry — a note (one situation and "
        "what to do about it) or a skill (a repeatable procedure). Entries "
        "outlive the conversation and are searched at the start of every "
        "future turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "delete"],
                "description": (
                    "create: write a new entry. update: replace an existing "
                    "entry's body, and its description if you give one. "
                    "delete: remove it."
                ),
            },
            "name": {
                "type": "string",
                "description": (
                    "Lowercase letters, digits and hyphens — e.g. "
                    "'retry-failed-uploads'. Name it for the situation, not "
                    "for this conversation. Not a path."
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
                    "yields no text' — not a topic label. Required to create."
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
                    "A few words on what you are recording and why, shown to "
                    "the user beside the call. E.g. 'noting that the parser "
                    "was the problem, not the file'."
                ),
            },
        },
        "required": ["action", "name"],
    }
    requires_services = []

    agent_prompt = (
        "## Writing things down\n"
        "`memory_curate` is how anything outlives this conversation. Write an "
        "entry when you learn an action worth repeating or avoiding — "
        "especially a trap, a correction the user made, or something that "
        "broke. Those are worth more than successes, because nobody writes "
        "them down.\n"
        "One test: **will this change what an agent does next time?** If you "
        "cannot name the action, there is nothing to write. Records of what "
        "happened, anything true only inside this conversation, and anything "
        "obvious from reading the code are all noise that makes the rest "
        "harder to find.\n"
        "A **note** is one situation and its lesson. A **skill** is a "
        "repeatable procedure worth its own folder, for `references/` and "
        "`scripts/` you add with your file tools afterwards. Search first with "
        "`memory_recall list` — if an entry already covers the situation, "
        "update it rather than adding a second, because two entries about one "
        "situation is how a corpus stops being useful.\n"
        "Facts with no action attached — names, paths, which machine "
        "something runs on, a stated preference — are not entries. Those go "
        "in MEMORY.md, which is yours to maintain and is already in this "
        "prompt; this tool cannot touch it.\n"
        "Nothing here needs approval: it is all inside your workspace."
    )

    def run(self, sdk, **kwargs):
        """Create, update or delete one entry."""
        action = (kwargs.get("action") or "").strip().lower()
        name = (kwargs.get("name") or "").strip().lower()
        if not _valid_name(name):
            return sdk.fail(
                f"{name!r} is not a usable entry name. Use 1-64 lowercase "
                "letters, digits and hyphens, with no leading, trailing or "
                "doubled hyphen — and a name, never a path.")

        if action == "create":
            return self._create(sdk, name, kwargs)
        if action == "update":
            return self._update(sdk, name, kwargs)
        if action == "delete":
            return self._delete(sdk, name)
        return sdk.fail(
            f"Unknown action: {action!r}. Use create, update or delete.")

    # ── the three operations ─────────────────────────────────────────

    def _create(self, sdk, name, kwargs):
        """Write a new entry, refusing to shadow one that exists."""
        existing, kind = _existing(sdk, name)
        if existing:
            return sdk.fail(
                f"A {kind} named {name!r} already exists. Read it with "
                "memory_recall and update it — two entries about one "
                "situation is how a corpus stops being useful.")

        description = self._description(kwargs)
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
        where = (" Add references/ and scripts/ beside it with your file tools."
                 if kind == "skill" else "")
        return sdk.ok(None, llm_summary=f"Wrote {kind} '{name}'.{where}")

    def _update(self, sdk, name, kwargs):
        """Replace an entry's body, and its description when one is given.

        A full replacement rather than a patch, because the caller read the
        entry with ``memory_recall`` before deciding to change it — and a
        partial edit against a file the model has not seen is how a corpus
        acquires contradictions.

        The description is required here rather than read back from the file,
        and that is the reason this tool declares no ``fs.read`` at all: the
        writer is not a reader, which keeps the two tools' capabilities
        disjoint and keeps this one unable to be used to see anything. It also
        happens to be the better prompt — sharpening the field the entry is
        retrieved by is usually the most valuable part of a revision.
        """
        path, kind = _existing(sdk, name)
        if not path:
            return sdk.fail(
                f"No memory entry named {name!r} to update. Use "
                "memory_recall list to see what is there, or create it.")

        body = (kwargs.get("body") or "").strip()
        if not body:
            return sdk.fail("'update' needs a body — it replaces the old one.")
        description = self._description(kwargs)
        if not description:
            return sdk.fail(
                "'update' needs a description too. Restate it, or sharpen it "
                "so the entry also fires on the situation that just came up — "
                "that is usually the most valuable thing to change.")

        sdk.fs.write(path, _document(name, description, body,
                                     self._conversation(sdk)))
        return sdk.ok(None, llm_summary=f"Updated {kind} '{name}'.")

    def _delete(self, sdk, name):
        """Remove an entry, and a skill's whole folder with it."""
        path, kind = _existing(sdk, name)
        if not path:
            return sdk.fail(f"No memory entry named {name!r} to delete.")
        target = (sdk.path.join(_memory_root(sdk), SKILLS_DIRNAME, name)
                  if kind == "skill" else path)
        sdk.fs.delete(target)
        return sdk.ok(None, llm_summary=f"Deleted {kind} '{name}'.")

    # ── helpers ──────────────────────────────────────────────────────

    def _description(self, kwargs):
        """The supplied description, collapsed to one line and capped.

        Newlines are flattened because this is rendered as a single bullet in
        every prompt where the entry ranks, and a description that breaks the
        line breaks the block.
        """
        raw = " ".join((kwargs.get("description") or "").split())
        return raw[:MAX_DESCRIPTION]

    def _conversation(self, sdk):
        """Which conversation this lesson came from, for the ``source`` field."""
        try:
            return (sdk.session.get() or {}).get("conversation_id")
        except sdk.Failed:
            return None
