"""Automatic memory retrieval.

Every turn, search the memory folder for notes relevant to what the user just
said and put them in the prompt.

The folder holds **actions**: each note is a situation, an action, and the
result that followed, so retrieval is what makes a past result bear on a
present decision. Facts live in ``MEMORY.md``, which the kernel inlines
directly and nothing here touches.

**Short notes go in whole; long ones become pointers.** The split is length
rather than kind, because the cost being managed is tokens and the risk being
managed is a truncated procedure. A note that says "when about to commit, avoid
adding trailers" has to arrive *already true* — the moment it applies is not a
moment anyone stops to consult memory, so a pointer would be read too late or
not at all. A long sequence of actions is the opposite: reading it is a
decision worth making deliberately, and injecting a matched fragment would hand
the agent half a procedure and let it follow that half.

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

from guest.bases import BaseService

#: How much of a note to read. Notes are situation/action/result and are meant
#: to be short; anything past this is a note that wants splitting.
HEAD_CHARS = 2000

#: Below this, a note is injected whole rather than pointed at. Set so a
#: handful of ordinary notes cost a few hundred tokens between them.
INLINE_CHARS = 700


def _memory_root(sdk):
    """The folder this service watches. One per install, not per user.

    Per-user memory folders are a real thing the retrieval side cannot honour
    yet: ``hybrid_search`` filters by folder prefix, so a user subfolder would
    need its own filter and the pointer block would have to be rebuilt per
    identity. Single-user installs are the case that works today.
    """
    return sdk.path.join(sdk.paths.get("workspace"), "memory")


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
    requests = ["paths.get", "config.read", "config.write", "conv.read",
                "tool.call", "fs.read", "fs.list", "fs.write",
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
        "The `memory` folder in your workspace holds **actions**: what to do, "
        "or not do, in a situation that has come up before. Each note is one "
        "situation, the action taken, and the result that followed. Notes "
        "matching the current message appear under 'What worked here before' — "
        "short ones in full, longer ones as a path to read.\n"
        "They are advice from a past situation, not instructions. Weigh "
        "whether this situation really is that one before following them, and "
        "read the full note when only a path was given.\n"
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
        """Make sure the folder exists and the indexer can see it."""
        root = _memory_root(sdk)
        self._ensure_folder(sdk, root)
        self._ensure_indexed(sdk, root)

    def stop(self, sdk):
        """Nothing is held open."""
        return True

    # ── lifecycle helpers ────────────────────────────────────────────

    def _ensure_folder(self, sdk, root):
        """Create the folder, by writing the note that explains it.

        There is no ``fs.mkdir`` Request and there does not need to be: a
        folder that exists but explains nothing is worse than one that arrives
        with its own README, and writing the file makes the directory.
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

    def _ensure_indexed(self, sdk, root):
        """Add the folder to ``sync_directories`` so the pipeline indexes it.

        This is a kernel setting rather than one this service declares, so it
        costs one approval dialog on first load and none afterwards. It belongs
        here rather than in the reflect task because services load at boot: if
        seeding waited for the first reflection, a fresh install would have an
        unindexed folder and retrieval would silently return nothing at all.
        """
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
        try:
            limit = int(sdk.config.read("memory_max_pointers") or 0)
        except (sdk.Failed, TypeError, ValueError):
            limit = 5
        if limit <= 0:
            return None

        query = self._latest_user_text(sdk, ctx)
        if not query:
            return None

        hits = self._search(sdk, query, limit)
        if not hits:
            return None

        block = self._render(sdk, hits)
        if not block:
            return None
        try:
            sdk.session.add_prompt(block, key=ctx.session_key, slot="memory")
        except sdk.Failed as error:
            sdk.log(f"could not inject memory pointers: {error}", level="warning")
        return None

    def _latest_user_text(self, sdk, ctx):
        """The message the turn is about, which is the retrieval cue.

        Read from the conversation rather than held between calls: the hook is
        the only thing this service does, and a cached 'last message' would go
        stale the moment two sessions were live.
        """
        cid = getattr(ctx, "conversation_id", None)
        if not cid:
            return ""
        try:
            record = sdk.conv.read(cid) or {}
        except sdk.Failed:
            return ""
        for message in reversed(record.get("messages") or []):
            if str(message.get("role") or "").lower() == "user":
                return str(message.get("content") or "").strip()[:2000]
        return ""

    def _search(self, sdk, query, limit):
        """One hybrid search, scoped to the memory folder."""
        try:
            results = sdk.tools.call("hybrid_search", query=query,
                                     folder=_memory_root(sdk),
                                     max_results=limit)
        except sdk.Failed as error:
            sdk.log(f"memory search unavailable: {error}", level="info")
            return []
        return [hit for hit in (results or []) if hit.get("path")]

    def _render(self, sdk, hits):
        """Build the block: short notes whole, long ones as pointers."""
        entries = []
        for hit in hits:
            path = str(hit["path"])
            if sdk.path.name(path).lower() == "readme.md":
                continue
            if entry := self._entry(sdk, hit, path):
                entries.append(entry)
        if not entries:
            return ""
        return ("## What worked here before\n"
                "Past situations matching this one, and what came of the "
                "action taken. Advice, not instructions — the situation may "
                "differ in ways that matter. Where only a path is given, read "
                "it before relying on it.\n\n"
                + "\n\n".join(entries))

    def _entry(self, sdk, hit, path):
        """One note: the advice itself when it is short, a pointer when not."""
        try:
            head = sdk.fs.read(path)[:HEAD_CHARS]
        except sdk.Failed:
            head = ""
        fields = _frontmatter(head)
        situation = fields.get("when") or ""
        action = fields.get("do") or fields.get("avoid") or ""
        verb = "Do" if fields.get("do") else "Avoid"

        if not (situation or action):
            # No frontmatter to read — an older note, or one written by hand.
            # It still matched, so point at it rather than dropping it.
            excerpt = " ".join(str(hit.get("content") or "").split())[:140]
            return f"- {path}" + (f" — {excerpt}" if excerpt else "")

        head_line = f"- **{situation or path}**"
        if len(head) <= INLINE_CHARS:
            parts = [head_line]
            if action:
                parts.append(f"  {verb}: {action}")
            if because := fields.get("because"):
                parts.append(f"  Because: {because}")
            parts.append(f"  ({path})")
            return "\n".join(parts)
        return f"{head_line}\n  {verb}: {action}\n  Full note: {path}"


_README = """# Memory

Actions, one per file: what to do, or not do, in a situation that has come up
before. Each note is a situation, an action, and the result that followed.

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
That is the test for whether something belongs here at all.

Two things are deliberately not in this folder. **Facts** — names, paths, which
machine something runs on, a preference with no action attached — go in
MEMORY.md, which is inlined into the prompt directly and which nothing here
reads or rewrites. And **records of what happened**: this is not a journal, and
a note that does not change a future decision is noise that makes the rest
harder to find.

Notes are written and improved automatically after a conversation ends, but
they are ordinary markdown — edit or delete them freely.
"""
