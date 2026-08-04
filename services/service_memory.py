"""Automatic memory retrieval.

Every turn, search the memory folder for notes relevant to what the user just
said and put *pointers* to them in the prompt — never the notes themselves.
The agent reads what it decides is worth reading.

Pointers rather than bodies is the whole design. A memory is one atomic fact
and a skill is a procedure that is wrong in fragments, so injecting matched
*chunks* would hand the agent half a procedure and let it follow that half.
A path plus the situation the note claims to cover is enough to decide with,
and costs a few dozen tokens instead of a few thousand.

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

#: How much of a note to read when building its pointer line. Frontmatter
#: lives at the top, so this never needs to be the whole file.
HEAD_CHARS = 1200

#: Fields a pointer line can use, in the order they are tried for the summary.
SUMMARY_KEYS = ("when", "description")


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
            fields[key.strip().lower()] = value.strip().strip("'\"")
    return fields


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
        "Durable notes live in the `memory` folder of your workspace, one "
        "markdown file per idea — facts, preferences, reusable procedures, "
        "lessons from things that went wrong. At the start of each turn, notes "
        "matching the user's message are listed under 'Possibly relevant "
        "memories'.\n"
        "That list is a map, not the content: it gives you a path and what the "
        "note claims to cover. Read a file before relying on it, and do not "
        "answer from a pointer line alone.\n"
        "You may write notes yourself with the file-editing tools whenever you "
        "learn something with a life beyond this conversation — no approval is "
        "needed inside your workspace. Start each note with frontmatter "
        "(`name`, `type`, `description`, `when`, `keywords`, `created`, "
        "`updated`, `source`). `when` is the most important field: write it as "
        "the *situation* that should bring the note back, not as a topic "
        "label, because that is what future retrieval matches against. Notes "
        "you do not write are written for you after a conversation ends, so "
        "there is no need to record everything as you go."
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
        """Build the pointer block: one line per note, no bodies."""
        lines = []
        for hit in hits:
            path = str(hit["path"])
            if sdk.path.name(path).lower() == "readme.md":
                continue
            lines.append(f"- {path}{self._summarize(sdk, hit, path)}")
        if not lines:
            return ""
        return ("## Possibly relevant memories\n"
                "These matched the current message. Read one before relying "
                "on it — this list is a map, not the content.\n"
                + "\n".join(lines))

    def _summarize(self, sdk, hit, path):
        """What a note claims to cover, preferring its own declaration."""
        try:
            head = sdk.fs.read(path)[:HEAD_CHARS]
        except sdk.Failed:
            head = ""
        fields = _frontmatter(head)
        for key in SUMMARY_KEYS:
            if value := fields.get(key):
                dated = fields.get("updated") or fields.get("created") or ""
                return f" — {value}" + (f" ({dated})" if dated else "")
        excerpt = " ".join(str(hit.get("content") or "").split())[:140]
        return f" — {excerpt}" if excerpt else ""


_README = """# Memory

One markdown file per idea. Anything durable belongs here: facts, preferences,
reusable procedures, lessons from something that went wrong.

Files are found by search, not by name, so the frontmatter is what makes a note
reachable:

```
---
name: pdf-parse-debugging
type: skill
description: How to debug the PDF parse chain when extraction returns empty
when: A PDF yields no text, or extraction produces an empty document
keywords: [pdf, parse, extraction]
created: 2026-01-01
updated: 2026-01-01
source: conversation 1
supersedes: []
---
```

`when` is the field that does the work — it is matched against what the user
says, so write it as the situation that should bring the note back rather than
as a topic label.

MEMORY.md is separate and is not part of this: it is a scratch index inlined
into the prompt directly, and nothing here reads or rewrites it.
"""
