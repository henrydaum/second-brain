"""Automatic memory — retrieval at the start of a turn, a nudge at the end.

Every turn, ask Jev which memory entries fit what is happening right now and
put their names in the prompt. When the agent is about to finish, send it back
once with a note asking whether anything from the turn is worth saving.

One of two. This service picks entries, puts names in the prompt, and nudges;
``tool_memory`` is the only thing that touches the files in either direction.

The folder holds two kinds of entry and retrieval does not distinguish between
them. A **note** (``notes/<name>.md``) is one situation and what to do about
it. A **skill** (``skills/<name>/SKILL.md``) is a repeatable procedure that may
carry its own references and scripts. Both carry agentskills.io frontmatter —
``name``, ``description`` and ``when_to_retrieve`` — which is the whole reason
they can rank in one list and render as one shape. Facts live in
``MEMORY.md``, which the kernel inlines directly and nothing in this suite
touches.

**Retrieval is two Jev requests, and the hard part is knowing *when*.** The
first is one ``choice`` question over every entry's description, which
shortlists the few that could be the present situation. The second asks each
shortlisted entry's own ``when_to_retrieve`` questions — yes/no questions its
author wrote about what a conversation looks like when the entry applies — and
an entry is offered only if *every* one of them clears the threshold. The
shortlist makes it cheap; the entry's own questions make it specific. This is
TypeSafe's skill-suggestion pattern with the generic "does this fit?" gate
replaced by questions written for the entry.

It replaced one ``hybrid_search`` over the folder. That search is built for the
user's corpus — documents, ranked by resemblance to a query — and "is this past
situation the one I am in?" is a decision rather than a resemblance. It also
made memory depend on the indexing chain and an embedding model, and made
every entry wait for a sync before it could be found.

**The prompt gets names and descriptions, never bodies.** The only decision to
make from the prompt is whether a past situation is the present one, and the
description answers exactly that; what to do about it is what the entry is for.
Injecting the body as well was tried and had to come out — it grew with the
corpus, truncated long entries into advice stripped of its context, and
destroyed the one observable signal in the system. With the content already in
the prompt there is no reason to open anything, and which entries were opened is
the only evidence of which ones earn their place. So this service also records
what it surfaced, with the probabilities that surfaced it (``memory_usage``);
``tool_memory`` records the other half.

**Writing happens at the end of the turn, by the agent that did the work.**
The agent at ``end_turn`` already holds the whole turn in its context — cached
— so the nudge costs one extra model call and loses nothing. The note is
ephemeral: the model sees it, the transcript does not keep it. Writing itself
is still only ``tool_memory``'s.
"""

import json
import time

from guest.bases import BaseService
from guest.hooks import SendBack

#: How much of an entry to read when building its line. Only the frontmatter is
#: wanted, and that is at the top — but it now carries up to three questions
#: beside a description of up to 1,024 characters. Must match ``tool_memory``.
HEAD_CHARS = 4000

#: How long a description may be in the prompt block and in the shortlist
#: question. Long enough to recognise a situation, short enough that 255 of
#: them fit one Jev request.
MAX_DESCRIPTION_CHARS = 200

#: Where entries live, under the memory root, and the only two places read.
#: Everything else in ``memory/`` — ``MEMORY.md``, the README, drafts, anything
#: the agent leaves lying around — is outside them and therefore outside
#: retrieval. Must match the constants in ``tool_memory``.
MEMORY_DIRNAME = "memory"
NOTES_DIRNAME = "notes"
SKILLS_DIRNAME = "skills"

#: Jev's ceiling on the options of one ``choice`` question. A bigger corpus is
#: shortlisted in chunks, then the chunk winners compete in one more question.
MAX_CHOICE_OPTIONS = 255

#: An entry written before ``when_to_retrieve`` existed has no questions to
#: gate it, so the shortlist is the only evidence — and it must be stronger.
UNQUESTIONED_CHOICE_THRESHOLD = 0.5

#: How much of the latest agent turn is shown to Jev, and how much of any one
#: tool argument. An ``edit_file`` body can be 100 KB; the call's name and its
#: narration are what say what the agent was doing.
MAX_CONTEXT_CHARS = 6000
MAX_ARG_CHARS = 300
MAX_REQUEST_CHARS = 2000

#: Rows looked back through for the latest agent turn. A turn with more tool
#: calls than this is summarised by its most recent ones, which is fine.
CONTEXT_ROWS = 40

#: Separates an entry's name from its question id in the gate request. Entry
#: names cannot contain a colon, so the split is unambiguous.
QID_SEPARATOR = "::"

SHORTLIST_INSTRUCTIONS = (
    "Each option is a situation from a past conversation. Which one is the "
    "situation the user is in now, given their latest request and what the "
    "agent just did?"
)

#: How long an offer nobody took is kept. Long enough to span a conversation
#: someone comes back to; after that it says nothing.
USAGE_RETENTION_SECONDS = 7 * 24 * 3600

#: Sessions the kernel opens for subagents. Their turns are nudged by nobody:
#: the work belongs to whoever spawned them.
SUBAGENT_PREFIX = "spawn_subagent:"

#: What the agent is told when it is about to finish. Written to be declined:
#: most turns hold nothing, and the one way this goes wrong is an agent that
#: saves something every time because it was asked every time. The comeback is
#: *quiet*, so nothing the agent says in it reaches the user — the note says so,
#: or a model trained to always answer will answer the user's question again.
NUDGE = (
    "[Memory check — from the system, not the user. The user will not see "
    "anything you write in reply to this.] Did this turn teach anything a "
    "future conversation would need: a correction the user made, something "
    "that failed and what fixed it, a procedure that worked, or something the "
    "user asked you to remember? If so, save it with the `memory` tool — "
    "update an existing entry when one covers it. A lasting fact about the "
    "user goes in `MEMORY.md` instead. Most turns hold nothing. "
    "Either way, do not reply to the user again; end with an empty message."
)

#: Added to the nudge when the agent opened entries this turn. A memory that
#: was read and then contradicted by what happened is the one moment its
#: staleness is visible — and an entry that surfaced when it did not apply is
#: the one moment its questions are visibly wrong.
STALE_CHECK = (
    " You opened these memories this turn: {names}. If any turned out wrong "
    "or outdated, fix it with `memory update`, or `memory delete` it if it no "
    "longer applies at all. If one came up when it did not fit, sharpen its "
    "`when_to_retrieve` questions."
)

#: Added when an opened entry predates ``when_to_retrieve``. The agent that
#: just used it knows best what situation it fits.
UNQUESTIONED_CHECK = (
    " These have no `when_to_retrieve` questions yet: {names}. Give them some "
    "with `memory update`."
)


def _memory_root(sdk):
    """The folder this service reads. One per install, not per user."""
    return sdk.path.join(sdk.paths.get("workspace"), MEMORY_DIRNAME)


def _entry_dirs(sdk):
    """The two folders that hold entries.

    Membership is location, not content, and that is the point. Requiring a
    field in the frontmatter made *being an entry* a property the writer had to
    restate correctly in every file, and getting it subtly wrong made the entry
    silently unreachable. A path cannot be subtly wrong.
    """
    root = _memory_root(sdk)
    return (sdk.path.join(root, NOTES_DIRNAME),
            sdk.path.join(root, SKILLS_DIRNAME))


def _frontmatter(text):
    """Parse the leading ``---`` block into a dict, tolerating anything.

    Deliberately not a YAML parser: an entry is written by a language model and
    the failure mode that matters is a malformed block taking the whole turn
    down. Unparseable lines are skipped. Splitting on the *first* colon is what
    lets ``when_to_retrieve`` carry a line of JSON.
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


def _questions(raw):
    """An entry's ``when_to_retrieve``, or ``{}`` when it has none usable.

    Only ``noul`` questions are kept — the gate is "every one is yes", which
    means nothing for a choice or a score. A value that does not parse is an
    entry with no questions, not a broken turn.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(qid): q for qid, q in parsed.items()
            if isinstance(q, dict) and q.get("type") == "noul"
            and q.get("instructions")}


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


class MemoryRetrieve(BaseService):
    """Surface the memory entries that fit the current turn."""

    name = "memory_retrieve"
    description = "Asks Jev which memory notes and skills fit the current turn and points the agent at them."

    exports = []
    hooks = {"turn_start": "on_turn_start", "end_turn": "on_end_turn"}
    requests = ["paths.get", "config.read", "service.call",
                "fs.read", "fs.list", "fs.write",
                "db.define", "db.query", "db.write",
                "session.add_prompt_extra", "session.remove_prompt_extra"]
    # What retrieval asks, what the agent writes with when nudged, and what it
    # reads a skill's references with (``memory read`` names them but
    # deliberately does not load them).
    dependencies_files = ["services/service_rlcd.py",
                          "tools/tool_memory.py",
                          "tools/tool_read_file.py"]
    dependencies_pip = []

    config_settings = [
        ("Memory pointers", "memory_max_pointers",
         "The most memory entries to surface at the start of each turn. 0 disables retrieval.",
         3, {"type": "slider", "range": (0, 15, 15), "is_float": False}),
        ("Memory candidates", "memory_candidates",
         "How many entries the first pass shortlists for their own questions to be asked.",
         5, {"type": "slider", "range": (1, 25, 24), "is_float": False}),
        ("Memory question threshold", "memory_question_threshold",
         "Every one of an entry's when_to_retrieve questions must score above this for it to be surfaced.",
         0.5, {"type": "slider", "range": (0.0, 1.0, 20), "is_float": True}),
    ]

    # Says only what this service is the authority on: which folders are read,
    # and what the list it injects is. How to open an entry, and what earns
    # one, belong to the ``memory`` tool's own block.
    agent_prompt = (
        "## Memory\n"
        "`memory/` in your workspace holds what you have learned, as `notes/` "
        "and `skills/`. Only those two folders are searched, so the rest of "
        "`memory/` is free for drafts and scratch files.\n"
    )

    def on_uninstall(self, sdk):
        """Drop the usage table.

        ``memory_usage`` is unambiguously this plugin's — nothing else writes it
        and nothing else can read anything out of it. The notes and skills
        themselves are never touched. They are the user's writing, in the
        user's workspace.
        """
        try:
            sdk.db.define("DROP TABLE IF EXISTS memory_usage")
        except sdk.Failed as error:
            sdk.log(f"could not drop the memory usage table: {error}",
                    level="warning")

    def start(self, sdk):
        """Make the folder and the usage table, and prune stale offers."""
        # Standing misconfigurations already reported. See ``_say_once``.
        self._said = set()
        self._ensure_folder(sdk, _memory_root(sdk))
        self._ensure_usage_table(sdk)
        self._prune_usage(sdk)

    def _ensure_usage_table(self, sdk):
        """The one table that records the life of a memory: offered, then taken.

        Defined here because this service is the first writer — it inserts a
        row per offer, with the probabilities that made it, and
        ``tool_memory`` fills ``recalled_at``. Offered-then-opened against
        ``p_choice`` and ``p_min`` is what the thresholds should be tuned on.
        """
        try:
            sdk.db.define(
                "CREATE TABLE IF NOT EXISTS memory_usage ("
                " id INTEGER PRIMARY KEY,"
                " name TEXT NOT NULL,"
                " conversation_id INTEGER,"
                " offered_at REAL,"
                " recalled_at REAL,"
                " p_choice REAL,"
                " p_min REAL)")
            # Every recall looks a pending offer up by (conversation, name),
            # against a table that gains a row per offer forever.
            sdk.db.define(
                "CREATE INDEX IF NOT EXISTS memory_usage_lookup"
                " ON memory_usage (conversation_id, name, recalled_at)")
        except sdk.Failed as error:
            sdk.log(f"could not create the memory usage table: {error}",
                    level="warning")
            return
        # A table made before the probabilities were recorded. SQLite has no
        # ADD COLUMN IF NOT EXISTS, so a duplicate column is the success case.
        for column in ("p_choice", "p_min"):
            try:
                sdk.db.define(f"ALTER TABLE memory_usage ADD COLUMN {column} REAL")
            except sdk.Failed as error:
                if "duplicate" not in str(error).lower():
                    sdk.log(f"could not add {column} to memory_usage: {error}",
                            level="warning")

    def _prune_usage(self, sdk):
        """Forget offers nobody took, once they are old enough to mean nothing.

        Taken ones are kept: they are the history of which entries earn their
        place. Once per start is enough.
        """
        cutoff = time.time() - USAGE_RETENTION_SECONDS
        try:
            sdk.db.write(
                "DELETE FROM memory_usage"
                " WHERE recalled_at IS NULL AND offered_at < ?", [cutoff])
        except sdk.Failed as error:
            sdk.log(f"could not prune memory offers: {error}", level="warning")

    def stop(self, sdk):
        """Nothing is held open."""
        return True

    # ── lifecycle helpers ────────────────────────────────────────────

    def _ensure_folder(self, sdk, root):
        """Create the folder, by writing the note that explains it.

        ``notes/`` and ``skills/`` are deliberately *not* pre-created. A
        placeholder inside either would be read like any other entry, found to
        have no description, and reported as broken every single turn.
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
        """Ask Jev what fits and point the agent at it.

        Every failure path abstains rather than raising. Jev being unreachable
        and an empty folder are both reasons to offer nothing, never reasons
        for a turn to fail.
        """
        key = str(getattr(ctx, "session_key", ""))
        began = getattr(self, "_turn_began", None)
        if began is None:
            began = self._turn_began = {}
        began[key] = time.time()

        offered = []
        block = self._pointers(sdk, ctx, offered)
        if not block:
            # The overlay persists until its slot is rewritten, so a turn that
            # finds nothing has to say so — otherwise the previous turn's list
            # stays in the prompt, pointing at a situation that has passed.
            try:
                sdk.session.remove_prompt("memory")
            except sdk.Failed as error:
                sdk.log(f"could not clear memory pointers: {error}",
                        level="warning")
            return None
        try:
            # No ``key``: naming a session makes this "inject into *that*
            # session", which is unsafe from a hook's chain and therefore
            # refused outright. Omitting it means "my own session".
            sdk.session.add_prompt(block, slot="memory")
        except sdk.Failed as error:
            # Nothing was shown, so nothing was offered.
            sdk.log(f"could not inject memory pointers: {error}", level="warning")
            return None
        self._log_offered(sdk, ctx, offered)
        return None

    def _setting(self, sdk, key, default, cast):
        try:
            value = sdk.config.read(key)
            return default if value is None else cast(value)
        except (sdk.Failed, TypeError, ValueError):
            return default

    def _pointers(self, sdk, ctx, offered):
        """The block for this turn, or ``""`` when there is nothing to show.

        ``offered`` collects ``(name, p_choice, p_min)`` for the usage table.
        """
        limit = self._setting(sdk, "memory_max_pointers", 3, int)
        if limit <= 0:
            sdk.log("memory retrieval is disabled (memory_max_pointers is 0)",
                    level="debug")
            return ""

        state = self._state(sdk, ctx)
        if not state:
            # Ordinary at the very start of a conversation.
            sdk.log("memory: no user message to decide on yet", level="debug")
            return ""

        corpus = self._corpus(sdk)
        # Remembered for the nudge: which entries still need questions.
        self._unquestioned = {entry["name"] for entry in corpus
                              if not entry["questions"]}
        if not corpus:
            return ""

        chosen = self._retrieve(sdk, state, corpus, limit)
        if not chosen:
            return ""

        lines = []
        for entry, p_choice, p_min in chosen:
            label = (f"{entry['name']} (skill)" if entry["kind"] == "skill"
                     else entry["name"])
            lines.append(f"- {label} — {entry['description']}")
            offered.append((entry["name"], p_choice, p_min))
        return ("## Things you have done before\n"
                "These fit what is happening now. Each is a note or skill you "
                "wrote earlier; the description is all you get here, so "
                "`memory read` the name when one looks close enough to your "
                "situation to be worth learning from. Ignore any that do not "
                "fit what the user actually asked for.\n\n"
                + "\n".join(lines)
                + self._more(len(lines), len(corpus)))

    # ── state ────────────────────────────────────────────────────────

    def _state(self, sdk, ctx):
        """What Jev decides about: the user's request and the agent's last turn.

        ``{}`` when there is no user message yet.
        """
        latest = self._latest_user_message(sdk, ctx)
        if not latest:
            return {}
        row_id, request = latest
        return {"request": request,
                "recent_context": self._recent_context(sdk, ctx, row_id)}

    def _latest_user_message(self, sdk, ctx):
        """``(row id, text)`` of the message the turn is about, or ``None``.

        ``author`` is the test, not ``role`` alone. The kernel writes user-role
        rows the person never typed — a cancel notice, a doorman's note — and
        retrieval keyed off one of those decides about the wrong thing.
        """
        cid = getattr(ctx, "conversation_id", None)
        if not cid:
            return None
        try:
            rows = sdk.db.query(
                "SELECT id, content FROM conversation_messages"
                " WHERE conversation_id = ? AND LOWER(role) = 'user'"
                "   AND COALESCE(author, '') = ''"
                "   AND COALESCE(content, '') <> ''"
                " ORDER BY id DESC LIMIT 1", [int(cid)], max_rows=1)
        except sdk.Failed as error:
            sdk.log(f"memory: could not read conversation {cid}: {error}",
                    level="warning")
            return None
        if not rows:
            return None
        text = str(rows[0].get("content") or "").strip()[:MAX_REQUEST_CHARS]
        return (int(rows[0]["id"]), text) if text else None

    def _recent_context(self, sdk, ctx, before_id):
        """The agent's turn before this message: its words and its tool calls.

        Tool *results* are dropped — the user does not respond to them, and
        they are where the bulk is. The call itself stays, with its narration,
        because that is what says what the agent was doing. Stops at the
        previous real user message, so only the latest agent turn is shown.
        """
        cid = getattr(ctx, "conversation_id", None)
        try:
            rows = sdk.db.query(
                "SELECT role, content, author FROM conversation_messages"
                " WHERE conversation_id = ? AND id < ?"
                " ORDER BY id DESC LIMIT ?",
                [int(cid), int(before_id), CONTEXT_ROWS], max_rows=CONTEXT_ROWS)
        except sdk.Failed as error:
            sdk.log(f"memory: could not read the last agent turn: {error}",
                    level="debug")
            return ""
        parts = []
        for row in rows or []:
            role = str(row.get("role") or "").lower()
            if role == "user":
                if not row.get("author"):
                    break  # the previous real message: that turn is over
                continue
            if role != "assistant":
                continue  # tool results, markers
            parts.append(self._render_assistant(str(row.get("content") or "")))
        text = "\n".join(part for part in reversed(parts) if part)
        if len(text) > MAX_CONTEXT_CHARS:
            text = "…" + text[-MAX_CONTEXT_CHARS:]
        return text

    def _render_assistant(self, content):
        """One assistant row as text: what it said, then what it called."""
        packed = None
        if content.startswith("{"):
            try:
                packed = json.loads(content)
            except ValueError:
                packed = None
        if not isinstance(packed, dict) or "tool_calls" not in packed:
            return content.strip()
        lines = [str(packed.get("content") or "").strip()]
        for call in packed.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            name = function.get("name") or "?"
            args = function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"arguments": args}
            if not isinstance(args, dict):
                args = {}
            shown = ", ".join(f"{key}={_clip(value, MAX_ARG_CHARS)}"
                              for key, value in args.items())
            lines.append(f"tool: {name}({shown})")
        return "\n".join(line for line in lines if line)

    # ── corpus ───────────────────────────────────────────────────────

    def _corpus(self, sdk):
        """Every entry: name, kind, description and questions, sorted by name.

        Sorted so the shortlist question is byte-identical across turns while
        the corpus is unchanged, which is what lets the provider cache it.
        """
        notes_dir, skills_dir = _entry_dirs(sdk)
        candidates = []
        try:
            for item in sdk.fs.list(notes_dir, pattern="*.md", details=True) or []:
                if not item.get("is_dir"):
                    candidates.append((sdk.path.stem(item["name"]), "note",
                                       sdk.path.join(notes_dir, item["name"])))
        except sdk.Failed:
            pass
        try:
            for item in sdk.fs.list(skills_dir, details=True) or []:
                if item.get("is_dir"):
                    candidates.append((item["name"], "skill",
                                       sdk.path.join(skills_dir, item["name"],
                                                     "SKILL.md")))
        except sdk.Failed:
            pass

        corpus, malformed, seen = [], [], set()
        for name, kind, path in sorted(candidates):
            if name in seen:
                continue
            seen.add(name)
            try:
                fields = _frontmatter(sdk.fs.read(path)[:HEAD_CHARS])
            except sdk.Failed:
                continue
            description = fields.get("description")
            if not description:
                malformed.append(name)
                continue
            corpus.append({"name": name, "kind": kind,
                           "description": _clip(description, MAX_DESCRIPTION_CHARS),
                           "questions": _questions(fields.get("when_to_retrieve"))})
        if malformed:
            # The symptom is otherwise an entry that is never once offered.
            self._say_once(
                sdk, "no-description:" + ",".join(malformed),
                "memory entries with no description were skipped: "
                + ", ".join(malformed))
        return corpus

    # ── the two Jev passes ───────────────────────────────────────────

    def _evaluate(self, sdk, state, questions):
        """One Jev request. ``None`` when it failed, which is said once."""
        try:
            result = sdk.services.call("rlcd", "evaluate", state, questions)
        except sdk.Failed as error:
            self._say_once(
                sdk, "rlcd-down",
                f"memory retrieval is off — the rlcd service could not be "
                f"asked: {error}. Install rlcd and set its API key, or set "
                f"memory_max_pointers to 0 to stop trying.")
            return None
        if not isinstance(result, dict) or not result.get("ok"):
            error = (result or {}).get("error") or {}
            self._say_once(
                sdk, "rlcd-down",
                f"memory retrieval is off — Jev answered "
                f"{error.get('kind', 'an error')}"
                + (f" (HTTP {error['status']})" if error.get("status") else "")
                + f": {error.get('message', '')}")
            return None
        return result.get("answers") or {}

    def _shortlist(self, sdk, state, entries, k):
        """The top ``k`` entries by one ``choice`` question, with probabilities.

        More than 255 entries do not fit one question, and probabilities from
        different questions do not compare — each sums to one — so each chunk
        sends its top ``k`` to one final question over the winners.
        """
        if len(entries) <= MAX_CHOICE_OPTIONS:
            return self._choose(sdk, state, entries, k)
        winners = []
        for start in range(0, len(entries), MAX_CHOICE_OPTIONS):
            ranked = self._choose(sdk, state,
                                  entries[start:start + MAX_CHOICE_OPTIONS], k)
            if ranked is None:
                return None
            winners.extend(entry for entry, _ in ranked)
        winners = sorted(winners, key=lambda entry: entry["name"])
        return self._shortlist(sdk, state, winners, k)

    def _choose(self, sdk, state, entries, k):
        if len(entries) == 1:
            return [(entries[0], 1.0)]
        question = {"shortlist": {
            "type": "choice",
            "instructions": SHORTLIST_INSTRUCTIONS,
            "criteria": {entry["name"]: entry["description"] for entry in entries},
        }}
        answers = self._evaluate(sdk, state, question)
        if answers is None:
            return None
        probabilities = (answers.get("shortlist") or {}).get("probabilities") or {}
        by_name = {entry["name"]: entry for entry in entries}
        ranked = sorted(((by_name[name], float(p))
                         for name, p in probabilities.items() if name in by_name),
                        key=lambda pair: (-pair[1], pair[0]["name"]))
        return ranked[:k]

    def _retrieve(self, sdk, state, corpus, limit):
        """``[(entry, p_choice, p_min)]`` for what clears the gate, best first."""
        k = max(1, self._setting(sdk, "memory_candidates", 5, int))
        threshold = self._setting(sdk, "memory_question_threshold", 0.5, float)

        shortlist = self._shortlist(sdk, state, corpus, k)
        if not shortlist:
            return []

        questions = {}
        for entry, _ in shortlist:
            for qid, question in entry["questions"].items():
                questions[f"{entry['name']}{QID_SEPARATOR}{qid}"] = {
                    "type": "noul", "instructions": question["instructions"]}
        answers = {}
        if questions:
            answers = self._evaluate(sdk, state, questions)
            if answers is None:
                return []

        passed, rejected = [], []
        for entry, p_choice in shortlist:
            if not entry["questions"]:
                if p_choice >= UNQUESTIONED_CHOICE_THRESHOLD:
                    passed.append((entry, p_choice, None, p_choice))
                else:
                    rejected.append(f"{entry['name']} (choice {p_choice:.2f})")
                continue
            scores = [float((answers.get(f"{entry['name']}{QID_SEPARATOR}{qid}")
                             or {}).get("noul", 0.0))
                      for qid in entry["questions"]]
            p_min = min(scores)
            if p_min > threshold:
                passed.append((entry, p_choice, p_min, p_min))
            else:
                rejected.append(f"{entry['name']} (min {p_min:.2f})")
        if rejected:
            sdk.log("memory: shortlisted but not offered: " + ", ".join(rejected),
                    level="debug")
        passed.sort(key=lambda row: -row[3])
        return [(entry, p_choice, p_min)
                for entry, p_choice, p_min, _ in passed[:limit]]

    def _more(self, shown, total):
        """Say that the corpus is bigger than the list, when it is."""
        if total <= shown:
            return ""
        return (f"\n\nShowing {shown} of {total}. `memory list` shows "
                "them all.")

    # ── the end of the turn ──────────────────────────────────────────

    def on_end_turn(self, sdk, ctx, ending):
        """Send the agent back once to save anything worth keeping.

        Three cases pass straight through: a doorman already fired this turn,
        the turn is not a clean finish, or the session is a subagent's.
        Ephemeral, so the note is never recorded; quiet, so whatever the agent
        says once it is done is dropped and the user's reply stays the last
        word.
        """
        if getattr(ending, "doorman_fires", 0):
            return None
        if getattr(ending, "reason", "") not in ("", "model_finished"):
            return None
        if str(getattr(ctx, "session_key", "")).startswith(SUBAGENT_PREFIX):
            return None
        note = NUDGE
        if opened := self._opened_this_turn(sdk, ctx):
            note += STALE_CHECK.format(names=", ".join(opened))
            unquestioned = [name for name in opened
                            if name in (getattr(self, "_unquestioned", None) or ())]
            if unquestioned:
                note += UNQUESTIONED_CHECK.format(names=", ".join(unquestioned))
        return SendBack(note, ephemeral=True, quiet=True)

    def _opened_this_turn(self, sdk, ctx):
        """Entries the agent read with ``memory read`` since this turn began."""
        cid = getattr(ctx, "conversation_id", 0)
        began = (getattr(self, "_turn_began", None) or {}).pop(
            str(getattr(ctx, "session_key", "")), None)
        if not (cid and began):
            return []
        try:
            rows = sdk.db.query(
                "SELECT DISTINCT name FROM memory_usage"
                " WHERE conversation_id = ? AND recalled_at >= ?"
                " ORDER BY name", [int(cid), began], max_rows=20)
        except sdk.Failed as error:
            sdk.log(f"memory: could not read this turn's recalls: {error}",
                    level="warning")
            return []
        return [str(row.get("name")) for row in rows or [] if row.get("name")]

    def _log_offered(self, sdk, ctx, offered):
        """Record which entries were surfaced, and how strongly.

        Half of a pair: ``tool_memory`` fills in ``recalled_at`` if the agent
        goes on to open one. The prompt is stored nowhere, so this is the only
        place the offer is knowable.
        """
        cid = getattr(ctx, "conversation_id", None)
        if not (cid and offered):
            return
        now = time.time()
        for name, p_choice, p_min in offered:
            try:
                sdk.db.write(
                    "INSERT INTO memory_usage"
                    " (name, conversation_id, offered_at, recalled_at,"
                    "  p_choice, p_min)"
                    " VALUES (?, ?, ?, NULL, ?, ?)",
                    [str(name), int(cid), now, p_choice, p_min])
            except sdk.Failed as error:
                sdk.log(f"could not record a memory offer: {error}",
                        level="warning")
                return

    def _say_once(self, sdk, topic, message):
        """Log a standing misconfiguration the first time only.

        This runs at the top of every turn, and the conditions are *states*
        rather than events. Reset per process, so a restart says it again.
        Self-initialising rather than trusting ``start`` to have run first.
        """
        said = getattr(self, "_said", None)
        if said is None:
            said = self._said = set()
        if topic in said:
            sdk.log(message, level="debug")
            return
        said.add(topic)
        sdk.log(message, level="warning")


_README = """# Memory

    notes/       one file per situation — searched
    skills/      one folder per procedure — searched
    MEMORY.md    facts, inlined into the agent's prompt in full
    (anything else lives here and is ignored)

`notes/` holds what to do, or not do, in a situation that has come up before.
`skills/` holds repeatable procedures, each in its own folder, following the
[agentskills.io](https://agentskills.io) layout — a `SKILL.md` plus optional
`references/`, `scripts/` and `assets/`.

Both are found the same way. At the start of every turn a small model reads
every entry's `description` and shortlists the few that could fit, then asks
each shortlisted entry's `when_to_retrieve` questions about the conversation.
An entry is suggested only when every one of its questions comes back yes.
Write the description as the situation that should bring the entry back, never
as a topic label, and the questions as things that are true of a conversation
exactly when the entry applies.

```
---
name: pdf-yields-no-text
description: A PDF yields no text, or extraction produces an empty document
when_to_retrieve: {"pdf": {"type": "noul", "instructions": "Is the user working with a PDF file?"}, "empty": {"type": "noul", "instructions": "Did extracting or reading text from a file come back empty or fail?"}}
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
entry. Two things deliberately do not belong here: **facts** — names, paths,
which machine something runs on, a preference with no action attached — belong
in MEMORY.md, which is inlined into the prompt directly; and **records of what
happened**, because this is not a journal.

At the end of each turn the agent is asked whether anything is worth writing
down, and writes it with the `memory` tool. Entries are ordinary markdown —
edit or delete them freely.
"""
