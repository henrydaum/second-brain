"""What the kernel reads off the files that make up memory.

Same shape as ``test_store_attachment_tools``: kernel invariants that happen to
be *about* store files. The subject is the kernel's own verdict — does this
load, are these Requests real, is the retrieval free of dialogs, can this reach
outside the folder — and the store file is the input.

The two matter together because each is useless alone.
``service_memory_retrieve`` ranks the corpus at ``turn_start``, injects
descriptions and records what it offered, and at ``end_turn`` sends the agent
back once to save anything worth keeping. ``tool_memory`` is the only thing that
touches the files — in either direction — and records what was opened. The
curator task that used to reflect on a finished conversation is gone: the agent
that did the work already holds the turn in its context.

Reading and writing are one tool on purpose. They were two, whose
*declarations* were disjoint — the writer had no ``fs.read`` — and that bought
nothing a writer could not get through ``read_file`` one tool over, while
costing a revision the ability to keep a description it had just read. What
confines the tool is that it takes a name and derives every path itself, so the
tests below pin the character set and the two path templates rather than the
split.

What connects the two is not an import but the *folder* and one table, so
the things worth pinning are the declarations that decide whether any of it
runs at all: the hook moments and the shared constants.

Skips cleanly when no store ref is reachable.
"""

import json
from pathlib import Path

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source, store_worktree

SERVICE = "services/service_memory_retrieve.py"
MEMORY = "tools/tool_memory.py"
BUNDLE = "bundles/bundle_memory.json"

SUITE = [SERVICE, MEMORY]


def _source_or_skip(relative: str) -> str:
    text = store_source(relative)
    if text is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    return text


def _declarations(relative: str) -> dict:
    from sandbox.validator import validate

    return validate(_source_or_skip(relative),
                    filename=Path(relative).name).declarations


def _install_closure() -> set[str]:
    """Every store file installing this bundle lands, by the manager's own walk.

    The manifest names the four memory plugins and nothing else; everything
    else arrives because something in the closure declares it. Reading the
    closure rather than the manifest is what lets uninstall be *narrow* — it
    follows the edge backwards, so a shared dependency the bundle merely
    reached is not the bundle's to take away — while install stays complete.
    """
    from bundled.commands.helpers.package_manager import read_dependency_meta

    out: set[str] = set()
    queue = list(json.loads(_source_or_skip(BUNDLE))["files"])
    while queue:
        rel = queue.pop()
        if rel in out:
            continue
        out.add(rel)
        queue.extend(read_dependency_meta(rel, _source_or_skip(rel)).dependencies_files)
    return out


def _load_store_class(relative: str, name: str):
    namespace = {}
    exec(compile(_source_or_skip(relative), relative, "exec"), namespace)
    return namespace[name]


# ──────────────────────────────────────────────────────────────────────
# Does it load, and does it declare things that exist.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("relative", SUITE)
def test_the_memory_bundle_conforms(relative):
    """``conforms`` is the whole question: it means the file loads in a box."""
    from sandbox.validator import validate

    report = validate(_source_or_skip(relative), filename=Path(relative).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


@pytest.mark.parametrize("relative", SUITE)
def test_every_declared_request_is_a_real_one(relative):
    """``requests`` is the approval grant, so a typo silently narrows it."""
    from guest.requests import ALL_TYPES

    assert set(_declarations(relative)["requests"]) <= set(ALL_TYPES)


# ──────────────────────────────────────────────────────────────────────
# The corpus: one shape for notes and skills, decided by location.
# ──────────────────────────────────────────────────────────────────────

def test_both_files_that_hold_a_path_agree_on_where_entries_live():
    """Membership is a path, which is the one thing a writer cannot fumble.

    Requiring a field in the frontmatter made *being an entry* something the
    writer had to restate correctly in every file, and getting it subtly wrong
    — no fences, the key in the body — made the entry silently unreachable. A
    path cannot be subtly wrong.

    Two files derive that path independently, and they cannot be merged into
    one: they are different families, so a ``tools/helpers/`` module could not
    hold the service's copy. Merging the two tools took three copies to two —
    this test is what keeps the last two equal, which is why it survives the
    merge rather than being retired by it.
    """
    for relative in (SERVICE, MEMORY):
        source = _source_or_skip(relative)
        assert 'MEMORY_DIRNAME = "memory"' in source, relative
        assert 'NOTES_DIRNAME = "notes"' in source, relative
        assert 'SKILLS_DIRNAME = "skills"' in source, relative


def test_a_skill_and_a_note_rank_and_render_identically():
    """The point of the agentskills.io frontmatter is that there is one shape.

    Both kinds carry ``name`` and ``description``, so retrieval reads one field
    off both and the prompt line differs by a label and nothing else. A branch
    on kind anywhere in ranking would mean skills and notes competing in
    separate pools, which is the separation this design exists to remove.
    """
    service = _source_or_skip(SERVICE)

    assert '"description"' in service
    # One rendering, one label, no second code path.
    assert 'f"{name} (skill)" if kind == "skill" else name' in service
    # Everything under a skill folder collapses onto the skill itself, so a
    # matched reference file cannot occupy a line of its own.
    assert "_skill_of" in service


def test_the_prompt_carries_descriptions_and_not_the_entries():
    """Inlining the body destroys the signal the whole loop runs on.

    With the content already in the prompt there is no reason to recall
    anything, so nothing downstream can tell which entries were used — and that
    pair is what selects the curator's job. The description alone answers the
    only question the prompt has to answer, which is whether a past situation
    is this one.
    """
    service = _source_or_skip(SERVICE)

    assert 'entries.append(f"- {label} — {description}")' in service
    assert "MAX_DESCRIPTION_CHARS" in service
    # No body, no excerpt, no chunk fallback.
    assert 'hit.get("content")' not in service


def test_the_block_says_how_much_it_is_not_showing():
    """A block of five with no total reads as "this is all you have".

    An agent that believes it has five memories does not go looking for the
    sixty-two others, so the count is what turns an inventory into a sample.
    """
    service = _source_or_skip(SERVICE)
    assert "Showing {shown} of {total}" in service
    assert "_corpus_size" in service


def test_an_entry_with_no_description_is_reported_not_guessed_at():
    """Inside the entry folders, a missing description is a broken entry.

    There is nothing to render and falling back to the matched chunk would put
    a fragment with no context into a list that promises situations. The
    symptom otherwise is an entry that ranks well and is never once offered,
    which is indistinguishable from having no memories at all.
    """
    service = _source_or_skip(SERVICE)

    assert "malformed" in service
    assert "with no description were skipped" in service


# There is deliberately no test that the three halves *word* the rule the same
# way — that an entry naming no action is not worth writing. There was one, and
# it matched prose fragments in each file's description and agent prompt. What
# it caught in practice was a rewording: the rule was still stated, in better
# words, and the test failed for the words. A prompt is edited constantly and by
# design, so pinning its phrasing puts a red suite in front of every
# improvement, and the only way to keep it green is to write the sentence the
# test wants. That is the test dictating the prompt, which is backwards.
#
# What is worth pinning is what a reader cannot check by reading: structure,
# declarations, reach, the disjoint tool halves below. Whether guidance reads
# well is a judgement, and judgement belongs to whoever is editing it.


# ──────────────────────────────────────────────────────────────────────
# The tools. Two halves of one folder, with disjoint reach.
# ──────────────────────────────────────────────────────────────────────

def test_the_tool_cannot_be_handed_a_path():
    """The confinement is structural, not a check that could be bypassed.

    The tool takes a *name* and derives the path from it, so there is no
    argument that escapes ``workspace/memory`` and none that names MEMORY.md.
    That is what makes it safe to give a subagent nobody is watching — and a
    ``path`` parameter appearing later would quietly undo it.

    This now guards the writer as much as the reader. When they were two tools
    the writer's inability to read was offered as the safety property; it never
    was one, and this is.
    """
    properties = set(_load_store_class(MEMORY, "Memory").parameters["properties"])
    assert "path" not in properties
    assert "name" in properties
    assert not (properties & {"folder", "file", "filename"})


def test_a_name_cannot_be_a_path():
    """The character set is the check, and it is the load-bearing one.

    Lowercase alphanumerics and hyphens is the agentskills.io rule, and it also
    happens to exclude every way of writing a traversal: a dot, a separator, a
    drive letter, a leading tilde.

    It guards every write as well as every read, which it did not have to when
    the writer was a separate file. ``MEMORY.md`` and ``note.md`` are refused
    twice over: they are not legal names, and no legal name produces them.
    """
    namespace = {}
    exec(compile(_source_or_skip(MEMORY), MEMORY, "exec"), namespace)
    valid = namespace["_valid_name"]

    assert valid("retry-failed-uploads")
    assert valid("pdf2text")
    for bad in ("../escape", "a/b", r"a\b", "C:name", "~", "UPPER",
                "-lead", "trail-", "a--b", "", "x" * 65, "with space",
                "MEMORY.md", "note.md"):
        assert not valid(bad), f"{bad!r} must be refused"


def test_one_tool_reads_and_writes_and_nothing_it_is_handed_becomes_a_path():
    """The union is deliberate, and it is safe for a reason that is not a split.

    This replaces ``test_the_writer_cannot_read_and_the_reader_cannot_write``,
    which pinned that ``memory_curate`` declared no ``fs.read``. That property
    was retired knowingly. It read as defence in depth and was not: the curator
    profile also grants ``read_file``, ``grep`` and three search tools, so a
    writer without ``fs.read`` was denied nothing it could not get one tool
    over — while the workaround it forced, restating a description on every
    update, was a real cost paid every time an entry was revised. And it never
    guarded the folder in the first place, because everything under
    ``workspace/`` is a standing free-write grant: ``fs.write`` there raises no
    dialog whether or not the same file can read.

    What does the guarding is that no argument ever becomes a path. Every
    ``sdk.fs`` call and every ``sdk.path.join`` takes a value derived from
    ``_memory_root``; ``kwargs`` reaches the document body and a description
    string, and nothing else. Checked structurally, because this is the
    property a future action could quietly break.
    """
    import ast

    declared = set(_declarations(MEMORY)["requests"])
    assert {"fs.read", "fs.write", "fs.delete"} <= declared

    source = _source_or_skip(MEMORY)
    tree = ast.parse(source)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        # ``sdk.fs.<verb>(path, ...)`` and ``sdk.path.join(root, ...)`` — the
        # first argument is the one that decides what gets touched.
        owner = func.value
        is_fs = (isinstance(owner, ast.Attribute) and owner.attr == "fs")
        is_join = (func.attr == "join" and isinstance(owner, ast.Attribute)
                   and owner.attr == "path")
        if not (is_fs or is_join):
            continue
        first = ast.get_source_segment(source, node.args[0]) or ""
        assert "kwargs" not in first, f"a model-supplied value reached {first!r}"
        checked += 1
    assert checked > 10, "the walk found nothing, so it is not proving anything"


def test_the_two_paths_a_name_can_mean_are_the_only_two():
    """``MEMORY.md`` is unreachable by computation, not by grep.

    One function produces every path this tool can touch, for reading and for
    writing alike, and ``_existing`` goes through it — so the write side cannot
    grow a third template without this test seeing it. Together with the
    character set, that is the whole confinement: no legal name produces
    ``MEMORY.md``, and ``MEMORY.md`` is not a legal name.
    """
    import ast

    source = _source_or_skip(MEMORY)
    namespace = {}
    exec(compile(source, MEMORY, "exec"), namespace)

    class _Sdk:
        paths = type("P", (), {"get": staticmethod(lambda key: "/w")})()
        path = type("Q", (), {"join": staticmethod(lambda *p: "/".join(p))})()

    assert namespace["_paths_for"](_Sdk(), "x") == [
        ("/w/memory/notes/x.md", "note"),
        ("/w/memory/skills/x/SKILL.md", "skill"),
    ]

    # And the writer's probe resolves through it rather than rebuilding it.
    tree = ast.parse(source)
    existing = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "_existing")
    assert any(getattr(n.func, "id", "") == "_paths_for"
               for n in ast.walk(existing) if isinstance(n, ast.Call)), (
        "_existing must resolve through _paths_for, not its own path list")


def test_update_keeps_the_description_it_was_not_given():
    """The capability the merge bought, and the pin that keeps it.

    A revision that only fixes a body should not have to restate a sentence it
    just read; requiring one was a workaround for a tool that could not read.
    The body is still a full replacement — so this cannot reintroduce the
    "partial edit against a file the model has not seen" problem, since the
    model still supplies a whole entry.

    An entry with no description anywhere is still refused: it would rank and
    never once be offered, which looks exactly like having no memory at all.
    """
    source = _source_or_skip(MEMORY)
    update = source.split("def _update", 1)[1].split("\n    def ", 1)[0]

    assert "_supplied_description(kwargs)" in update
    assert "_stored_description(sdk, path)" in update, "it must read the old one"
    assert "needs a description" in update, "and still refuse when there is none"

    # The two directions a description travels are named apart. One function
    # called ``_description`` for both is how a model-supplied string ends up
    # somewhere only a stored one belongs.
    assert "def _stored_description(self, sdk, path)" in source
    assert "def _supplied_description(self, kwargs)" in source


def test_one_tool_five_actions_and_only_action_is_required():
    """Two tools became five actions on one, which is one declaration.

    ``required`` cannot express "a name unless you are listing", so the schema
    declares the floor and ``run`` enforces the rest. Stating it in JSON Schema
    with ``if``/``then`` was available and refused: nothing in the kernel reads
    those keywords and weak models handle them badly.
    """
    parameters = _load_store_class(MEMORY, "Memory").parameters

    assert parameters["required"] == ["action"]
    assert set(parameters["properties"]["action"]["enum"]) == {
        "read", "list", "create", "update", "delete"}
    assert {"name", "kind", "description", "body", "narration"} <= set(
        parameters["properties"])


def test_list_is_answered_before_a_name_is_required():
    """The conditional lives in one place, and its order is the whole of it.

    ``list`` takes no name, so it has to be answered before the name check —
    otherwise the one action that needs no argument fails without one.
    """
    source = _source_or_skip(MEMORY)
    body = source.split("def run(self, sdk", 1)[1].split("\n    def ", 1)[0]

    assert body.index('action == "list"') < body.index("_valid_name")
    assert "NAMED_ACTIONS" in body


def test_reading_a_skill_points_at_its_resources_without_loading_them():
    """Progressive disclosure done by the tool rather than by prose.

    The spec's whole structure is SKILL.md plus files loaded on demand. Listing
    the resource paths on read means the agent never has to guess a path or
    list a directory, and never pays for a reference it does not open.
    """
    source = _source_or_skip(MEMORY)
    assert 'SKILL_RESOURCE_DIRS = ("references", "scripts", "assets")' in source
    assert "_resources" in source


# ──────────────────────────────────────────────────────────────────────
# The usage table: one place, two writers of one fact each and a reader.
# ──────────────────────────────────────────────────────────────────────

def test_offered_and_recalled_are_one_table():
    """The service offers, the memory tool takes.

    Splitting these was the old design's expense: the offer lived in one table
    and the *take* had to be reconstructed by parsing every assistant message
    for a ``read_file`` call and normalizing the path it named. A dedicated
    tool records the take directly, which is the whole reason it exists.
    """
    service = _source_or_skip(SERVICE)
    tool = _source_or_skip(MEMORY)

    # The service defines it and inserts the offer.
    assert "CREATE TABLE IF NOT EXISTS memory_usage" in service
    assert "recalled_at" in service
    assert "db.define" in _declarations(SERVICE)["requests"]

    # The tool fills it in, or records a recall nobody offered. That the same
    # tool also writes *entries* changes nothing about this table.
    assert "SET recalled_at = ?" in tool
    assert "INSERT INTO memory_usage" in tool


def test_recalls_survive_pruning_because_they_are_the_data():
    """Offers are the volume and are pruned; recalls are rare, small, and kept.

    Which entries earn their place, over time, is the input to any future pass
    over what nobody has recalled in months. The sweep moved into the service
    when the curator task that used to run it was retired, and it must still
    touch only offers nobody took.
    """
    service = _source_or_skip(SERVICE)

    assert "DELETE FROM memory_usage" in service
    assert "WHERE recalled_at IS NULL AND offered_at < ?" in service


# ──────────────────────────────────────────────────────────────────────
# Retrieval stands at one moment, and raises no dialog there.
# ──────────────────────────────────────────────────────────────────────

def test_retrieval_stands_at_the_one_moment_that_runs_per_turn():
    """``turn_start``, not ``llm_call`` — the distinction is the latency floor.

    A hook at ``llm_call`` would re-run the search on every model call within a
    turn, for a query that only changed once. ``turn_start`` fires once per
    logical turn, which is exactly the granularity of "what is this message
    about".
    """
    declared = _declarations(SERVICE)
    assert declared["family"] == "service"
    assert declared["name"] == "memory_retrieve"
    assert declared["hooks"] == {"turn_start": "on_turn_start",
                                 "end_turn": "on_end_turn"}
    # It contributes guidance but exposes no callable surface: nothing should
    # be reaching into memory through ``service.call``.
    assert declared["exports"] == []


def test_the_service_can_inject_and_can_search():
    """The two Requests the read half cannot work without.

    ``session.add_prompt_extra`` is how pointers reach the prompt at all, and
    ``tool.call`` is how the search happens — the service deliberately owns no
    retrieval of its own, so that installing a better search tool improves
    memory without touching this file.
    """
    declared = _declarations(SERVICE)
    assert "session.add_prompt_extra" in declared["requests"]
    assert "tool.call" in declared["requests"]
    # What it calls, and what the agent writes with when nudged.
    assert {"tools/tool_hybrid_search.py",
            "tools/tool_memory.py"} <= set(declared["dependencies_files"])


def test_injecting_memory_pointers_raises_no_dialog():
    """The whole design fails if retrieval interrupts the turn it serves.

    ``session.add_prompt_extra`` used to be ALWAYS_UNSAFE, which would have put
    an approval dialog in front of every single turn. It is now safe for the
    caller's own session and unsafe only when it names somebody else's, which
    is the property this test exists to keep.
    """
    from sandbox import Chain, Request
    from sandbox.guest.requests import SESSION_ADD_PROMPT
    from sandbox.policy import CONSEQUENTIAL, classify

    assert SESSION_ADD_PROMPT not in CONSEQUENTIAL
    # The chain a *hook* actually has. This test used to classify a bare
    # ``Chain(root="repl")``, which is not what stands at a doorway, and — far
    # worse — omitted ``key``, which is the argument the whole branch turns on.
    # It passed for a call shape the service did not make while the real call
    # was refused on every turn.
    hook_chain = Chain(root="service:memory_retrieve",
                       links=("memory_retrieve",))
    decision = classify(
        Request(SESSION_ADD_PROMPT, {"text": "pointers", "slot": "memory"}),
        hook_chain)
    assert decision.safe, "injecting into its own session must never ask"

    # And the reason the service must not name a session: from this chain,
    # naming one is unsafe, and a hook is unattended, so it is refused outright
    # rather than asked. The source is checked because the classification alone
    # cannot say which of the two calls the plugin makes.
    named = classify(
        Request(SESSION_ADD_PROMPT,
                {"text": "pointers", "slot": "memory", "key": "repl"}),
        hook_chain)
    assert not named.safe
    source = _source_or_skip(SERVICE)
    assert "add_prompt(block, slot=" in source, "it must not name a session"
    assert "key=ctx.session_key" not in source


def test_the_service_writes_its_kernel_setting_only_at_install():
    """It needs one setting, and there is exactly one moment it may ask.

    ``sync_directories`` must contain the memory folder or nothing is indexed.
    It was attempted from ``start`` — refused, because a service has no
    session and an unattended unsafe Request is refused rather than asked — and
    then from the ``turn_start`` hook, which was made to work and then reverted
    because it asked at the moment furthest from anything the user chose to do.

    ``on_install`` is the moment that works: it runs under the chain of the
    ``/packages`` command the person typed, which is attended, so the write can
    be *asked* about instead of refused. What this pins is that the capability
    stays there — a ``config.write`` from ``start`` or from the hook would be
    the reverted design creeping back, and it would fail silently, which is how
    it survived so long the first two times.
    """
    import ast

    declared = _declarations(SERVICE)
    assert "config.write" in declared["requests"]
    assert "config.read" in declared["requests"]

    source = _source_or_skip(SERVICE)
    tree = ast.parse(source)
    writers = {
        node.name for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(inner, ast.Attribute) and inner.attr == "write"
                and isinstance(inner.value, ast.Attribute)
                and inner.value.attr == "config"
                for inner in ast.walk(node))
    }
    assert writers, "something has to do the seeding"
    # Named helpers are fine; being reachable from anything but on_install is
    # not. Nothing in the runtime path may hold this capability.
    for banned in ("start", "stop", "on_turn_start", "on_end_turn",
                   "on_uninstall"):
        assert banned not in writers, f"{banned} writes config"

    for method in ("on_install", "on_uninstall"):
        assert f"def {method}(self, sdk)" in source, method


# ──────────────────────────────────────────────────────────────────────
# The nudge: once, at the end of a clean turn, and never recorded.
# ──────────────────────────────────────────────────────────────────────

class _Ending:
    def __init__(self, reason="model_finished", doorman_fires=0):
        self.reason = reason
        self.doorman_fires = doorman_fires
        self.final_text = "done"


class _Ctx:
    def __init__(self, session_key="repl"):
        self.session_key = session_key


def _nudge(ending, ctx=None):
    service = _load_store_class(SERVICE, "MemoryRetrieve")()
    return service.on_end_turn(None, ctx or _Ctx(), ending)


def test_a_clean_finish_is_sent_back_once_with_an_ephemeral_note():
    """Always, not on a judgement — the agent that lived the turn decides.

    Ephemeral because a transcript with this line after every reply would be
    the conversation talking to itself; the model sees it, history does not.
    """
    verdict = _nudge(_Ending())
    assert type(verdict).__name__ == "SendBack"
    assert verdict.ephemeral
    assert verdict.allow_tools, "saving a memory is a tool call"
    assert verdict.quiet, "the comeback must never answer the user twice"
    assert "`memory`" in verdict.note

    # And ``quiet`` survives the crossing, which is spelled out field by field.
    from sandbox.guest.hooks import unwrap
    from sandbox.hooks import rebuild
    assert rebuild("end_turn", unwrap(verdict)).quiet


def test_the_nudge_names_the_memories_opened_this_turn_so_stale_ones_get_fixed():
    """A memory read and then contradicted is the one moment its staleness is
    visible. The nudge names what was opened since the turn began — asked by
    time, since ``recalled_at`` is stamped on every read — and asks for a fix
    or a delete. A turn that opened nothing gets the plain nudge."""
    service = _load_store_class(SERVICE, "MemoryRetrieve")()
    asked = []

    class _Db:
        def query(self, sql, params, max_rows=None):
            asked.append((sql, params))
            return [{"name": "pdf-yields-no-text"}]

    sdk = type("Sdk", (), {"db": _Db(), "Failed": _Failed,
                           "log": lambda *a, **k: None})()
    ctx = _Ctx()
    ctx.conversation_id = 7
    service._turn_began = {"repl": 100.0}

    note = service.on_end_turn(sdk, ctx, _Ending()).note
    assert "pdf-yields-no-text" in note
    assert "`memory update`" in note and "`memory delete`" in note
    assert asked and asked[0][1] == [7, 100.0]

    # The turn's start is consumed, so a second ending asks nothing.
    assert "You opened" not in service.on_end_turn(sdk, ctx, _Ending()).note


def test_a_turn_that_finds_nothing_clears_the_previous_turns_pointers():
    """Overlays persist until their slot is rewritten, so abstaining in
    silence left last turn's list in the prompt, pointing at a situation that
    had passed. Every empty turn removes the slot instead."""
    service = _load_store_class(SERVICE, "MemoryRetrieve")()
    removed = []

    class _Session:
        def remove_prompt(self, handle, key=""):
            removed.append(handle)

        def add_prompt(self, *a, **k):
            raise AssertionError("nothing to add")

    class _Config:
        def read(self, key):
            return 0  # retrieval off: the simplest route to an empty turn

    sdk = type("Sdk", (), {"session": _Session(), "config": _Config(),
                           "Failed": _Failed, "log": lambda *a, **k: None})()
    service.on_turn_start(sdk, _Ctx(), None)
    assert removed == ["memory"]
    assert "session.remove_prompt_extra" in _declarations(SERVICE)["requests"]


def test_the_nudge_never_stacks_and_skips_what_is_not_a_clean_finish():
    """Its own second visit, another doorman's note, a budget wrap-up and a
    subagent's turn all pass straight through. The first is what stops the
    nudge being asked again after the agent answers it."""
    assert _nudge(_Ending(doorman_fires=1)) is None
    assert _nudge(_Ending(reason="budget_exhausted")) is None
    assert _nudge(_Ending(), _Ctx("spawn_subagent:12")) is None


def test_the_curator_task_is_gone():
    """Replaced by the nudge, and deliberately not left installable beside it:
    two writers reflecting on one conversation would disagree about it."""
    worktree = store_worktree()
    if worktree is None:
        pytest.skip("no store worktree")
    assert not (Path(worktree) / "tasks/task_memory_curate.py").exists()


# ──────────────────────────────────────────────────────────────────────
# The manifest.
# ──────────────────────────────────────────────────────────────────────

def test_the_manifest_names_the_two_and_lets_the_closure_do_the_rest():
    """A manifest is what this package *is*, not what it needs on the way.

    It listed all sixteen files for a while, which installed correctly and
    uninstalled catastrophically: uninstall follows the dependency edge
    backwards, so anything the manifest names is the bundle's to remove, and
    naming ``service_embed`` and the four indexing tasks meant removing memory
    tore out the machine's whole text index — plus torch — and took
    ``read_file``, ``grep``, ``glob`` and ``sql_query`` with it.

    Duplicating the closure by hand also hid seven missing declarations: with
    the manifest supplying them, nothing ever noticed that
    ``tool_lexical_search`` never named the task that fills the index it reads.
    Install walks the edges either way, so the manifest carrying them bought
    nothing and cost the ability to leave.
    """
    worktree = store_worktree()
    if worktree is None:
        pytest.skip("no store worktree to read the manifest from")
    path = Path(worktree) / BUNDLE
    if not path.exists():
        pytest.skip(f"{BUNDLE} is not present on the local store worktree")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["name"] and manifest["description"]
    files = manifest["files"]
    assert files == sorted(files), "manifest files must stay sorted"
    assert set(files) == set(SUITE), (
        "the manifest is the two memory plugins; anything else they need is "
        "reached through dependencies_files")

    # Everything the bundle installs is still everything it needs, reached the
    # way the package manager reaches it.
    closure = _install_closure()
    assert "tools/tool_hybrid_search.py" in closure
    # read_file is for a skill's own references, which the memory tool names
    # but deliberately does not load.
    assert "tools/tool_read_file.py" in closure
    # Memory writes through ``memory`` alone; a general file editor arriving
    # with it would be a capability nobody installing memory asked for.
    assert "tools/tool_edit_file.py" not in closure

    for relative in closure:
        assert (Path(worktree) / relative).exists(), relative


# ──────────────────────────────────────────────────────────────────────
# Telling the user when memory changed under them.
# ──────────────────────────────────────────────────────────────────────

class _Failed(Exception):
    pass


class _Sdk:
    """Enough SDK to drive ``_notify``. Every namespace here is one Request."""

    Failed = _Failed

    def __init__(self, attended=False, setting=True):
        self._attended = attended
        self._setting = setting
        #: Every ``session.push`` this tool made, as ``(message, kwargs)``.
        self.pushed = []
        self.logs = []
        self.session = self._Session(self)
        self.config = self._Config(self)

    def log(self, message, level="info"):
        self.logs.append((level, message))

    class _Session:
        def __init__(self, sdk):
            self._sdk = sdk

        def get(self, key=None):
            if self._sdk._attended == "unreadable":
                raise _Failed("no runtime")
            return {"key": "k", "attended": self._sdk._attended}

        def push(self, message, key="", **kwargs):
            self._sdk.pushed.append((message, kwargs))

    class _Config:
        def __init__(self, sdk):
            self._sdk = sdk

        def read(self, key):
            if self._sdk._setting == "unreadable":
                raise _Failed("no config")
            return self._sdk._setting


def _memory():
    """The real ``Memory`` tool, loaded the way a box loads it.

    Named for the tool rather than for the act: ``MemoryCurate`` is the *task*
    class now, so a helper called ``_curate`` returning a tool would be
    actively misleading.
    """
    return _load_store_class(MEMORY, "Memory")()


def test_a_background_write_is_announced_as_a_notification():
    """The curator writes where nobody is looking; this is the only trace.

    A *notification*, not a chat message. This used to emit
    ``chat_message_pushed`` by literal channel name — reaching around
    ``session.push`` to a bus channel the tool does not own, for the one thing
    that channel offered and the Request did not: a ``source`` field. Nothing
    read it, so the note arrived as an ordinary line of chat anyway.
    """
    tool = _memory()
    sdk = _Sdk(attended=False)

    tool._notify(sdk, "create", "retry-failed-uploads")

    assert len(sdk.pushed) == 1
    message, kwargs = sdk.pushed[0]
    assert message == "retry-failed-uploads"
    assert kwargs["title"] == "Memory created"
    assert kwargs["notify"] is True
    # No ``key``: that targets one session, and the kernel broadcasts without
    # it. The write happened in a session with no person on it, so there is
    # nothing to reply *to* — the note goes wherever the user actually is.
    assert not kwargs.get("key")


def test_the_tool_does_not_state_its_own_source():
    """Attribution is the kernel's to stamp, off the provenance chain.

    The old emit passed ``"source": "memory"`` in the payload, which the new
    design specifically does not allow: a plugin that can name its own source
    can claim to be the plugin watcher. Pinned as a negative because the
    failure is silent — a forged source looks exactly like a true one.
    """
    tool = _memory()
    sdk = _Sdk(attended=False)

    tool._notify(sdk, "create", "x")

    assert "source" not in sdk.pushed[0][1]


def test_every_action_has_its_own_word():
    tool = _memory()
    for action, expected in (("create", "created"), ("update", "updated"),
                             ("delete", "deleted")):
        sdk = _Sdk(attended=False)
        tool._notify(sdk, action, "x")
        assert sdk.pushed[0][1]["title"] == f"Memory {expected}"


def test_a_write_the_user_watched_is_not_announced():
    """They asked for it in conversation and the reply already says so.

    ``attended`` rather than "is this a subagent" because the kernel already
    owns that question, and a concurrent multi-user frontend can override it
    per session — any guess made from a session key would be wrong there.
    """
    tool = _memory()
    sdk = _Sdk(attended=True)

    tool._notify(sdk, "create", "x")

    assert sdk.pushed == []


def test_the_setting_turns_it_off_and_defaults_on():
    tool = _memory()

    off = _Sdk(attended=False, setting=False)
    tool._notify(off, "create", "x")
    assert off.pushed == []

    unset = _Sdk(attended=False, setting=None)
    tool._notify(unset, "create", "x")
    assert len(unset.pushed) == 1, "unset means default, and the default is on"


def test_the_two_unreadable_cases_fail_in_opposite_directions():
    """A spurious notification is worse than a missing one; a missing setting
    is not a refusal."""
    tool = _memory()

    blind = _Sdk(attended="unreadable")
    tool._notify(blind, "create", "x")
    assert blind.pushed == [], "not knowing where we are means staying quiet"

    no_config = _Sdk(attended=False, setting="unreadable")
    tool._notify(no_config, "create", "x")
    assert len(no_config.pushed) == 1, "the default is on"


def test_announcing_can_never_fail_the_write():
    """The entry is already on disk by the time this runs, so raising here
    would report an error for something that fully succeeded."""
    tool = _memory()
    sdk = _Sdk(attended=False)
    sdk.session.push = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bus"))

    tool._notify(sdk, "create", "x")     # must not raise

    assert any(level == "debug" for level, _ in sdk.logs)


def test_only_the_three_writes_announce():
    """The helper is only worth having if every write reaches it — and if no
    read does.

    Read off the source, because a missed call site is invisible from the
    helper's own tests: the notification simply never happens for that one
    action. The negative half is a hazard the merge created. Reads and writes
    share a class now, so a ``_notify`` in ``_read`` or ``_list`` would put a
    line in the user's chat every time the agent opened a memory — which is
    noise that would get the setting turned off, taking the announcements that
    matter with it.
    """
    import ast

    tree = ast.parse(_source_or_skip(MEMORY))

    def notifies(method):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == method)
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "_notify"]

    for method, action in (("_create", "create"), ("_update", "update"),
                           ("_delete", "delete")):
        calls = notifies(method)
        assert len(calls) == 1, f"{method} does not announce exactly once"
        assert calls[0].args[1].value == action, f"{method} announces the wrong action"

    for method in ("_read", "_list", "_rendered", "_entries"):
        assert not notifies(method), f"{method} is a read and must stay silent"
