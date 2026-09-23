"""What the kernel reads off the files that make up memory.

Same shape as ``test_store_attachment_tools``: kernel invariants that happen to
be *about* store files. The subject is the kernel's own verdict — does this
load, are these Requests real, is the retrieval free of dialogs, can this reach
outside the folder — and the store file is the input.

The two matter together because each is useless alone.
``service_memory_retrieve`` asks Jev which entries fit at ``turn_start`` — a
shortlist over every description, then each shortlisted entry's own
``when_to_retrieve`` questions — injects descriptions and records what it
offered, and at ``end_turn`` sends the agent
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

    Both kinds compete in one shortlist question and render as one line, which
    differs by a label and nothing else. Separate pools per kind would be the
    separation this design exists to remove.
    """
    sdk = _FakeSdk()
    sdk.add_note("stuck-upload", "An upload hangs", {"q": "Is an upload stuck?"})
    sdk.add_skill("deploy-ui", "Deploying the web UI", {"q": "Is the user deploying?"})
    block = _turn(sdk)

    criteria = sdk.choice_calls[0]["shortlist"]["criteria"]
    assert set(criteria) == {"stuck-upload", "deploy-ui"}
    assert "- stuck-upload — An upload hangs" in block
    assert "- deploy-ui (skill) — Deploying the web UI" in block


def test_the_prompt_carries_descriptions_and_not_the_entries():
    """Inlining the body destroys the signal the whole loop runs on.

    With the content already in the prompt there is no reason to recall
    anything, so nothing downstream can tell which entries were used.
    """
    sdk = _FakeSdk()
    sdk.add_note("stuck-upload", "An upload hangs", {"q": "Is an upload stuck?"},
                 body="SECRET BODY TEXT")
    block = _turn(sdk)
    assert "An upload hangs" in block
    assert "SECRET BODY TEXT" not in block


def test_the_block_says_how_much_it_is_not_showing():
    """A short list with no total reads as "this is all you have"."""
    sdk = _FakeSdk()
    sdk.add_note("stuck-upload", "An upload hangs", {"q": "Is an upload stuck?"})
    sdk.add_note("slow-search", "Search is slow", {"q": "Is search slow?"})
    sdk.noul = {"slow-search::q": 0.1}
    block = _turn(sdk)
    assert "Showing 1 of 2" in block


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
    assert 'stored["description"]' in update, "it must read the old one"
    assert "needs a description" in update, "and still refuse when there is none"

    # The two directions a description travels are named apart. One function
    # called ``_description`` for both is how a model-supplied string ends up
    # somewhere only a stored one belongs.
    assert "def _stored(self, sdk, path)" in source
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


def test_the_service_can_inject_and_can_ask_jev():
    """The two Requests the read half cannot work without.

    ``session.add_prompt_extra`` is how pointers reach the prompt at all, and
    ``service.call`` is how Jev is asked. ``tool.call`` is gone with
    ``hybrid_search``: memory is not the user's corpus and needs no index.
    """
    declared = _declarations(SERVICE)
    assert "session.add_prompt_extra" in declared["requests"]
    assert "service.call" in declared["requests"]
    assert "tool.call" not in declared["requests"]
    deps = set(declared["dependencies_files"])
    assert {"services/service_rlcd.py", "tools/tool_memory.py"} <= deps
    assert "tools/tool_hybrid_search.py" not in deps


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


def test_the_service_writes_no_kernel_setting():
    """Nothing to index means nothing to seed.

    It used to add the memory folder to ``sync_directories`` from
    ``on_install`` so ``hybrid_search`` could find entries. Retrieval reads
    the folder directly now, so the service holds no ``config.write`` at all —
    and a capability it does not need is one that cannot be misused from an
    unattended hook.
    """
    declared = _declarations(SERVICE)
    assert "config.write" not in declared["requests"]
    source = _source_or_skip(SERVICE)
    assert "def on_install(self, sdk)" not in source
    assert "def on_uninstall(self, sdk)" in source


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
    assert "services/service_rlcd.py" in closure
    # No index, no embedder: memory is not the user's corpus.
    assert "tools/tool_hybrid_search.py" not in closure
    assert "services/service_embed.py" not in closure
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


# ──────────────────────────────────────────────────────────────────────
# Retrieval: a Jev shortlist, then each entry's own questions.
# ──────────────────────────────────────────────────────────────────────

class _Path:
    @staticmethod
    def join(*parts):
        return "/".join(parts)

    @staticmethod
    def normalize(path):
        return path

    @staticmethod
    def stem(name):
        return name.rsplit("/", 1)[-1].rsplit(".", 1)[0]


class _FakeSdk:
    """Just enough SDK to drive a turn, with Jev scripted.

    ``choice`` maps an entry name to a weight (uniform when absent), ``noul``
    maps a namespaced question id to its probability (0.9 when absent).
    """

    Failed = _Failed

    def __init__(self, config=None):
        self.config_values = {"memory_max_pointers": 3, "memory_candidates": 5,
                              "memory_question_threshold": 0.5, **(config or {})}
        self.files = {}
        self.choice = {}
        self.noul = {}
        self.choice_calls, self.gate_calls = [], []
        self.jev_down = False
        self.logs, self.prompts, self.removed, self.writes = [], [], [], []
        self.user_row = {"id": 10, "content": "my upload is stuck"}
        self.context_rows = []
        self.paths = type("P", (), {"get": staticmethod(lambda key: "/w")})()
        self.path = _Path()
        sdk = self
        self.config = type("C", (), {
            "read": staticmethod(lambda key: sdk.config_values.get(key))})()
        self.fs = type("F", (), {"list": staticmethod(self._list),
                                 "read": staticmethod(self._read),
                                 "write": staticmethod(lambda p, t: None)})()
        self.db = type("D", (), {"query": staticmethod(self._query),
                                 "write": staticmethod(
                                     lambda sql, params=None: sdk.writes.append(params)),
                                 "define": staticmethod(lambda sql: None)})()
        self.services = type("S", (), {"call": staticmethod(self._call)})()
        self.session = type("Z", (), {
            "add_prompt": staticmethod(
                lambda text, slot=None: sdk.prompts.append(text)),
            "remove_prompt": staticmethod(lambda slot: sdk.removed.append(slot)),
        })()

    def log(self, message, level="info"):
        self.logs.append((level, message))

    def _entry(self, name, description, questions, body):
        lines = ["---", f"name: {name}", f"description: {description}"]
        if questions:
            lines.append("when_to_retrieve: " + json.dumps(
                {qid: {"type": "noul", "instructions": text}
                 for qid, text in questions.items()}))
        return "\n".join(lines) + "\n---\n\n" + body + "\n"

    def add_note(self, name, description, questions=None, body="Do the thing."):
        self.files[f"/w/memory/notes/{name}.md"] = self._entry(
            name, description, questions, body)

    def add_skill(self, name, description, questions=None, body="Step one."):
        self.files[f"/w/memory/skills/{name}/SKILL.md"] = self._entry(
            name, description, questions, body)

    def _list(self, path, pattern=None, details=False):
        prefix = path.rstrip("/") + "/"
        children = {}
        for full in self.files:
            if full.startswith(prefix):
                rest = full[len(prefix):]
                head, _, tail = rest.partition("/")
                children[head] = bool(tail)
        if not children:
            raise _Failed(f"not found: {path}")
        if pattern == "*.md":
            children = {k: v for k, v in children.items() if k.endswith(".md")}
        return [{"name": k, "is_dir": v} for k, v in sorted(children.items())]

    def _read(self, path):
        if path not in self.files:
            raise _Failed(f"not found: {path}")
        return self.files[path]

    def _query(self, sql, params=None, max_rows=None):
        if "SELECT id, content" in sql:
            return [self.user_row] if self.user_row else []
        if "SELECT role, content, author" in sql:
            return list(self.context_rows)
        return []

    def _call(self, service, method, *args):
        assert (service, method) == ("rlcd", "evaluate")
        if self.jev_down:
            raise _Failed("rlcd is not loaded")
        state, questions = args
        answers = {}
        if "shortlist" in questions:
            self.choice_calls.append(questions)
            options = questions["shortlist"]["criteria"]
            weights = {n: self.choice.get(n, 1.0) for n in options}
            total = sum(weights.values())
            answers["shortlist"] = {"type": "choice", "probabilities":
                                    {n: w / total for n, w in weights.items()}}
        else:
            self.gate_calls.append(questions)
            for qid in questions:
                answers[qid] = {"type": "noul", "noul": self.noul.get(qid, 0.9)}
        self.last_state = state
        return {"ok": True, "answers": answers}


class _TurnCtx:
    session_key = "repl"
    conversation_id = 7


def _turn(sdk, service=None):
    service = service or _load_store_class(SERVICE, "MemoryRetrieve")()
    before = len(sdk.prompts)
    service.on_turn_start(sdk, _TurnCtx(), {})
    return sdk.prompts[-1] if len(sdk.prompts) > before else ""


def test_an_entry_fires_only_when_every_question_clears_the_threshold():
    """The gate is the *minimum*: one confident no vetoes two confident yeses."""
    sdk = _FakeSdk()
    sdk.add_note("alpha", "Alpha situation", {"a": "A?", "b": "B?", "c": "C?"})
    sdk.add_note("bravo", "Bravo situation", {"a": "A?", "b": "B?"})
    sdk.noul = {"alpha::a": 0.9, "alpha::b": 0.9, "alpha::c": 0.4,
                "bravo::a": 0.9, "bravo::b": 0.8}
    block = _turn(sdk)
    assert "bravo" in block and "alpha" not in block
    # The usage row carries the probability that let it through.
    offered = [row for row in sdk.writes if row and row[0] == "bravo"]
    assert offered and offered[0][-1] == 0.8


def test_the_threshold_and_the_shortlist_size_are_settings():
    sdk = _FakeSdk({"memory_question_threshold": 0.3, "memory_candidates": 1})
    sdk.add_note("alpha", "Alpha situation", {"a": "A?"})
    sdk.add_note("bravo", "Bravo situation", {"a": "A?"})
    sdk.choice = {"alpha": 3.0, "bravo": 1.0}
    sdk.noul = {"alpha::a": 0.4}
    block = _turn(sdk)
    # Only the top candidate was asked its questions, and 0.4 clears 0.3.
    assert set(sdk.gate_calls[0]) == {"alpha::a"}
    assert "alpha" in block and "bravo" not in block


def test_the_prompt_is_capped_at_the_pointer_setting():
    sdk = _FakeSdk({"memory_max_pointers": 2, "memory_candidates": 10})
    for i in range(10):
        sdk.add_note(f"entry-{i}", f"Situation {i}", {"q": "Q?"})
    block = _turn(sdk)
    assert block.count("\n- entry-") == 2


def test_one_gate_request_asks_every_candidate_with_namespaced_ids():
    """Two entries may both call a question ``q``; the ids must not collide."""
    sdk = _FakeSdk()
    sdk.add_note("alpha", "Alpha situation", {"q": "A?"})
    sdk.add_note("bravo", "Bravo situation", {"q": "B?"})
    _turn(sdk)
    assert len(sdk.gate_calls) == 1
    assert set(sdk.gate_calls[0]) == {"alpha::q", "bravo::q"}


def test_the_shortlist_question_is_identical_turn_to_turn():
    """Sorted options keep the request byte-stable, which is what caches."""
    sdk = _FakeSdk()
    for name in ("zulu", "alpha", "mike"):
        sdk.add_note(name, f"{name} situation", {"q": "Q?"})
    service = _load_store_class(SERVICE, "MemoryRetrieve")()
    _turn(sdk, service)
    _turn(sdk, service)
    first, second = sdk.choice_calls
    assert first == second
    assert list(first["shortlist"]["criteria"]) == ["alpha", "mike", "zulu"]


def test_a_corpus_past_one_choice_question_is_chunked_then_compared():
    """Probabilities from two questions do not compare, so winners meet again."""
    sdk = _FakeSdk()
    for i in range(300):
        sdk.add_note(f"entry-{i:03d}", f"Situation {i}", {"q": "Q?"})
    _turn(sdk)
    sizes = [len(call["shortlist"]["criteria"]) for call in sdk.choice_calls]
    assert sizes == [255, 45, 10]


def test_an_unquestioned_entry_needs_a_strong_shortlist_win():
    """Entries from before ``when_to_retrieve`` still work while migrating."""
    sdk = _FakeSdk()
    sdk.add_note("old", "Old situation")
    sdk.add_note("other", "Other situation")
    sdk.add_note("third", "Third situation")
    block = _turn(sdk)  # a three-way tie, 0.33 each: too weak
    assert block == "" and sdk.removed == ["memory"]
    assert sdk.gate_calls == [], "nothing had questions, so no gate request"

    sdk.choice = {"old": 8.0}
    assert "old" in _turn(sdk)


def test_jev_being_down_offers_nothing_and_says_so_once():
    sdk = _FakeSdk()
    sdk.add_note("alpha", "Alpha situation", {"q": "A?"})
    sdk.jev_down = True
    service = _load_store_class(SERVICE, "MemoryRetrieve")()
    assert _turn(sdk, service) == "" and _turn(sdk, service) == ""
    assert sdk.removed == ["memory", "memory"], "the old list must be cleared"
    warnings = [m for level, m in sdk.logs if level == "warning" and "rlcd" in m]
    assert len(warnings) == 1


def test_the_state_is_the_request_plus_the_agent_turn_without_tool_results():
    sdk = _FakeSdk()
    sdk.add_note("alpha", "Alpha situation", {"q": "A?"})
    call = {"function": {"name": "edit_file", "arguments": json.dumps(
        {"path": "a.py", "new_text": "x" * 5000,
         "narration": "fixing the upload retry"})}}
    # Newest first, as the query returns them.
    sdk.context_rows = [
        {"role": "assistant", "content": "Retried it for you.", "author": None},
        {"role": "tool", "content": "TOOL RESULT PAYLOAD", "author": None},
        {"role": "assistant", "content": json.dumps(
            {"content": "Looking.", "tool_calls": [call]}), "author": None},
        {"role": "user", "content": "[cancel notice]", "author": "cancel_notice"},
        {"role": "user", "content": "an older message", "author": None},
        {"role": "assistant", "content": "ANCIENT REPLY", "author": None},
    ]
    _turn(sdk)
    state = sdk.last_state
    assert state["request"] == "my upload is stuck"
    context = state["recent_context"]
    assert "TOOL RESULT PAYLOAD" not in context
    assert "ANCIENT REPLY" not in context, "only the latest agent turn"
    assert "tool: edit_file(" in context and "fixing the upload retry" in context
    assert "x" * 400 not in context, "a long argument is clipped"
    assert context.index("Looking.") < context.index("Retried it for you.")


def test_both_files_read_the_same_amount_of_frontmatter():
    for relative in (SERVICE, MEMORY):
        assert "HEAD_CHARS = 4000" in _source_or_skip(relative), relative


# ──────────────────────────────────────────────────────────────────────
# Writing ``when_to_retrieve``.
# ──────────────────────────────────────────────────────────────────────

class _ToolSdk:
    Failed = _Failed

    def __init__(self, verdict=None, existing=None):
        self.verdict = verdict or {"ok": True}
        self.files = dict(existing or {})
        self.validated = []
        self.paths = type("P", (), {"get": staticmethod(lambda key: "/w")})()
        self.path = _Path()
        sdk = self
        self.fs = type("F", (), {
            "list": staticmethod(self._list),
            "read": staticmethod(self._read),
            "write": staticmethod(lambda p, t: sdk.files.__setitem__(p, t)),
        })()
        self.services = type("S", (), {"call": staticmethod(self._call)})()
        self.session = type("Z", (), {
            "get": staticmethod(lambda: {"conversation_id": 3, "attended": True}),
            "push": staticmethod(lambda *a, **k: None)})()
        self.config = type("C", (), {"read": staticmethod(lambda key: None)})()

    def _list(self, path, **_):
        if path in self.files:
            return [{"name": path}]
        raise _Failed(path)

    def _read(self, path):
        if path not in self.files:
            raise _Failed(path)
        return self.files[path]

    def _call(self, service, method, questions):
        assert (service, method) == ("rlcd", "validate")
        self.validated.append(questions)
        return self.verdict

    def fail(self, message):
        return ("fail", message)

    def ok(self, value, llm_summary=""):
        return ("ok", llm_summary)

    def log(self, *a, **k):
        pass


def _memory_tool():
    return _load_store_class(MEMORY, "Memory")()


_Q = {"pdf": {"type": "noul", "instructions": "Is the user working with a PDF?"}}


def test_create_requires_questions_and_passes_them_to_jev_validate():
    sdk = _ToolSdk()
    tool = _memory_tool()
    args = {"action": "create", "name": "pdf-empty",
            "description": "A PDF yields no text", "body": "Check the parser."}
    assert tool.run(sdk, **args)[0] == "fail"
    assert tool.run(sdk, **args, when_to_retrieve=_Q)[0] == "ok"
    assert sdk.validated == [_Q]

    written = sdk.files["/w/memory/notes/pdf-empty.md"]
    service = {}
    exec(compile(_source_or_skip(SERVICE), SERVICE, "exec"), service)
    fields = service["_frontmatter"](written)
    assert service["_questions"](fields["when_to_retrieve"]) == _Q


def test_only_one_to_three_yes_no_questions_are_accepted():
    tool = _memory_tool()
    base = {"action": "create", "name": "x", "description": "d", "body": "b"}
    bad = [
        {"q": {"type": "score", "instructions": "How bad?", "criteria": ["a", "b"]}},
        {f"q{i}": {"type": "noul", "instructions": "Q?"} for i in range(4)},
        {},
        {"q": {"type": "noul", "instructions": ""}},
    ]
    for questions in bad:
        sdk = _ToolSdk()
        result = tool.run(sdk, **base, when_to_retrieve=questions)
        assert result[0] == "fail", questions
        assert sdk.validated == [], "shape errors never reach Jev"


def test_jevs_refusal_is_handed_back_to_the_model():
    sdk = _ToolSdk(verdict={"ok": False, "reason": "too many tokens"})
    result = _memory_tool().run(sdk, action="create", name="x", description="d",
                                body="b", when_to_retrieve=_Q)
    assert result[0] == "fail" and "too many tokens" in result[1]


def test_update_keeps_the_questions_it_was_not_given():
    path = "/w/memory/notes/pdf-empty.md"
    existing = {path: ("---\nname: pdf-empty\ndescription: A PDF yields no text\n"
                       "when_to_retrieve: " + json.dumps(_Q) + "\n---\n\nold\n")}
    sdk = _ToolSdk(existing=existing)
    result = _memory_tool().run(sdk, action="update", name="pdf-empty", body="new")
    assert result[0] == "ok"
    assert json.dumps(_Q) in sdk.files[path]
    assert sdk.validated == [], "keeping what is on disk needs no re-check"
