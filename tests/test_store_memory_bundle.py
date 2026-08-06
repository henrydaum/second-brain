"""What the kernel reads off the four files that make up memory.

Same shape as ``test_store_attachment_tools``: kernel invariants that happen to
be *about* store files. The subject is the kernel's own verdict — does this
load, are these Requests real, is the retrieval free of dialogs, can this reach
outside the folder — and the store file is the input.

The four matter together because they are two pairs, and each half is useless
alone. Retrieval is ``service_memory_retrieve`` (ranks the corpus at
``turn_start`` and injects descriptions) plus ``tool_memory_recall`` (opens one
by name). Curation is ``task_memory_reflect`` (spawns a curator when a
conversation ends) plus ``tool_memory_curate`` (the only writer). Automatic
half, agent-invoked half, twice over — which is what the four names say.
What connects them is not an import
but the *folder* and one table, so the things worth pinning are the
declarations that decide whether any of it runs at all: the hook moment, the
trigger channel, the default job, the shared constants, and the profile that
keeps the curator inside the folder.

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
TASK = "tasks/task_memory_reflect.py"
RECALL = "tools/tool_memory_recall.py"
CURATE = "tools/tool_memory_curate.py"
BUNDLE = "bundles/bundle_memory.json"

SUITE = [SERVICE, TASK, RECALL, CURATE]


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

def test_all_four_agree_on_where_entries_live():
    """Membership is a path, which is the one thing a writer cannot fumble.

    Requiring a field in the frontmatter made *being an entry* something the
    writer had to restate correctly in every file, and getting it subtly wrong
    — no fences, the key in the body — made the entry silently unreachable. A
    path cannot be subtly wrong.

    Three files derive that path independently — the two tools *could* share a
    ``tools/helpers/`` module, but the service is another family and could not,
    so sharing would take three copies to two rather than to one. This test is
    what keeps them equal either way. The task is deliberately not among them:
    it reaches the corpus only through ``tool.call``, so it holds no path
    knowledge at all and cannot drift.
    """
    for relative in (SERVICE, RECALL, CURATE):
        source = _source_or_skip(relative)
        assert 'MEMORY_DIRNAME = "memory"' in source, relative
        assert 'NOTES_DIRNAME = "notes"' in source, relative
        assert 'SKILLS_DIRNAME = "skills"' in source, relative

    assert "NOTES_DIRNAME" not in _source_or_skip(TASK)


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


def test_the_corpus_is_actions_and_facts_live_elsewhere():
    """An entry that names no action cannot change what anyone does.

    That rule is the whole filter on what gets written: it kills the dominant
    failure mode, which is a curator producing tidy summaries of what happened,
    and it implements "a neutral result is not worth recording" for free — a
    neutral result changes no action, so there is nothing to write.

    Facts go in MEMORY.md, which the kernel inlines and no part of this suite
    may touch. Every half has to say so or the agent is told one thing and the
    curator does another.
    """
    for relative in (SERVICE, TASK, CURATE):
        source = _source_or_skip(relative)
        assert "MEMORY.md" in source, relative

    # The rule is stated where it decides something: to the agent that writes
    # mid-conversation, and to the curator that writes after one. The service
    # only has to say where facts go instead. Matched on the fragment that
    # survives line-wrapping in both prompts.
    assert "there is nothing to write" in _source_or_skip(CURATE)
    assert "there is nothing to write" in _source_or_skip(TASK)
    assert "not entries" in _source_or_skip(SERVICE)


# ──────────────────────────────────────────────────────────────────────
# The tools. Two halves of one folder, with disjoint reach.
# ──────────────────────────────────────────────────────────────────────

def test_neither_tool_can_be_handed_a_path():
    """The confinement is structural, not a check that could be bypassed.

    Both tools take a *name* and derive the path from it, so there is no
    argument that escapes ``workspace/memory`` and none that names MEMORY.md.
    That is what makes them safe to give a subagent nobody is watching — and
    a ``path`` parameter appearing later would quietly undo it.
    """
    for relative in (RECALL, CURATE):
        parameters = _load_store_class(
            relative, "MemoryRecall" if relative == RECALL else "MemoryCurate"
        ).parameters
        properties = set(parameters["properties"])
        assert "path" not in properties, relative
        assert "name" in properties, relative
        assert not (properties & {"folder", "file", "filename"}), relative


def test_a_name_cannot_be_a_path():
    """The character set is the check, and it is the load-bearing one.

    Lowercase alphanumerics and hyphens is the agentskills.io rule, and it also
    happens to exclude every way of writing a traversal: a dot, a separator, a
    drive letter, a leading tilde.
    """
    for relative, class_name in ((RECALL, "MemoryRecall"), (CURATE, "MemoryCurate")):
        namespace = {}
        exec(compile(_source_or_skip(relative), relative, "exec"), namespace)
        valid = namespace["_valid_name"]

        assert valid("retry-failed-uploads")
        assert valid("pdf2text")
        for bad in ("../escape", "a/b", r"a\b", "C:name", "~", "UPPER",
                    "-lead", "trail-", "a--b", "", "x" * 65, "with space",
                    "MEMORY.md", "note.md"):
            assert not valid(bad), f"{relative}: {bad!r} must be refused"


def test_the_writer_cannot_read_and_the_reader_cannot_write():
    """Disjoint capabilities, declared rather than described.

    ``memory_curate`` requires a description on update instead of reading the
    old one back, which is what lets it declare no ``fs.read`` at all — so the
    tool that can write is not also a way to see what is in the folder.
    Symmetrically the reader cannot write a file; its one write is the usage
    row that records the recall.
    """
    recall = set(_declarations(RECALL)["requests"])
    curate = set(_declarations(CURATE)["requests"])

    assert not (recall & {"fs.write", "fs.delete"})
    assert not (curate & {"fs.read"})
    assert {"fs.write", "fs.delete"} <= curate
    assert "fs.read" in recall


def test_reading_a_skill_points_at_its_resources_without_loading_them():
    """Progressive disclosure done by the tool rather than by prose.

    The spec's whole structure is SKILL.md plus files loaded on demand. Listing
    the resource paths on read means the agent never has to guess a path or
    list a directory, and never pays for a reference it does not open.
    """
    source = _source_or_skip(RECALL)
    assert 'SKILL_RESOURCE_DIRS = ("references", "scripts", "assets")' in source
    assert "_resources" in source


# ──────────────────────────────────────────────────────────────────────
# The usage table: one place, three writers of one fact each.
# ──────────────────────────────────────────────────────────────────────

def test_offered_and_recalled_are_one_table():
    """The service offers, the recall tool takes, the task reads.

    Splitting these was the old design's expense: the offer lived in one table
    and the *take* had to be reconstructed by parsing every assistant message
    for a ``read_file`` call and normalizing the path it named. A dedicated
    tool records the take directly, which is the whole reason it exists.
    """
    service = _source_or_skip(SERVICE)
    recall = _source_or_skip(RECALL)
    task = _source_or_skip(TASK)

    for source, name in ((service, "service"), (recall, "recall"), (task, "task")):
        assert "memory_usage" in source, name

    # The service defines it and inserts the offer.
    assert "CREATE TABLE IF NOT EXISTS memory_usage" in service
    assert "recalled_at" in service
    assert "db.define" in _declarations(SERVICE)["requests"]

    # The tool fills it in, or records a recall nobody offered.
    assert "SET recalled_at = ?" in recall
    assert "INSERT INTO memory_usage" in recall

    # The task only reads. Its one write is the retention sweep.
    assert "SELECT DISTINCT name FROM memory_usage" in task
    assert "UPDATE memory_usage" not in task


def test_ordering_is_free_so_no_message_ids_are_compared():
    """Only a *later* recall can fill a NULL, so the pair is ordered by
    construction. Everything the old design needed to establish that — the
    offered message id, the id comparison, path absolutization — is gone, and
    a reappearance would mean the tracking had been reconstructed again.

    Asserted against the code rather than the prose, since the docstrings
    describe the machinery that was removed and should go on doing so.
    """
    task = _source_or_skip(TASK)

    assert "offered_message_id" not in task
    assert "_read_file_calls" not in task
    assert '"read_file"' not in task, "no tool-call name is matched any more"
    assert "tool_calls" not in task
    assert "import json" not in task
    assert "sdk.path." not in task


def test_recalls_survive_reflection_because_they_are_the_data():
    """Reflection reads the log and must not consume it.

    Which entries earn their place, over time, is the input to a future pass
    over what nobody has recalled in months. Clearing per conversation — which
    is what the old design did — throws that away for a table that was never
    the problem. Offers are the volume and are pruned; recalls are rare, small,
    and kept.
    """
    task = _source_or_skip(TASK)

    assert "_forget_retrievals" not in task
    assert "DELETE FROM memory_usage" in task
    assert "WHERE recalled_at IS NULL AND offered_at < ?" in task


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
    assert declared["hooks"] == {"turn_start": "on_turn_start"}
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
    # What it calls. The curator's other read tools are declared here as well,
    # since ``on_install`` writes their names into a whitelist and a name
    # nobody installed grants nothing — but which file supplies each is the
    # closure's business, pinned by
    # ``test_the_bundle_ships_every_tool_the_curator_profile_names``.
    assert {"tools/tool_hybrid_search.py", "tools/tool_memory_recall.py",
            "tools/tool_memory_curate.py"} <= set(declared["dependencies_files"])


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


def test_the_service_writes_its_two_kernel_settings_only_at_install():
    """It needs two settings, and there is exactly one moment it may ask.

    ``sync_directories`` must contain the memory folder or nothing is indexed;
    ``agent_profiles`` must hold the curator profile or no curator can spawn.
    Both were attempted from ``start`` — refused, because a service has no
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
    for banned in ("start", "stop", "on_turn_start", "on_uninstall"):
        assert banned not in writers, f"{banned} writes config"

    for method in ("on_install", "on_uninstall"):
        assert f"def {method}(self, sdk)" in source, method


# ──────────────────────────────────────────────────────────────────────
# The curator: when it runs, and what it may do while it does.
# ──────────────────────────────────────────────────────────────────────

def test_the_curator_listens_for_the_conversation_that_ended():
    """``ended``, not ``changed`` — the two name opposite conversations.

    ``session_conversation_changed`` names the one being switched *to*, which
    is what a frontend redrawing a banner wants and the exact opposite of what
    reflection needs. Subscribing to the wrong one would reflect on the
    conversation the user is about to start typing in.
    """
    from events.event_channels import SESSION_CONVERSATION_ENDED

    declared = _declarations(TASK)
    assert declared["family"] == "task"
    assert declared["name"] == "memory_reflect"
    assert declared["trigger"] == "event"
    assert declared["trigger_channels"] == [SESSION_CONVERSATION_ENDED]


def test_nothing_here_runs_on_a_clock():
    """The event covers every ordinary ending, so a sweep finds nothing.

    There was an hourly one, declared as a backstop for an event lost to a
    crash. The cost is what settles it: this task *spawns subagents*, so a
    speculative wake-up is the most expensive kind there is, and it bought
    recovery for a case that leaves exactly one reflection unwritten. A lesson
    not learned is not a corruption — the watermark table is still consistent —
    so the gap is accepted rather than paid for twenty-four times a day.

    Stated as a test rather than simply deleted because the way a task gets a
    schedule has since changed: it is an ``on_install`` that calls the
    timekeeper, so adding one here would be a method rather than a declaration
    and would not show up as a stray attribute anybody notices.
    """
    from sandbox.bridge import lifecycle_entries

    assert not _declarations(TASK).get("default_jobs"), (
        "a retired declaration; the validator drops it and nothing reads it")
    assert not lifecycle_entries(_source_or_skip(TASK), "on_install"), (
        "nothing here creates a timekeeper job either")


def test_the_curator_is_spawned_under_a_profile_that_cannot_reach_edit_file():
    """The toolset is the safety property, and it is declarative.

    A curator is a real agent turn running unattended. With the default profile
    it writes notes with ``edit_file``, which reaches every file in the
    workspace including MEMORY.md. Named profile plus a curation tool that
    derives its own paths is what confines it — and the two halves have to
    agree on the profile name or the spawn fails.
    """
    service = _source_or_skip(SERVICE)
    task = _source_or_skip(TASK)

    assert 'CURATOR_PROFILE = "memory_curator"' in service
    assert 'CURATOR_PROFILE = "memory_curator"' in task
    assert "profile=CURATOR_PROFILE" in task

    namespace = {}
    exec(compile(service, SERVICE, "exec"), namespace)
    allowed = set(namespace["CURATOR_TOOLS"])
    assert "edit_file" not in allowed
    assert {"memory_recall", "memory_curate"} <= allowed
    # The list grows as the suite learns what a curator needs, and every
    # addition is a read. Stated as a rule rather than as a fixed list, so a
    # future writer has to be added deliberately and against this line.
    assert not {"edit_file", "write_file", "run_command", "delete_file",
                "spawn_subagent"} & allowed


def test_the_bundle_ships_every_tool_the_curator_profile_names():
    """A whitelist naming a tool nobody installed silently grants nothing.

    The profile and the manifest are edited in different files, and the failure
    is invisible from either: ``scoped_registry`` filters against what is
    *registered*, so an uninstalled name is dropped without a word and the
    curator simply runs without the capability its prompt describes. That is
    the same silent-narrowing shape as the missing tools that made a curator
    unable to search at all.

    The question is asked of the *install closure*, not the manifest, because
    the manifest deliberately names only the four memory plugins and lets
    ``dependencies_files`` supply the rest — which is the same walk the package
    manager does.
    """
    namespace = {}
    exec(compile(_source_or_skip(SERVICE), SERVICE, "exec"), namespace)
    shipped = {Path(rel).stem.split("_", 1)[1] for rel in _install_closure()
               if rel.startswith("tools/tool_")}

    missing = sorted(set(namespace["CURATOR_TOOLS"]) - shipped)
    assert not missing, f"the profile names tools the bundle does not ship: {missing}"


def test_a_missing_curator_profile_stops_the_curator_rather_than_widening_it():
    """Nothing in this suite can create the profile, so the failure has to be
    loud and it has to be closed.

    ``agent_profiles`` is a kernel setting and no part of a plugin can write
    one — a task's chain is unattended, and the service's attempt was removed
    for the same reason. So until the package manager seeds settings at
    install, the profile is created by hand, and the only thing this code owes
    is to refuse to run without it: ``_reflect`` returns False, which leaves
    the watermark where it is so a later sweep retries, rather than spawning an
    unrestricted curator that can reach ``edit_file``.
    """
    task = _source_or_skip(TASK)
    assert "profile=CURATOR_PROFILE" in task

    reflect = task.split("def _reflect", 1)[1].split("    def ", 1)[0]
    assert "return False" in reflect
    # The kernel half — refusing the spawn outright — is
    # ``test_an_unknown_profile_is_refused_rather_than_widened`` below.


def test_an_unknown_profile_is_refused_rather_than_widened():
    """The kernel half of the same rule.

    Substituting ``default`` for a profile that does not exist would run the
    child with every installed tool while the caller believed it was confined,
    and nothing anywhere would say so.
    """
    from runtime.subagents import SubagentRegistry

    class Runtime:
        config = {"agent_profiles": {"default": {}}}
        sessions = {}

    registry = SubagentRegistry(Runtime(), Runtime.config)
    with pytest.raises(ValueError, match="memory_curator"):
        registry.spawn("go", owner="repl", profile="memory_curator")


def test_subagent_curation_is_opt_in_and_never_reaches_scheduled_ones():
    """Three kinds of conversation, separated by the category the kernel sets.

    An interactive ``sdk.agent.spawn`` files its child under ``Subagent``; a
    scheduled one under ``Scheduled``/``Scheduled (one-time)``
    (``runtime/subagents.py`` ``_scheduled_category``). The setting opens the
    first and must never open the second: a scheduled job pins its conversation
    and reuses it forever, so it has no ending to reflect on and would hand the
    curator the same growing transcript every hour.
    """
    declared = _declarations(TASK)
    keys = {entry[1] for entry in declared["config_settings"]}
    assert "memory_reflect_include_subagents" in keys

    setting = next(e for e in declared["config_settings"]
                   if e[1] == "memory_reflect_include_subagents")
    assert setting[3] is False, "opt-in: it costs a curator run per subagent"

    source = _source_or_skip(TASK)
    assert "NOT LIKE 'Scheduled%'" in source, "unconditional, in both modes"


def test_the_curator_cannot_reach_its_own_output_in_either_mode():
    """Once subagents are curated, the session-key guard cannot fire.

    With subagents off, a child's event is dropped outright. With them on that
    guard is deliberately bypassed — which is exactly when the curator's own
    conversation becomes an ordinary candidate. The title filter is what stands
    in the way, and it is exact rather than fragile because the task sets that
    title itself when it spawns and matches the same constant when it queries.
    """
    source = _source_or_skip(TASK)
    assert "COALESCE(c.title, '') NOT LIKE ?" in source
    assert "CURATOR_TITLE}%" in source, "the title must be the bound value"


def test_the_curator_does_not_react_to_its_own_children():
    """One conversation ending produced four runs, and this is why.

    A subagent gets its own conversation and closes its session when it is
    done, so every curator completion emits the same channel that spawned it.
    Unfiltered, the curator's own transcript reads as a conversation that has
    gone quiet — so it reflects on itself, spawns another curator, and only
    stops when a transcript happens to fall under the message floor.
    """
    from runtime.subagents import SESSION_PREFIX

    source = _source_or_skip(TASK)
    assert SESSION_PREFIX in source, "the event path must skip child sessions"
    assert "<> 'Subagent'" in source, "the sweep path must skip child conversations"


def test_the_curator_sees_the_tool_calls_it_has_to_branch_on():
    """A user/assistant-only transcript hides where the work happened."""
    source = _source_or_skip(TASK)
    assert "tool_name" in source
    assert "<> 'system'" in source, "everything but system rows belongs in it"


def test_reflection_transcript_starts_after_the_previous_watermark():
    task = _source_or_skip(TASK)

    assert "AS previous_id" in task
    transcript = task.split("def _transcript", 1)[1]
    assert "AND id > ?" in transcript
    assert "[cid, previous_id, max_id, limit]" in transcript


def test_installing_the_bundle_does_not_reflect_on_the_whole_archive():
    """The watermark defaults to zero, so history reads as new.

    On a fresh install every conversation ever held qualifies at once: nobody
    has reflected on them, so every message counts as unreflected. Observed as
    six runs draining years of conversations three at a time and writing notes
    about work from months ago as though it had just happened.
    """
    declared = _declarations(TASK)
    keys = {entry[1] for entry in declared["config_settings"]}
    assert "memory_reflect_max_age_hours" in keys

    source = _source_or_skip(TASK)
    assert "MAX(COALESCE(m.timestamp, 0)) >= ?" in source


def test_a_conversation_the_agent_never_spoke_in_is_skipped():
    """The corpus records what the agent did, so no agent means nothing to say.

    The message floor does not cover this — someone can reach it without the
    agent ever answering: messages typed at a turn that failed or was
    cancelled, or a conversation opened only to run slash commands.
    """
    source = _source_or_skip(TASK)
    assert "LOWER(m.role) = 'assistant'" in source
    assert "THEN 1 ELSE 0 END) >= 1" in source


def test_the_watermark_is_a_table_the_task_owns():
    """The watermark is what makes reflection idempotent.

    Declaring the table in ``writes`` is not bookkeeping: the orchestrator
    writes the task's returned rows into it with INSERT OR REPLACE, which is
    how the watermark advances without the task needing ``db.write`` at all.
    """
    declared = _declarations(TASK)
    assert declared["writes"] == ["memory_reflections"]
    assert "memory_reflections" in declared["output_schema"]
    assert "agent.spawn" in declared["requests"]

    source = _source_or_skip(TASK)
    assert "db.write" in declared["requests"]
    assert "INSERT" not in source.upper(), (
        "the watermark advances through the returned rows, not by hand")


def test_the_facts_job_is_gone_and_memory_md_is_the_agents_own():
    """MEMORY.md is inlined into every prompt and belongs to the agent the
    user actually talks to. A background subagent pruning it is a thing nobody
    watches editing the one file everybody reads."""
    task = _source_or_skip(TASK)

    assert "memory_reflect_curate_facts" not in task
    assert "memory_index_cap" not in task
    assert "Job three" not in task
    # And no tool the curator holds can address it: ``MEMORY.md`` is not a
    # legal entry name, which ``test_a_name_cannot_be_a_path`` pins directly.


def test_the_event_task_implements_the_event_entry_point():
    """``run_event``, not ``run`` — implementing the wrong one fails silently.

    The task template still shows an event task overriding ``run``, so this is
    a trap worth a test rather than a comment.
    """
    source = _source_or_skip(TASK)
    assert "def run_event(self, sdk, payload)" in source
    assert "def run(self, sdk" not in source


# ──────────────────────────────────────────────────────────────────────
# The manifest.
# ──────────────────────────────────────────────────────────────────────

def test_the_manifest_names_the_four_and_lets_the_closure_do_the_rest():
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
        "the manifest is the four memory plugins; anything else they need is "
        "reached through dependencies_files")

    # Everything the bundle installs is still everything it needs, reached the
    # way the package manager reaches it.
    closure = _install_closure()
    assert "tools/tool_hybrid_search.py" in closure
    # read_file is for a skill's own references, which memory_recall names but
    # deliberately does not load.
    assert "tools/tool_read_file.py" in closure
    # And the one the curator must not have. Its absence is not the guard — the
    # profile is — but pulling it in here would make the restriction look
    # accidental.
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
        self.emitted = []
        self.logs = []
        self.session = self._Session(self)
        self.config = self._Config(self)
        self.events = self._Events(self)

    def log(self, message, level="info"):
        self.logs.append((level, message))

    class _Session:
        def __init__(self, sdk):
            self._sdk = sdk

        def get(self, key=None):
            if self._sdk._attended == "unreadable":
                raise _Failed("no runtime")
            return {"key": "k", "attended": self._sdk._attended}

    class _Config:
        def __init__(self, sdk):
            self._sdk = sdk

        def read(self, key):
            if self._sdk._setting == "unreadable":
                raise _Failed("no config")
            return self._sdk._setting

    class _Events:
        def __init__(self, sdk):
            self._sdk = sdk

        def emit(self, channel, payload=None):
            self._sdk.emitted.append((channel, payload))


def _curate():
    """The real ``MemoryCurate``, loaded the way a box loads it."""
    return _load_store_class(CURATE, "MemoryCurate")()


def test_a_background_write_is_announced_to_the_chat():
    """The curator writes where nobody is looking; this is the only trace."""
    tool = _curate()
    sdk = _Sdk(attended=False)

    tool._notify(sdk, "create", "retry-failed-uploads")

    assert len(sdk.emitted) == 1
    channel, payload = sdk.emitted[0]
    assert channel == "chat_message_pushed"
    assert payload["message"] == "Memory created: retry-failed-uploads"
    # No ``session_key``: the frontend reads that field to target one session
    # and broadcasts to every live one without it. The write happened in a
    # session with no person on it, so there is nothing to reply *to*.
    assert "session_key" not in payload


def test_every_action_has_its_own_word():
    tool = _curate()
    for action, expected in (("create", "created"), ("update", "updated"),
                             ("delete", "deleted")):
        sdk = _Sdk(attended=False)
        tool._notify(sdk, action, "x")
        assert sdk.emitted[0][1]["message"] == f"Memory {expected}: x"


def test_a_write_the_user_watched_is_not_announced():
    """They asked for it in conversation and the reply already says so.

    ``attended`` rather than "is this a subagent" because the kernel already
    owns that question, and a concurrent multi-user frontend can override it
    per session — any guess made from a session key would be wrong there.
    """
    tool = _curate()
    sdk = _Sdk(attended=True)

    tool._notify(sdk, "create", "x")

    assert sdk.emitted == []


def test_the_setting_turns_it_off_and_defaults_on():
    tool = _curate()

    off = _Sdk(attended=False, setting=False)
    tool._notify(off, "create", "x")
    assert off.emitted == []

    unset = _Sdk(attended=False, setting=None)
    tool._notify(unset, "create", "x")
    assert len(unset.emitted) == 1, "unset means default, and the default is on"


def test_the_two_unreadable_cases_fail_in_opposite_directions():
    """A spurious notification is worse than a missing one; a missing setting
    is not a refusal."""
    tool = _curate()

    blind = _Sdk(attended="unreadable")
    tool._notify(blind, "create", "x")
    assert blind.emitted == [], "not knowing where we are means staying quiet"

    no_config = _Sdk(attended=False, setting="unreadable")
    tool._notify(no_config, "create", "x")
    assert len(no_config.emitted) == 1, "the default is on"


def test_announcing_can_never_fail_the_write():
    """The entry is already on disk by the time this runs, so raising here
    would report an error for something that fully succeeded."""
    tool = _curate()
    sdk = _Sdk(attended=False)
    sdk.events.emit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bus"))

    tool._notify(sdk, "create", "x")     # must not raise

    assert any(level == "debug" for level, _ in sdk.logs)


def test_all_three_operations_announce():
    """The helper is only worth having if every write reaches it.

    Read off the source, because a missed call site is invisible from the
    helper's own tests — the notification simply never happens for that one
    action.
    """
    import ast

    tree = ast.parse(_source_or_skip(CURATE))
    for method, action in (("_create", "create"), ("_update", "update"),
                           ("_delete", "delete")):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == method)
        calls = [n for n in ast.walk(node)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "_notify"]
        assert len(calls) == 1, f"{method} does not announce exactly once"
        assert calls[0].args[1].value == action, f"{method} announces the wrong action"
