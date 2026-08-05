"""What the kernel reads off the two files that make up memory.

Same shape as ``test_store_attachment_tools``: kernel invariants that happen to
be *about* store files. The subject is the kernel's own verdict — does this
load, are these Requests real, is the retrieval free of dialogs — and the store
file is the input.

The pair matters together because they are the two halves of one loop and each
is useless alone. ``service_memory`` reads (searches the folder at
``turn_start`` and injects pointers); ``task_memory_reflect`` writes (spawns a
curator when a conversation goes quiet). What connects them is not an import
but the *folder*, so the things worth pinning are the declarations that decide
whether either half ever runs: the hook moment, the trigger channel, and the
default job.

Skips cleanly when no store ref is reachable.
"""

import json
from pathlib import Path

import pytest

# Aliases the guest package under the bare name ``guest``, which is how plugin
# source resolves its imports both in-process and in a child.
import sandbox  # noqa: F401
from tests.support import store_source, store_worktree

SERVICE = "services/service_memory.py"
TASK = "tasks/task_memory_reflect.py"
BUNDLE = "bundles/bundle_memory.json"


def _source_or_skip(relative: str) -> str:
    text = store_source(relative)
    if text is None:
        pytest.skip(f"{relative} is not present on a local store ref")
    return text


def _declarations(relative: str) -> dict:
    from sandbox.validator import validate

    return validate(_source_or_skip(relative),
                    filename=Path(relative).name).declarations


@pytest.mark.parametrize("relative", [SERVICE, TASK])
def test_the_memory_bundle_conforms(relative):
    """``conforms`` is the whole question: it means the file loads in a box."""
    from sandbox.validator import validate

    report = validate(_source_or_skip(relative), filename=Path(relative).name)
    errors = [f for f in report.findings if f.level == "error"]
    assert not errors, report.render()


@pytest.mark.parametrize("relative", [SERVICE, TASK])
def test_every_declared_request_is_a_real_one(relative):
    """``requests`` is the approval grant, so a typo silently narrows it."""
    from guest.requests import ALL_TYPES

    assert set(_declarations(relative)["requests"]) <= set(ALL_TYPES)


def test_retrieval_stands_at_the_one_moment_that_runs_per_turn():
    """``turn_start``, not ``llm_call`` — the distinction is the latency floor.

    A hook at ``llm_call`` would re-run the search on every model call within a
    turn, for a query that only changed once. ``turn_start`` fires once per
    logical turn, which is exactly the granularity of "what is this message
    about".
    """
    declared = _declarations(SERVICE)
    assert declared["family"] == "service"
    assert declared["name"] == "memory"
    assert declared["hooks"] == {"turn_start": "on_turn_start"}
    # It contributes guidance but exposes no callable surface: nothing should
    # be reaching into memory through ``service.call``.
    assert declared["exports"] == []


def test_the_corpus_is_actions_and_facts_live_elsewhere():
    """A note that names no action cannot change what anyone does.

    That rule is the whole filter on what gets written: it kills the dominant
    failure mode, which is a curator producing tidy summaries of what happened,
    and it implements "a neutral result is not worth recording" for free —
    a neutral result changes no action, so there is nothing to write.

    Facts go in MEMORY.md, which the kernel inlines and the curator must never
    touch. The two halves have to agree about that or the agent is told one
    thing and the curator does another.
    """
    task = _source_or_skip(TASK)
    service = _source_or_skip(SERVICE)

    for source, name in ((task, "task"), (service, "service")):
        assert "do:" in source and "avoid:" in source, name
        assert "MEMORY.md" in source, name

    assert "cannot change anything" in service
    assert "not touch" in task or "never touches" in task

    # The taxonomy that used to gate rendering is gone: a fact is a situation
    # with a short action, so the split that matters is length, not kind.
    assert "type: skill" not in task
    assert "supersedes" not in task


def test_the_curator_sees_the_tool_calls_it_has_to_branch_on():
    """Whether the agent used memory is only visible in what it called.

    The curator's two jobs are chosen by that one fact — improve the notes that
    were used, or write down the novel solution that was reached without them.
    A user/assistant-only transcript cannot answer it, and would silently
    collapse both jobs into the second.
    """
    source = _source_or_skip(TASK)
    assert "tool_name" in source
    assert "<> 'system'" in source, "everything but system rows belongs in it"


def test_the_prompt_carries_situations_and_not_the_advice():
    """Inlining the note destroys the signal the whole loop runs on.

    With the advice already in the prompt there is no reason to open the file,
    so nothing downstream can tell which notes were used — and that pair is
    what selects the curator's job. The situation alone answers the only
    question the prompt has to answer, which is whether a past case is this
    one; what was tried and how it went is what the file is for.
    """
    service = _source_or_skip(SERVICE)

    # The line is the situation and the path. Neither the action nor the
    # outcome may be rendered into it.
    assert 'f"- {situation}\\n  ({path})"' in service
    assert "INLINE_CHARS" not in service
    assert '"because"' not in service


def test_a_note_is_a_note_because_of_where_it_is():
    """Membership is a path, which is the one thing a writer cannot fumble.

    Requiring a ``when`` in the frontmatter made *being a note* something the
    writer had to restate correctly in every file, and getting it subtly wrong
    — no fences, the key in the body — made the note silently unreachable.
    It also needed an exception list naming every file in the folder that was
    not a note, which grows and whose omissions are invisible.

    Both sides have to agree on the folder name or the curator writes where
    nothing looks.
    """
    service = _source_or_skip(SERVICE)
    task = _source_or_skip(TASK)

    for source, name in ((service, "service"), (task, "task")):
        assert 'NOTES_DIRNAME = "actions"' in source, name

    # Searching is scoped to it, so nothing outside can rank at all — and the
    # exception list that scoping replaced is gone.
    assert "folder=_notes_root(sdk)" in service
    assert '"readme.md", "memory.md"' not in service


def test_a_note_with_no_situation_is_reported_not_guessed_at():
    """Inside the notes folder, a missing ``when`` is a broken note.

    There is no situation to render and falling back to the matched chunk
    would put a fragment with no context into a list that promises situations.
    The symptom otherwise is a note that ranks well and is never once offered,
    which is indistinguishable from having no memories.
    """
    service = _source_or_skip(SERVICE)

    assert 'if situation else ""' in service
    assert 'hit.get("content")' not in service, "no chunk fallback may remain"
    assert "with no 'when' were skipped" in service


def test_both_halves_of_the_used_pair_are_recorded_and_read():
    """Neither half is available alone.

    The offer lives in the system prompt, which is stored nowhere — so the
    service writes it down. The open is a read_file call, which reaches the
    transcript because the agent had to name the path to make it. The service
    must therefore write the log and the task must read it; either one missing
    leaves the curator unable to tell its two jobs apart, silently.
    """
    service = _source_or_skip(SERVICE)
    task = _source_or_skip(TASK)

    assert "memory_retrievals" in service and "memory_retrievals" in task
    declared = _declarations(SERVICE)["requests"]
    assert "db.define" in declared and "db.write" in declared

    # The log answers one question once; nothing else prunes this table.
    assert "DELETE FROM memory_retrievals" in task

    # Matched on the filename because a stored tool call escapes separators
    # twice — see tests/test_tool_call_args_persist.py. Matching the whole
    # path works on POSIX and silently fails on Windows.
    assert "sdk.path.name(path) in transcript" in task


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
    assert declared["dependencies_files"] == ["tools/tool_hybrid_search.py"]


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
    decision = classify(
        Request(SESSION_ADD_PROMPT, {"text": "pointers", "slot": "memory"}),
        Chain(root="repl"))
    assert decision.safe


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


def test_the_curator_is_also_swept_because_a_crash_emits_nothing():
    """The hourly job is the backstop, not a second trigger.

    The event makes reflection prompt; it cannot make it reliable, because a
    crash emits nothing at all. The sweep asks the same watermark question with
    no event, which is what turns a lost event into a delay rather than into
    work silently dropped.
    """
    from events.event_channels import SESSION_CONVERSATION_ENDED

    jobs = _declarations(TASK)["default_jobs"]
    assert list(jobs) == ["memory_reflect_sweep"]
    assert jobs["memory_reflect_sweep"]["channel"] == SESSION_CONVERSATION_ENDED
    assert jobs["memory_reflect_sweep"]["cron"] == "0 * * * *"


def test_the_facts_job_reads_the_kernel_budget_rather_than_copying_it():
    """MEMORY.md has two readers who must agree on how long it may be.

    The kernel truncates the index at ``memory_index_cap``; the curator prunes
    to fit inside it. A constant on each side would drift the moment either
    changed, and the failure is quiet — facts written past the cap are simply
    never in the prompt. The task cannot import the kernel to ask, so the
    number is config and both sides read it.
    """
    declared = _declarations(TASK)
    keys = {entry[1] for entry in declared["config_settings"]}
    assert "memory_reflect_curate_facts" in keys

    setting = next(e for e in declared["config_settings"]
                   if e[1] == "memory_reflect_curate_facts")
    assert setting[3] is True

    source = _source_or_skip(TASK)
    assert 'sdk.config.read("memory_index_cap")' in source
    assert "config.read" in declared["requests"]


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
    # Applied as a bound parameter, unconditionally — not inside the branch
    # that the setting turns off.
    assert "COALESCE(c.title, '') NOT LIKE ?" in source
    assert "CURATOR_TITLE}%" in source, "the title must be the bound value"


def test_the_curator_does_not_react_to_its_own_children():
    """One conversation ending produced four runs, and this is why.

    A subagent gets its own conversation and closes its session when it is
    done, so every curator completion emits the same channel that spawned it.
    Unfiltered, the curator's own transcript reads as a conversation that has
    gone quiet — so it reflects on itself, spawns another curator, and only
    stops when a transcript happens to fall under the message floor.

    Both guards matter and they cover different paths: the session key is
    available on the event, and the category is what the hourly sweep sees when
    it meets a finished curator's conversation with no event to inspect.
    """
    from runtime.subagents import SESSION_PREFIX

    source = _source_or_skip(TASK)
    assert SESSION_PREFIX in source, "the event path must skip child sessions"
    assert "<> 'Subagent'" in source, "the sweep path must skip child conversations"


def test_installing_the_bundle_does_not_reflect_on_the_whole_archive():
    """The watermark defaults to zero, so history reads as new.

    On a fresh install every conversation ever held qualifies at once: nobody
    has reflected on them, so every message counts as unreflected. Observed as
    six runs draining years of conversations three at a time and writing notes
    about work from months ago as though it had just happened.

    A recency window rather than a one-off backfill guard, because it keeps
    being true — a conversation abandoned last spring must not become a
    candidate the day somebody opens it to read.
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
    cancelled, or a conversation opened only to run slash commands. The curator
    would then be handed a transcript with no agent in it and asked what should
    be done differently next time, which is a subagent spawned to answer an
    unanswerable question.
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
    assert "db.write" not in declared["requests"]
    assert "agent.spawn" in declared["requests"]


def test_the_event_task_implements_the_event_entry_point():
    """``run_event``, not ``run`` — implementing the wrong one fails silently.

    The task template still shows an event task overriding ``run``, so this is
    a trap worth a test rather than a comment.
    """
    source = _source_or_skip(TASK)
    assert "def run_event(self, sdk, payload)" in source
    assert "def run(self, sdk" not in source


def test_the_bundle_lists_both_halves_and_what_they_retrieve_through():
    """A bundle missing the search chain installs a memory that never recalls."""
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
    assert {SERVICE, TASK, "tools/tool_hybrid_search.py"} <= set(files)
