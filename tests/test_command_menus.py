"""Menus that tell the truth: state on the label, and only the useful half.

Two habits had drifted apart across the command tree. ``/services`` built its
action list from the service's live state, so it offered "Load it" *or*
"Unload it" and never both. Everything else declared a module-level constant
list, so both halves of every toggle were always on screen and picking the
wrong one looked like a broken command rather than a redundant option. And no
picker showed state at all, so learning which of six services were running
took six round trips.

These test the pure helpers rather than a live form, because that is where the
decision is — and it keeps the assertions readable.
"""

import importlib.util
from pathlib import Path

import sandbox  # noqa: F401  - installs the ``guest`` package alias

_COMMANDS = Path(__file__).resolve().parents[1] / "bundled" / "commands"


def _load(stem):
    """Import one command module for its helpers."""
    spec = importlib.util.spec_from_file_location(
        f"_menu_{stem}", _COMMANDS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ──────────────────────────────────────────────────────────────────────
# Only the half of a toggle that does something.
# ──────────────────────────────────────────────────────────────────────

def test_frontends_offers_one_half_of_the_enable_toggle():
    """Both were always shown, and one was always a no-op."""
    module = _load("command_frontends")

    on, _ = module._actions_for({"repl"}, "repl")
    off, _ = module._actions_for({"repl"}, "telegram")

    assert "disable" in on and "enable" not in on
    assert "enable" in off and "disable" not in off


def test_schedule_offers_one_half_of_the_enable_toggle():
    """Same shape, and the job's enabled state was already read one function
    away to build its label."""
    module = _load("command_schedule")

    on, on_labels = module._actions_for({"enabled": True})
    off, off_labels = module._actions_for({"enabled": False})

    assert "disable" in on and "enable" not in on
    assert "enable" in off and "disable" not in off
    # Delete survives either way; this narrows the pair, not the menu.
    assert "delete" in on and "delete" in off
    assert "Disable it" in on_labels and "Enable it" in off_labels


def test_tasks_offers_one_half_of_the_pause_toggle():
    """``/tasks`` already branched on ``trigger``, so the mechanism was there
    — nothing simply read ``paused``."""
    module = _load("command_tasks")

    running, running_labels = module._actions_for({"trigger": "path"})
    paused, _ = module._actions_for({"trigger": "path", "paused": True})

    assert "pause" in running and "unpause" not in running
    assert "unpause" in paused and "pause" not in paused
    # Labels were the raw action strings, so the menu read in machine words.
    assert "Pause it" in running_labels
    assert "pause" not in running_labels


def test_tasks_keeps_the_trigger_specific_actions_apart():
    """A path task resets and retries; an event task is triggered."""
    module = _load("command_tasks")

    path, _ = module._actions_for({"trigger": "path"})
    event, _ = module._actions_for({"trigger": "event"})

    assert {"reset", "retry"} <= set(path) and "trigger" not in path
    assert "trigger" in event and not {"reset", "retry"} & set(event)


def test_llm_offers_one_half_of_the_load_toggle_and_hides_a_dead_default():
    """The unload branch used to be unreachable in practice anyway: it read
    ``loaded`` off the service registry, which has never held a profile."""
    module = _load("command_llm")
    registry = {"profiles": [
        {"model_name": "open", "loaded": True},
        {"model_name": "shut", "loaded": False},
    ]}

    on, on_labels = module._actions_for(registry, "open", "open")
    off, _ = module._actions_for(registry, "open", "shut")

    assert "unload" in on and "load" not in on
    assert "load" in off and "unload" not in off
    assert "Unload it" in on_labels
    # "Set default" on the profile that already is the default did nothing.
    assert "set_default" not in on
    assert "set_default" in off


# ──────────────────────────────────────────────────────────────────────
# State on the label, so a picker is readable without opening anything.
# ──────────────────────────────────────────────────────────────────────

def test_a_service_picker_shows_state_without_a_round_trip():
    """The status table existed and was reachable only from the code path the
    interactive form never takes."""
    module = _load("command_services")

    def label(loaded):
        """Render one managed service's picker label."""
        return module._service_label(
            {"name": "x", "loaded": loaded, "lifecycle": "managed"})

    assert label(True).startswith("●")
    assert label(False).startswith("○")


def test_a_service_extension_reads_as_neither_loaded_nor_unloaded():
    """An extension has no lifecycle to offer, so a load marker would lie."""
    module = _load("command_services")

    label = module._service_label(
        {"name": "x", "loaded": True, "lifecycle": "extension"})

    assert not label.startswith("●") and not label.startswith("○")


def test_the_services_picker_carries_the_whole_status_table():
    module = _load("command_services")

    prompt = module._select_prompt([
        {"name": "timekeeper", "loaded": True, "lifecycle": "managed",
         "model_name": ""},
        {"name": "compactor", "loaded": False, "lifecycle": "managed",
         "model_name": ""},
    ])

    assert "timekeeper" in prompt and "compactor" in prompt
    assert "Loaded" in prompt and "Unloaded" in prompt


def test_an_llm_picker_marks_the_open_pools():
    module = _load("command_llm")
    registry = {"profiles": [
        {"model_name": "open", "loaded": True},
        {"model_name": "shut", "loaded": False},
    ]}

    assert module._model_label(registry, "open", "open").startswith("●")
    assert module._model_label(registry, "open", "shut").startswith("○")
    # The default is still called out, as it always was.
    assert "(default)" in module._model_label(registry, "open", "open")


# ──────────────────────────────────────────────────────────────────────
# /schedule: ask what kind of job, not which channel.
# ──────────────────────────────────────────────────────────────────────

class _TaskSdk:
    """Enough SDK to enumerate schedulable targets."""

    def __init__(self, tasks):
        self.tasks = _TaskList(tasks)


class _TaskList:
    """The ``sdk.tasks`` namespace, list only."""

    def __init__(self, tasks):
        self._tasks = tasks

    def list(self, details=False):
        """Every registered task."""
        return self._tasks


_EVENT_TASK = {"name": "index_lexical", "trigger": "event",
               "trigger_channels": ["file.indexed"],
               "event_payload_schema": {}}


def test_scheduling_asks_the_kind_before_anything_else():
    """The old first step mixed ``background agent`` in among task names, so
    the only clue the first was a different sort of thing was the space."""
    module = _load("command_schedule")

    steps = module._add_steps(_TaskSdk([_EVENT_TASK]), {})

    assert [step["name"] for step in steps] == ["kind"]
    assert set(steps[0]["enum"]) == {module.AGENT_KIND, module.TASK_KIND}


def test_choosing_the_agent_never_asks_which_task():
    """A form is cumulative, so ``kind`` stays at the head — what matters is
    that ``target`` never appears and the payload steps do."""
    module = _load("command_schedule")

    steps = module._add_steps(_TaskSdk([]), {"kind": module.AGENT_KIND})
    names = [step["name"] for step in steps]

    assert "target" not in names
    assert names[:3] == ["kind", "new_job_name", "cron"]
    assert "prompt" in names and "title" in names


def test_choosing_a_task_asks_which_one_before_going_further():
    """Nothing else can be asked yet: the payload schema depends on which
    task, so the form stops here until it is answered."""
    module = _load("command_schedule")

    steps = module._add_steps(_TaskSdk([_EVENT_TASK]), {"kind": "task"})

    assert [step["name"] for step in steps] == ["kind", "target"]
    assert steps[1]["enum"] == ["index_lexical"]


def test_a_bare_kernel_does_not_offer_a_task_kind_it_cannot_fulfil():
    module = _load("command_schedule")

    steps = module._add_steps(_TaskSdk([]), {})

    assert steps[0]["enum"] == [module.AGENT_KIND]


def test_a_scheduled_agent_with_no_prompt_is_refused_at_creation():
    """It used to be accepted, then fail inside ``spawn`` with "prompt is
    required" and push that into the chat once per firing forever."""
    module = _load("command_schedule")

    missing = module._missing_required(module.SUBAGENT_SCHEMA, {"title": "hi"})

    assert missing == ["prompt"]
    assert module._missing_required(
        module.SUBAGENT_SCHEMA, {"prompt": "do the thing"}) == []


# ──────────────────────────────────────────────────────────────────────
# /commands: four sections that each hold something.
# ──────────────────────────────────────────────────────────────────────

def test_every_declared_section_is_used_and_every_command_has_one():
    """There were six sections; two were never used by any command
    ("Services & Tools", "Other") and two overlapped so plainly that /config
    sat in "Config & System" with /setup one section away in "System"."""
    import re

    from plugins.command_registry import _HELP_SECTIONS

    module = _load("command_commands")
    assert module._HELP_SECTIONS == _HELP_SECTIONS, (
        "the two copies of the section order have drifted")

    found = set()
    for path in sorted(_COMMANDS.glob("command_*.py")):
        match = re.search(r'^\s*category = "([^"]*)"',
                          path.read_text(encoding="utf-8"), re.M)
        assert match, f"{path.name} declares no category"
        assert match.group(1) in _HELP_SECTIONS, (
            f"{path.name} is filed under {match.group(1)!r}, which no longer "
            "exists")
        found.add(match.group(1))

    assert found == set(_HELP_SECTIONS), (
        f"declared but unused: {set(_HELP_SECTIONS) - found}")


def test_ending_the_app_is_not_filed_under_conversation():
    """/quit and /restart end the *process*; under "Conversation" they read as
    ways to end a chat, beside /clear and /new."""
    import re

    for stem in ("command_quit", "command_restart"):
        source = (_COMMANDS / f"{stem}.py").read_text(encoding="utf-8")
        assert re.search(r'^\s*category = "System"', source, re.M), stem
