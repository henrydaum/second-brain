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

def test_a_picker_option_is_a_name_and_nothing_else():
    """The state markers are gone from every picker.

    They encoded status in a glyph the REPL prints into its `(options: ...)`
    hint, where it is noise a person cannot type - and the same status is now
    a column in the table each picker carries in its prompt, which both
    frontends render. Pinned as a negative because the temptation to decorate
    a label is what put them there the first time.
    """
    markers = "●○·⏸"
    services = _load("command_services")
    frontends = _load("command_frontends")
    llm = _load("command_llm")

    labels = [
        llm._model_label("open", "open"),
        llm._model_label("open", "shut"),
        llm._model_label("open", "add"),
    ]
    assert not any(mark in label for label in labels for mark in markers)
    # The words survive; only the glyphs went.
    assert "(default)" in labels[0]
    assert not hasattr(services, "_service_label")
    assert not hasattr(frontends, "_frontend_label")
    assert not hasattr(_load("command_tasks"), "_task_label")


def test_the_services_picker_carries_the_whole_status_table():
    module = _load("command_services")

    prompt = module._select_prompt([
        {"name": "timekeeper", "loaded": True, "lifecycle": "managed"},
        {"name": "compactor", "loaded": False, "lifecycle": "managed"},
    ])

    assert "timekeeper" in prompt and "compactor" in prompt
    assert "Loaded" in prompt and "Unloaded" in prompt


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


# ──────────────────────────────────────────────────────────────────────
# /tasks: the overview, and the two actions that reach every task.
# ──────────────────────────────────────────────────────────────────────

class _Md:
    """The two ``sdk.md`` helpers these renderers use, as the guest sees them."""

    @staticmethod
    def table(headers, rows, leading_blank=True):
        body = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
        return " | ".join(headers) + "\n" + body

    @staticmethod
    def quote(text):
        return "\n".join(f"> {line}" for line in text.splitlines())


class _Sdk:
    md = _Md()


def _task(name, **kwargs):
    task = {"name": name, "trigger": "path", "paused": False, "counts": {}}
    task.update(kwargs)
    return task


def test_the_tasks_table_shows_every_status_at_a_glance():
    """The six columns exist so tasks can be *compared* — which is stuck,
    which is behind, which is paused. The renderer was reachable only from the
    no-argument path, i.e. never from the menu."""
    module = _load("command_tasks")

    table = module._show(_Sdk(), [
        _task("extract", counts={"PENDING": 2, "PROCESSING": 1,
                                 "DONE": 40, "FAILED": 3}),
        _task("embed", paused=True),
    ])

    assert "Task | Status | Pending | Running | Done | Failed" in table
    assert "extract | Active | 2 | 1 | 40 | 3" in table
    # A task with no rows reads as zero, not as blank: absent means none.
    assert "embed | Paused | 0 | 0 | 0 | 0" in table


def test_bulk_options_state_their_cost_on_the_option():
    """"Reset all tasks" is one click from re-running the pipeline over every
    indexed file, so the number of rows belongs in the thing being chosen."""
    module = _load("command_tasks")
    tasks = [
        _task("extract", counts={"PENDING": 2, "DONE": 40, "FAILED": 3}),
        _task("index", counts={"FAILED": 5, "DONE": 10}),
        # Event tasks have no reset at all, so they must not be counted.
        _task("digest", trigger="event", counts={"FAILED": 99}),
    ]

    retry, reset = module._bulk_labels(tasks)

    assert "8 rows" in retry            # 3 + 5, and not the event task's 99
    assert "60 rows" in reset           # 45 + 15
    assert module._bulk_totals(tasks)[2] == 2


def test_a_bulk_reset_skips_event_tasks_and_reports_what_it_did():
    module = _load("command_tasks")
    reset = []

    class Tasks:
        @staticmethod
        def reset(name, failed_only=False):
            reset.append((name, failed_only))

    sdk = _Sdk()
    sdk.tasks = Tasks()
    tasks = [_task("extract", counts={"DONE": 4}),
             _task("digest", trigger="event", counts={"DONE": 9})]

    message = module._run_bulk(sdk, tasks, "reset_all")

    assert reset == [("extract", False)]
    assert "4 row" in message


def test_a_bulk_retry_with_nothing_failed_does_nothing():
    """An empty run must say so rather than reporting a successful no-op."""
    module = _load("command_tasks")

    class Tasks:
        @staticmethod
        def reset(name, failed_only=False):
            raise AssertionError("nothing had failed")

    sdk = _Sdk()
    sdk.tasks = Tasks()

    assert "Nothing has failed" in module._run_bulk(
        sdk, [_task("extract", counts={"DONE": 3})], "retry_all")


# ──────────────────────────────────────────────────────────────────────
# /llm's tuning fields: editable, never asked for, and unset by absence.
# ──────────────────────────────────────────────────────────────────────

class _LlmSdk:
    """Enough of an SDK for ``/llm`` to read and write config."""

    class Failed(Exception):
        """The base the command catches."""

    def __init__(self, profiles, default=""):
        self._config = {"llm_profiles": profiles, "default_llm_profile": default}
        self.md = type("Md", (), {
            "card": staticmethod(lambda title, pairs: pairs),
            "table": staticmethod(lambda *a, **k: ""),
        })()
        self.llm = type("Llm", (), {
            "list": staticmethod(lambda: {"profiles": [], "backends": []}),
            "load": staticmethod(lambda name: True),
        })()

    @property
    def config(self):
        """``read``/``write`` over the dict this fake holds."""
        outer = self

        class Config:
            @staticmethod
            def read(key):
                return outer._config.get(key)

            @staticmethod
            def write(key, value, scope=None):
                outer._config[key] = value

        return Config


def test_adding_a_profile_never_writes_the_tuning_keys():
    """They are edit-only, and that is structural rather than a filter:
    ``PROFILE_FIELDS`` is what ``_profile`` builds a new profile *from*, so a
    name in it is a key written to every profile whether asked for or not."""
    module = _load("command_llm")

    profile = module._profile({"llm_endpoint": "http://x", "llm_context_size": 8})

    assert not set(module.TUNING_FIELDS) & set(profile)
    assert "llm_extra_params" not in profile
    assert set(module.TUNING_FIELDS) <= set(module.FIELDS)  # still editable


def test_the_edit_menu_labels_line_up_with_its_fields():
    """Two parallel lists, and a mismatch renames every field after the gap."""
    module = _load("command_llm")

    assert len(module.FIELDS) == len(module.FIELD_LABELS)
    assert len(module.REASONING_LEVELS) == len(module.REASONING_LABELS)
    assert module.REASONING_LEVELS[0] == module.OFF


def test_effort_is_a_menu_entry_over_the_extras_dict():
    """The same relationship ``llm_capability_image`` has with
    ``llm_capabilities``: a picker beats remembering a vocabulary and JSON
    syntax to set one member, and the value still belongs with its
    neighbours rather than in a key of its own."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit",
        "field": "llm_reasoning_effort", "value": "High"})

    assert sdk._config["llm_profiles"]["m"] == {
        "llm_extra_params": {"reasoning_effort": "high"}}
    assert "llm_reasoning_effort" not in sdk._config["llm_profiles"]["m"]


def test_off_is_stored_as_a_null_not_as_the_word():
    """``none`` is a real level several providers accept, meaning "think as
    little as possible". Storing "off" beside it would be two spellings
    nobody could tell apart in a config file."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit",
        "field": "llm_reasoning_effort", "value": module.OFF})

    assert sdk._config["llm_profiles"]["m"] == {
        "llm_extra_params": {"reasoning_effort": None}}


def test_clearing_the_extras_dict_removes_the_key():
    """An empty dict reads as configured to anything scanning config by hand,
    and every profile written before this existed carries nothing."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {"llm_extra_params": {"temperature": 0.2}}}, default="m")

    module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit",
        "field": "llm_extra_params", "value": {}})

    assert sdk._config["llm_profiles"]["m"] == {}


def test_extra_params_are_parsed_by_the_form_not_the_handler():
    """Declaring the step ``object`` is what makes bad JSON re-ask at the
    step. Returning a sentence from ``run`` would mean re-running the whole
    command to fix a typo."""
    module = _load("command_llm")

    assert module._value_type("llm_extra_params") == "object"
    assert module._coerce("llm_extra_params", {"temperature": 0.2}) == {"temperature": 0.2}
    assert module._coerce("llm_extra_params", "not a dict") == {}


def test_the_card_says_what_the_model_will_actually_do():
    """Reasoning is always on the card now, because there is always an answer:
    a profile that says nothing still thinks at whatever the kernel supplies,
    and a card that stayed silent would be the only place you could not find
    that out."""
    module = _load("command_llm")
    sdk = _LlmSdk({})

    def card(profile):
        return module._describe(sdk, {}, {"m": {"llm_service_class": "X",
                                                **profile}}, "m", "m")

    assert ("Reasoning", "default") in card({})
    assert ("Reasoning", "high") in card(
        {"llm_extra_params": {"reasoning_effort": "high"}})
    assert ("Reasoning", "off (nothing sent)") in card(
        {"llm_extra_params": {"reasoning_effort": None}})
    # The effort has its own row, so it is not repeated in the raw dump.
    tuned = card({"llm_extra_params": {"reasoning_effort": "low",
                                       "temperature": 0.2}})
    assert ("Extra params", '{"temperature": 0.2}') in tuned


def test_the_kernel_coerces_what_the_two_tuning_steps_declare():
    """The seam between what ``/llm`` declares and what the form does with it.

    Both fields lean on kernel behaviour rather than parsing anything
    themselves — the enum resolves a label or a differently-cased word, and
    ``object`` parses the JSON and rejects a typo at the step. Neither is
    visible from the command's own helpers, and a wrong ``type`` string would
    fail by quietly storing the raw text.
    """
    from state_machine.conversation import FormStep as KernelStep

    from guest.forms import FormStep

    module = _load("command_llm")

    def rebuilt(field):
        return KernelStep.from_dict(dict(FormStep(
            "value", "p", True, module._value_type(field),
            enum=module._value_enum(field),
            enum_labels=module._value_enum_labels(field))))

    effort = rebuilt("llm_reasoning_effort")
    assert effort.coerce("High") == "high"
    assert effort.coerce("Off — send nothing") == module.OFF
    assert effort.validate("turbo")[0] is False

    extras = rebuilt("llm_extra_params")
    assert extras.coerce('{"temperature": 0.2}') == {"temperature": 0.2}
    assert extras.validate('{"temperature":')[0] is False




def test_reserved_params_are_refused_where_somebody_typed_them():
    """The backend merges extras with ``setdefault``, so one of these wins
    over the profile *silently* — and an ``api_key`` here also lands in
    plaintext config instead of behind the ``secret_`` prefix that declares it
    a credential."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    answer = module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit", "field": "llm_extra_params",
        "value": {"api_key": "sk-oops", "temperature": 0.2}})

    assert "api_key" in answer and "API key field" in answer
    assert "temperature" not in answer          # only the offenders are named
    assert sdk._config["llm_profiles"]["m"] == {}   # nothing was written


def test_the_reserved_list_covers_what_the_backend_merges():
    """Two connection settings and the three parts of the call itself."""
    module = _load("command_llm")

    assert set(module.RESERVED_PARAMS) == {
        "api_key", "api_base", "model", "messages", "tools", "stream"}


def test_the_command_and_the_kernel_agree_on_the_off_spelling():
    """A command is guest code and cannot import ``llm``, so the coupling is a
    literal on each side. A mismatch would store a word the kernel reads as a
    level and send it to a provider as one."""
    import llm

    module = _load("command_llm")

    assert module.OFF == llm.OFF_EFFORT
    assert module._coerce("llm_reasoning_effort", module.OFF) is None
