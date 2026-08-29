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
            # Same shape as the real one, because the tests below assert on
            # what a person reads rather than on the call being made.
            "quote": staticmethod(lambda text: "\n".join(
                f"> {line}" if line.strip() else ">"
                for line in (text or "").splitlines())),
        })()
        self.llm = type("Llm", (), {
            # ``list`` grew the setup questions, so the fake answers them.
            # ``params`` is what the extra-parameter menu is built from.
            "list": staticmethod(lambda **kw: {
                "profiles": [], "backends": [],
                "providers": ([
                    {"id": "acme", "label": "Acme",
                     "endpoint": "https://acme.test/v1",
                     "description": "Reads `ACME_API_KEY` from the environment."},
                ] if kw.get("providers") else []),
                "info": ({"context_size": 4096,
                          "description": "A chat model served by acme."}
                         if kw.get("info") else {}),
                "params": ([
                    {"name": "reasoning_effort", "label": "Reasoning effort",
                     "kind": "choice",
                     "choices": ["low", "medium", "high"],
                     "supported": True, "note": "",
                     "description": "Constrains effort on reasoning."},
                    {"name": "temperature", "label": "Temperature",
                     "kind": "number", "choices": [],
                     "supported": True, "note": "",
                     "description": "What sampling temperature to use."},
                    # Set on the profile below, so the *menu* hides it while
                    # the value step must still explain it.
                    {"name": "top_p", "label": "Top-p", "kind": "number",
                     "choices": [], "supported": True, "note": "",
                     "description": "Nucleus sampling."},
                    # A provider's own parameter: in no spec, so no prose.
                    {"name": "thinking", "label": "thinking", "kind": "text",
                     "choices": [], "supported": True, "note": "",
                     "description": ""},
                ] if kw.get("params") else []),
            }),
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
    """Parameters are edit-only, and that is structural rather than a filter:
    ``PROFILE_FIELDS`` is what ``_profile`` builds a new profile *from*, so a
    name in it is a key written to every profile whether asked for or not."""
    module = _load("command_llm")

    profile = module._profile({"llm_endpoint": "http://x",
                               "llm_context_size": 8})

    assert "llm_extra_params" not in profile
    # Still reachable afterwards, as the menu's own entry.
    names, _labels = module._fields(profile)
    assert module.EXTRA_PARAM in names

def test_the_edit_menu_labels_line_up_with_its_fields():
    """Two parallel lists, and a mismatch renames every field after the gap.

    Built per profile now rather than declared, because a configured
    parameter is an entry in it.
    """
    module = _load("command_llm")

    bare, bare_labels = module._fields({})
    assert len(bare) == len(bare_labels)
    assert not hasattr(module, "FIELDS")          # no longer a constant
    # The effort ladder is not a constant here either. It arrives per model on
    # the backend's ``reasoning_effort`` row.
    assert not hasattr(module, "REASONING_LEVELS")

    tuned, tuned_labels = module._fields(
        {"llm_extra_params": {"temperature": 0.2, "reasoning_effort": None}})
    assert len(tuned) == len(tuned_labels)
    assert len(tuned) == len(bare) + 2

def test_effort_is_one_parameter_among_the_others_now():
    """It used to be a field of its own, because it was the only member of
    ``llm_extra_params`` anybody could name. The backend names the rest now,
    so the special case stopped paying for itself — and a menu offering
    reasoning while hiding ``temperature`` teaches that reasoning is the only
    thing there is.

    What has not changed is where the value lands: in the dict, beside its
    neighbours, never in a key of its own."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit", "field": module.EXTRA_PARAM,
        "extra_param": "reasoning_effort", "extra_value": "high"})

    assert sdk._config["llm_profiles"]["m"] == {
        "llm_extra_params": {"reasoning_effort": "high"}}

    # And the same route sets anything else the model takes.
    module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit", "field": module.EXTRA_PARAM,
        "extra_param": "temperature", "extra_value": "0.2"})

    assert sdk._config["llm_profiles"]["m"]["llm_extra_params"] == {
        "reasoning_effort": "high", "temperature": 0.2}


def test_no_word_is_read_as_an_instruction():
    """The word was the bug. A free-text step told you to answer ``remove``
    while ``run`` compared against a different string, so typing it set the
    parameter *to* "remove" — and ``off`` had the same shape of problem in the
    other direction, since ``off`` is a plausible value for a real provider
    parameter and reading it as an instruction made that value unsettable.

    Declining a parameter is Remove, asked as its own question. Nothing typed
    into a value means anything but itself."""
    module = _load("command_llm")

    assert module._extra_value({"extra_value": "off"}) == "off"
    assert module._extra_value({"extra_value": "remove"}) == "remove"
    assert not hasattr(module, "SEND_NOTHING")

def test_a_configured_parameter_is_an_entry_you_can_open():
    """The point of the restructure: a parameter this profile sends sits in
    the same menu as its endpoint and context size, showing its value, so
    finding out what a profile does no longer means reading a JSON blob."""
    module = _load("command_llm")

    names, labels = module._fields(
        {"llm_extra_params": {"temperature": 0.2, "reasoning_effort": None}})
    entries = dict(zip(names, labels))

    assert entries[module.PARAM_PREFIX + "temperature"] == "temperature = 0.2"
    assert entries[module.PARAM_PREFIX + "reasoning_effort"] == (
        "reasoning_effort = (sends nothing)")
    # The prefix is what stops a provider parameter impersonating a field.
    assert module._param_field(module.PARAM_PREFIX + "temperature") == (
        "temperature")
    assert module._param_field("llm_endpoint") == ""

def test_removing_the_last_parameter_removes_the_key():
    """An empty dict reads as configured to anything scanning config by hand,
    and every profile written before extras existed carries nothing."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {"llm_extra_params": {"temperature": 0.2}}},
                  default="m")

    module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit",
        "field": module.PARAM_PREFIX + "temperature",
        "param_action": module.REMOVE})

    assert sdk._config["llm_profiles"]["m"] == {}


def test_changing_and_removing_are_asked_before_either_happens():
    """Two questions where one nearly did. The single-question version had to
    carry "delete" as a word inside a free-text value, which is how it came to
    set the parameter to the string "remove"."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {"llm_extra_params": {"reasoning_effort": "high"}}},
                  default="m")
    # The shared fake's ``card`` answers with pairs, which the other tests
    # read; this one drives the real form, which concatenates it into a
    # prompt.
    sdk.md.card = staticmethod(lambda title, pairs: "")

    steps = {step["name"]: step for step in module.LlmCommand().form(sdk, {
        "model_name": "m", "action": "edit",
        "field": module.PARAM_PREFIX + "reasoning_effort"})}

    assert steps["param_action"]["enum"] == [module.EDIT, module.REMOVE]
    # The value is not asked for until the answer says it is wanted.
    assert "extra_value" not in steps

    steps = {step["name"]: step for step in module.LlmCommand().form(sdk, {
        "model_name": "m", "action": "edit",
        "field": module.PARAM_PREFIX + "reasoning_effort",
        "param_action": module.EDIT})}
    assert "extra_value" in steps

def test_the_card_says_what_the_model_will_actually_do():
    """Every parameter gets its own row, so each can carry its own caveat.

    It used to be one JSON blob with reasoning promoted out of it into a row
    of its own — unreadable at a glance, and the promotion was the only reason
    one parameter could carry a warning and the rest could not. There is no
    "Reasoning" row now, because there is no parameter the kernel supplies and
    therefore nothing to report about a profile that sets none."""
    module = _load("command_llm")
    sdk = _LlmSdk({})

    def card(profile):
        return module._describe(sdk, {}, {"m": {"llm_service_class": "X",
                                                **profile}}, "m", "m")

    assert not [pair for pair in card({}) if pair[0] == "Reasoning"]

    tuned = card({"llm_extra_params": {"reasoning_effort": "low",
                                       "temperature": 0.2,
                                       "thinking": None}})
    assert ("reasoning_effort", "low") in tuned
    assert ("temperature", "0.2") in tuned
    assert ("thinking", "(sends nothing)") in tuned

def test_a_value_step_is_shaped_by_what_the_backend_said():
    """The parameter menu is per model, so the *value* question is too.

    A closed set of values becomes a picker of exactly those values; anything
    else is free text with no magic words in it.
    """
    module = _load("command_llm")

    sdk = _LlmSdk({}, default="")

    choice = module._value_step(
        sdk, "reasoning_effort",
        {"kind": "choice", "choices": ["low", "high"], "note": ""}, {})
    assert choice["enum"] == ["low", "high"]

    number = module._value_step(sdk, "temperature", {"kind": "number"}, {})
    assert not number["enum"]
    assert "off" not in number["prompt"] and "remove" not in number["prompt"]

    # A parameter the backend could not describe is still settable.
    unknown = module._value_step(sdk, "enable_thinking", None, {})
    assert not unknown["enum"]
    assert "off" not in unknown["prompt"] and "remove" not in unknown["prompt"]

def test_reserved_params_are_refused_where_somebody_typed_them():
    """The backend merges extras with ``setdefault``, so one of these wins
    over the profile *silently* — and an ``api_key`` here also lands in
    plaintext config instead of behind the ``secret_`` prefix that declares it
    a credential.

    Checked on the typed name, since that is now the only way one of these can
    be named at all: the menu offers what the model takes, and a reserved key
    can only arrive through "Something else"."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    answer = module.LlmCommand().run(sdk, {
        "model_name": "m", "action": "edit", "field": module.EXTRA_PARAM,
        "extra_param": module.CUSTOM, "custom_param_name": "api_key",
        "extra_value": "sk-oops"})

    assert "api_key" in answer and "API key field" in answer
    assert sdk._config["llm_profiles"]["m"] == {}   # nothing was written

def test_the_reserved_list_covers_what_the_backend_merges():
    """Two connection settings and the three parts of the call itself."""
    module = _load("command_llm")

    assert set(module.RESERVED_PARAMS) == {
        "api_key", "api_base", "model", "messages", "tools", "stream"}



def test_a_value_arrives_as_the_type_it_looks_like():
    """JSON where it parses, so a number is a number and an object is an
    object; a bare word is not valid JSON and stays the text it was.

    This used to also read ``off`` and ``null`` as instructions. ``null``
    needs no special case — JSON already reads it — and ``off`` was a bug."""
    module = _load("command_llm")

    assert module._extra_value({"extra_value": "0.2"}) == 0.2
    assert module._extra_value({"extra_value": "true"}) is True
    assert module._extra_value({"extra_value": "null"}) is None
    assert module._extra_value({"extra_value": '{"type": "enabled"}'}) == {
        "type": "enabled"}
    assert module._extra_value({"extra_value": "medium"}) == "medium"



# ──────────────────────────────────────────────────────────────────────
# Saying what a thing is, in the backend's voice.
# ──────────────────────────────────────────────────────────────────────

def _prompt(steps, field):
    """The prompt of one step, by field name."""
    for step in steps:
        if step["name"] == field:
            return step["prompt"]
    seen = [step["name"] for step in steps]
    raise AssertionError(f"no {field} step in {seen}")


def test_a_parameter_step_says_what_the_parameter_is():
    """The value step asked for a number and never said what the number did.

    Every one of these came out of the backend's own tables, so the command
    learns no provider vocabulary to render them — which is the same rule
    ``choices`` and ``note`` already follow.
    """
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    steps = module.LlmCommand().form(sdk, {
        "model_name": "m", "action": "edit", "field": module.EXTRA_PARAM,
        "extra_param": "temperature"})

    assert "> What sampling temperature to use." in _prompt(steps,
                                                            "extra_value")


def test_a_description_is_quoted_rather_than_asserted():
    """It is somebody else's sentence — read out of whatever the provider
    library documents — so it may describe the spec a parameter belongs to
    rather than the model in front of you. A blockquote makes that claim
    honestly; folding it into our own prose would not."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    assert module._about(sdk, {"description": "Two\nlines."}) == (
        "\n\n> Two\n> lines.")


def test_nothing_is_rendered_for_a_parameter_no_spec_names():
    """The ordinary answer, and the one this must not treat as a problem: a
    provider's own parameter appears in no spec, and a backend need not
    implement any of this. The step then reads exactly as it did before
    descriptions existed — no stray quote, no empty gap."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {}}, default="m")

    assert module._about(sdk, {"description": ""}) == ""
    assert module._about(sdk, {}) == ""
    assert module._about(sdk, None) == ""

    steps = module.LlmCommand().form(sdk, {
        "model_name": "m", "action": "edit", "field": module.EXTRA_PARAM,
        "extra_param": "thinking"})

    assert ">" not in _prompt(steps, "extra_value")


def test_a_parameter_already_set_is_still_explained():
    """Opening one from the edit menu takes the other route into the value
    step — the one that looks the spec up itself — and both have to carry the
    description or the explanation depends on how you arrived."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {"llm_extra_params": {"top_p": 0.9}}}, default="m")
    # The shared fake's ``card`` answers with pairs; this drives a real
    # form, whose profile step concatenates it.
    sdk.md.card = staticmethod(lambda title, pairs: "")

    steps = module.LlmCommand().form(sdk, {
        "model_name": "m", "action": "edit",
        "field": module.PARAM_PREFIX + "top_p",
        "param_action": module.EDIT})

    prompt = _prompt(steps, "extra_value")
    assert "Currently `0.9`" in prompt
    assert "> Nucleus sampling." in prompt


def test_a_typed_parameter_name_is_explained_too():
    """The menu hides what this profile already sets, which is a rule about
    what is worth *suggesting*. Resolving the spec against those same rows
    let that rule decide what could be explained, so typing the name of a
    parameter the backend knows perfectly well got you a bare prompt."""
    module = _load("command_llm")
    sdk = _LlmSdk({"m": {"llm_extra_params": {"top_p": 0.9}}}, default="m")
    # The shared fake's ``card`` answers with pairs; this drives a real
    # form, whose profile step concatenates it.
    sdk.md.card = staticmethod(lambda title, pairs: "")

    steps = module.LlmCommand().form(sdk, {
        "model_name": "m", "action": "edit", "field": module.EXTRA_PARAM,
        "extra_param": module.CUSTOM, "custom_param_name": "top_p"})

    assert "> Nucleus sampling." in _prompt(steps, "extra_value")


def test_setup_says_what_the_model_is_where_it_asks_about_the_model():
    """The context-size step is the first question that is about the model
    rather than about reaching it, and the three after it ask what the model
    can read. So the description belongs on it: everything it answers is
    about to be asked."""
    module = _load("command_llm")
    sdk = _LlmSdk({}, default="")

    steps = module.LlmCommand().form(sdk, {
        "model_name": "add", "new_model_name": "acme/one",
        "llm_endpoint": "https://acme.test/v1"})

    assert "> A chat model served by acme." in _prompt(steps,
                                                       "llm_context_size")


def test_setup_names_the_environment_variable_it_falls_back_to():
    """The key step tells you a blank works when the key is already in the
    environment "under the name this provider looks for" — a name the
    sentence could never supply, since only the backend knows it. It is
    quoted where the answer changes what you type."""
    module = _load("command_llm")
    sdk = _LlmSdk({}, default="")

    steps = module.LlmCommand().form(sdk, {
        "model_name": "add", "llm_provider": "acme"})

    assert "> Reads `ACME_API_KEY` from the environment." in _prompt(
        steps, "secret_llm_api_key")
    # And the same call still answers the endpoint it was always asked for.
    assert "https://acme.test/v1" in _prompt(steps, "llm_endpoint")
