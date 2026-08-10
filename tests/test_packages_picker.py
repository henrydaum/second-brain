"""Picking a package instead of typing one.

Installing something took two commands: ``/packages available`` printed a
category menu, then a table, and the name had to be *read off that table* and
retyped into ``/packages install <stem>``. The machinery to do it in one pass
was already there — ``form(sdk, args)`` is re-called after every answer, so a
later step's ``enum`` can be built from an earlier answer — and ``uninstall``
already used it for its picker. ``install`` was a bare text box.

Two properties matter beyond the cascade itself. The **one-argument spelling
still works**, because it is what the agent is told to type and what every
older invocation says; a stem lands in the ``category`` slot and the form has
to recognise it as a stem rather than prompt for a package inside a family
that does not exist. And the **card must not stand in front of that path**: an
agent meeting a missing required step gets a failure, not a form, so a confirm
step on the one-argument path would make the command uncallable by the agent
that documents it.

These drive the form's pure helpers against a fake catalogue rather than a
live box, because that is where the decision is.
"""

import importlib.util
from pathlib import Path

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox.guest.sdk import _Markdown

from bundled.commands.helpers.package_manager import read_dependency_meta

_COMMANDS = Path(__file__).resolve().parents[1] / "bundled" / "commands"


def _load(stem):
    spec = importlib.util.spec_from_file_location(
        f"_pkg_{stem}", _COMMANDS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Failed(Exception):
    def __init__(self, error=""):
        self.error = error


_AVAILABLE = [
    {"id": "tool_web_search", "path": "tools/tool_web_search.py",
     "family": "tools", "helper": False},
    {"id": "service_web", "path": "services/helpers/service_web.py",
     "family": "services", "helper": True},
]
_REMOVABLE = [
    {"id": "tool_old", "path": "tools/tool_old.py",
     "family": "tools", "helper": False},
]


class _Plugins:
    def list(self, source="registered", category="", role="",
             details=False, name=""):
        if source == "families":
            return ["tools", "services", "parsers", "bundles"]
        if source == "info":
            return {"id": name, "name": "web_search",
                    "description": "Search the web.", "family": "tools",
                    "path": f"tools/{name}.py",
                    "dependencies_files": ["services/service_web_search.py"],
                    "dependencies_pip": ["httpx"]}
        if source == "installed_info":
            return {"id": name, "name": name, "description": "", "family":
                    "tools", "path": f"tools/{name}.py",
                    "also_removes": ["tools/tool_dep.py"], "removes_pip": []}
        items = {"available": _AVAILABLE, "removable": _REMOVABLE}.get(source)
        if items is None:
            raise _Failed(f"unknown source {source}")
        return [item for item in items
                if not category or item["family"] == category]


class _SDK:
    Failed = _Failed
    plugins = _Plugins()
    md = _Markdown()


def _names(command, args):
    return [step["name"] for step in command.form(_SDK(), args)]


def _command():
    return _load("command_packages").PackagesCommand()


# ──────────────────────────────────────────────────────────────────────
# The cascade.
# ──────────────────────────────────────────────────────────────────────

def test_each_answer_opens_the_next_step():
    """Category, then that category's packages, then a card. No typing.

    ``stem`` sits in every list and is asked for by none of them — see
    ``test_the_stem_slot_is_never_shown_to_anybody``.
    """
    command = _command()

    assert _names(command, {}) == ["action"]
    assert _names(command, {"action": "install"}) == [
        "action", "stem", "category"]
    assert _names(command, {"action": "install", "category": "tools"}) == [
        "action", "stem", "category", "package_id"]
    assert _names(command, {"action": "install", "category": "tools",
                            "package_id": "tool_web_search"}) == [
        "action", "stem", "category", "package_id", "confirm"]


def test_the_package_step_offers_that_category_and_no_other():
    """The point of the category answer: options, not just a heading."""
    command = _command()

    steps = command.form(_SDK(), {"action": "install", "category": "tools"})

    assert steps[-1]["enum"] == ["tool_web_search"]


def test_uninstall_cascades_the_same_way_from_the_removable_list():
    command = _command()

    steps = command.form(_SDK(), {"action": "uninstall", "category": "tools"})

    assert [step["name"] for step in steps][-1] == "package_id"
    assert steps[-1]["enum"] == ["tool_old"]


def test_a_category_with_nothing_in_it_is_not_offered():
    """A button whose only outcome is an empty list is a dead end."""
    command = _command()

    steps = command.form(_SDK(), {"action": "install"})

    assert steps[-1]["enum"] == ["tools", "services"]
    assert "parsers" not in steps[-1]["enum"]


def test_update_asks_nothing_further():
    assert _names(_command(), {"action": "update"}) == ["action"]


# ──────────────────────────────────────────────────────────────────────
# The one-argument spelling, which is the agent's.
# ──────────────────────────────────────────────────────────────────────

def test_a_named_stem_ends_the_form():
    """``/packages install tool_web_search`` asks nothing further."""
    command = _command()

    assert _names(command, {"action": "install",
                            "stem": "tool_web_search"}) == ["action", "stem"]


def test_the_stem_slot_is_never_shown_to_anybody():
    """It exists to catch a command-line token, in a form the parser peels in
    step order. Neither required nor ``prompt_when_missing``, so no form ever
    opens it — and it is a *separate key* from the picker's because a step the
    parser skips gets filled with its default, which on one shared key would
    leave the picker permanently already-answered."""
    command = _command()

    stem = command.form(_SDK(), {"action": "install"})[1]

    assert stem["name"] == "stem"
    assert not stem["required"] and not stem.get("prompt_when_missing")


def test_the_card_never_stands_in_front_of_the_one_argument_path():
    """An agent that meets a missing required step gets a failure rather than
    a form, so a confirm step here would make the command uncallable by the
    agent its own ``agent_prompt`` instructs."""
    command = _command()

    for action in ("install", "uninstall"):
        target = "tool_web_search" if action == "install" else "tool_old"
        assert "confirm" not in _names(
            command, {"action": action, "stem": target})


def test_the_real_parser_routes_every_spelling_to_the_right_slot():
    """Driven through ``parse_command_line`` and ``_missing`` rather than the
    form alone, because the step *flags* are what decide this and reading them
    is not the same as running them. The first attempt made ``category``
    required and assumed enum membership went unchecked until validation;
    ``FormStep.coerce`` enforces it during the peel, so ``/packages install
    tool_web_search`` died on "category must be one of".
    """
    from plugins.command_registry import parse_command_line
    from state_machine.action import _missing
    from state_machine.conversation import CallableSpec, FormStep

    command = _command()

    def factory(args, cs=None):
        return [FormStep(**step) for step in command.form(_SDK(), args)]

    spec = CallableSpec(name="packages", handler=None, form=[],
                        form_factory=factory)

    def parse(line):
        args = parse_command_line(line, factory)
        return args, [step.name for step in _missing(spec, args)]

    # Named outright: runs, no card, nothing asked.
    assert parse("install tool_web_search") == (
        {"action": "install", "stem": "tool_web_search"}, [])
    assert parse("uninstall tool_old") == (
        {"action": "uninstall", "stem": "tool_old"}, [])
    # Category then package: the card is all that is left.
    args, missing = parse("install tools tool_web_search")
    assert args["category"] == "tools"
    assert args["package_id"] == "tool_web_search" and missing == ["confirm"]
    # Half-typed: the picker opens where the typing stopped.
    assert parse("install")[1] == ["category"]
    assert parse("install tools")[1] == ["package_id"]
    assert parse("")[1] == ["action"]
    assert parse("update") == ({"action": "update"}, [])


def test_run_takes_the_stem_from_whichever_slot_holds_it():
    command = _command()
    asked = []

    class _Install(_Plugins):
        def install(self, target):
            asked.append(("install", target))
            return "ok"

        def uninstall(self, target):
            asked.append(("uninstall", target))
            return "ok"

    class _S(_SDK):
        plugins = _Install()

    command.run(_S(), {"action": "install", "category": "tools",
                       "package_id": "tool_web_search", "confirm": "yes"})
    command.run(_S(), {"action": "install", "stem": "tool_web_search"})
    command.run(_S(), {"action": "uninstall", "stem": "tool_old"})

    assert asked == [("install", "tool_web_search"),
                     ("install", "tool_web_search"),
                     ("uninstall", "tool_old")]


# ──────────────────────────────────────────────────────────────────────
# The card.
# ──────────────────────────────────────────────────────────────────────

def test_the_confirm_step_is_one_button_carrying_the_card():
    """Cancelling and going back are what every form already offers, so the
    step exists to *show* rather than to ask a second question."""
    command = _command()

    step = command.form(_SDK(), {"action": "install", "category": "tools",
                                 "package_id": "tool_web_search"})[-1]

    assert step["enum"] == ["yes"] and step["enum_labels"] == ["Install"]
    assert "Search the web." in step["prompt"]
    assert "service_web_search" in step["prompt"]  # what else it installs
    assert "httpx" in step["prompt"]


def test_the_uninstall_card_says_what_else_comes_out():
    """The backwards dependency closure is the fact that matters here, and it
    is the one nothing showed before committing."""
    command = _command()

    step = command.form(_SDK(), {"action": "uninstall", "category": "tools",
                                 "package_id": "tool_old"})[-1]

    assert "tool_dep" in step["prompt"]
    assert step["enum_labels"] == ["Uninstall"]


def test_a_package_whose_metadata_will_not_read_is_still_installable():
    """The card describes it thinly rather than the command refusing."""
    command = _command()

    class _Broken(_Plugins):
        def list(self, source="registered", **kwargs):
            if source == "info":
                raise _Failed("cannot parse")
            return super().list(source=source, **kwargs)

    class _S(_SDK):
        plugins = _Broken()

    step = command.form(_S(), {"action": "install", "category": "tools",
                               "package_id": "tool_web_search"})[-1]

    assert "tool_web_search" in step["prompt"]


# ──────────────────────────────────────────────────────────────────────
# Where the description comes from.
# ──────────────────────────────────────────────────────────────────────

def test_the_store_reader_picks_up_name_and_description():
    """Same AST pass the dependencies come from; nothing is imported."""
    meta = read_dependency_meta("tools/tool_x.py", (
        'class T:\n'
        '    name = "x"\n'
        '    description = "Does a thing."\n'
        '    dependencies_pip = ["httpx"]\n'))

    assert meta.name == "x"
    assert meta.description == "Does a thing."
    assert meta.dependencies_pip == ("httpx",)


def test_an_unreadable_description_is_absent_rather_than_fatal():
    """Deliberately lenient where the dependency fields raise: a dependency
    the manager cannot read is a package it would install wrongly, but a
    description it cannot read costs one line of a card."""
    meta = read_dependency_meta("tools/tool_y.py", (
        'class T:\n'
        '    name = SOMETHING\n'
        '    description = 3\n'))

    assert meta.name == "" and meta.description == ""
