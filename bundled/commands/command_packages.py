"""Slash command plugin for `/packages`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


ACTIONS = ["install", "uninstall", "update"]
ACTION_LABELS = ["Install", "Uninstall", "Update installed"]
# Which listing each action browses, and what its confirm button says. The
# two flows differ only in these, so they share one form builder: browsing
# and acting used to be four commands because the browse listing had nowhere
# to go except into a message the user then had to read a name out of. It is
# a step prompt now, so the pick and the act are one pass.
_SOURCE = {"install": "available", "uninstall": "removable"}
_COMMIT = {"install": "Install", "uninstall": "Uninstall"}
# Labels and one-liners for the families the kernel reports. Deliberately a
# *lookup* rather than a list: the categories themselves come from
# ``plugins.list(source="families")``, which derives them from ``trees.ROOTS``.
# This was two parallel lists zipped together, so a family the kernel knew
# about but this file had not heard of was dropped without a word — which is
# what hid `llm` and `parsers` from a menu built out of their own counts.
_LABELS = {
    "tools": "Tools",
    "tasks": "Tasks",
    "services": "Services",
    "commands": "Commands",
    "frontends": "Frontends",
    "parsers": "Parsers",
    "llm": "LLM backends",
    "scripts": "Scripts",
    "bundles": "Bundles",
}
_BLURB = {
    "tools": "agent-callable tools",
    "tasks": "pipeline tasks",
    "services": "persistent backends and helpers",
    "commands": "slash commands",
    "frontends": "chat frontends and helpers",
    "parsers": "file-type readers",
    "llm": "model providers",
    "scripts": "runnable SDK snippets",
    "bundles": "named groups of store files",
}


class PackagesCommand(BaseCommand):
    """Browse and manage packages through the kernel-owned store."""

    name = "packages"
    description = "Install, uninstall, or update store files by category"
    category = "Capabilities"
    # Every action here changes what this system can do. The
    # declaration is what keeps them on the *up-front* approval path, where
    # the state machine asks before the body runs and the answer becomes a
    # grant covering the Requests below. Without it the command ran ungranted
    # and hit the execution-time approver mid-run, which is the path a command
    # cannot be asked from.
    approval_actions = ("install", "uninstall", "update")
    approval_actor_id = "user"
    requests = [
        "plugin.list", "plugin.install", "plugin.uninstall", "plugin.update"]

    def form(self, sdk, args):
        """Build dependent steps from the answers collected so far.

        Re-called after every answer, so each step's options are built from
        the ones before it: a category, then that category's packages, then
        a card describing the one picked. Nobody types a stem.
        """
        steps = [FormStep(
            "action", "Choose a package action.", True,
            enum=ACTIONS, enum_labels=ACTION_LABELS)]
        source = _SOURCE.get(args.get("action"))
        if not source:
            return steps

        # First the slot nobody is ever *shown*: the one-argument spelling,
        # ``/packages install tool_hybrid_search``, which is what the agent is
        # told to type and what every older invocation says. It is neither
        # required nor `prompt_when_missing`, so no form ever asks for it —
        # it exists to catch a stem off the command line, where the parser
        # peels arguments in step order. Optional is what buys the lookahead
        # (``plugins/command_registry.py``): a token matching no stem is left
        # here and falls through to the category below, because a required
        # step still follows. A required step would instead have taken the
        # token and had ``FormStep.coerce`` refuse it against the enum.
        #
        # It has to be a separate key from the picker's, because a step the
        # parser skips is *filled with its default* — so one shared key would
        # arrive already answered and the picker would never open.
        catalogue = sdk.plugins.list(source=source)
        steps.append(FormStep(
            "stem", "", False,
            enum=[item["id"] for item in catalogue]))
        if args.get("stem"):
            # Named outright: nothing left to ask, and deliberately no card.
            # An agent that meets a missing required step gets a failure
            # rather than a form, so a confirm step on this path would make
            # the command uncallable by the agent its own prompt instructs.
            return steps

        categories = _stocked(sdk, source)
        steps.append(FormStep(
            "category", _category_prompt(sdk, source), True,
            enum=categories,
            enum_labels=[_label(item) for item in categories], columns=2))
        category = args.get("category")
        if not category:
            return steps

        items = sdk.plugins.list(source=source, category=category)
        steps.append(FormStep(
            "package_id", _package_prompt(sdk, items, category), True,
            enum=[item["id"] for item in items],
            enum_labels=[_item_label(item) for item in items], columns=2))
        package_id = args.get("package_id")
        if not package_id:
            return steps

        # One button, because cancelling and going back are what every form
        # already offers. This step exists to *show* the card, not to ask a
        # second question — the approval dialog that follows states the
        # authority, and this states the thing.
        commit = _COMMIT[args["action"]]
        steps.append(FormStep(
            "confirm", _confirm_prompt(sdk, args["action"], package_id), True,
            enum=["yes"], enum_labels=[commit]))
        return steps

    def run(self, sdk, args):
        """Execute the selected package action."""
        action = args.get("action") or "install"
        # The picker's answer, or the stem somebody named outright.
        target = args.get("package_id") or args.get("stem") or ""
        try:
            if action == "install":
                return sdk.plugins.install(target)
            if action == "uninstall":
                return sdk.plugins.uninstall(target)
            if action == "update":
                return sdk.plugins.update()
            return f"Unknown action: {action}"
        except sdk.Failed as exc:
            return f"Package {action} failed: {exc.error}"


def _category_prompt(sdk, source):
    return _overview(sdk, source) + "\n\nChoose a category."


def _package_prompt(sdk, items, category):
    if not items:
        return f"Nothing in {_label(category).lower()}."
    return _items_table(sdk, items) + "\n\nChoose a package."


def _confirm_prompt(sdk, action, package_id):
    """The card somebody reads before committing to a package."""
    info = _info(sdk, action, package_id)
    pairs = [
        ("Name", info.get("name") or package_id),
        ("Category", _label(info.get("family") or "")),
        ("Path", info.get("path") or "—"),
    ]
    if action == "install":
        _add(pairs, "Also installs", _stems(info.get("dependencies_files")))
        _add(pairs, "Python packages", info.get("dependencies_pip"))
    else:
        _add(pairs, "Also removes", _stems(info.get("also_removes")))
        _add(pairs, "Python packages", info.get("removes_pip"))
    card = sdk.md.card(f"{_COMMIT[action]} {package_id}", pairs)
    # Prose under the card rather than in a cell: a tool's description is a
    # paragraph, and a table cell cannot hold one legibly.
    description = info.get("description") or ""
    if description:
        card += "\n\n" + sdk.md.quote(description)
    return card + f"\n\n{_COMMIT[action]}?"


def _add(pairs, label, values):
    if values:
        pairs.append((label, ", ".join(values)))


def _stems(paths):
    return [path.rsplit("/", 1)[-1].removesuffix(".py")
            for path in (paths or [])]


def _info(sdk, action, package_id):
    """One package's metadata, or just enough of it to draw a card.

    Read failures are absorbed on purpose: a store file whose description
    cannot be parsed is one the card describes thinly, never one the user
    cannot install.
    """
    source = "info" if action == "install" else "installed_info"
    try:
        return sdk.plugins.list(source=source, name=package_id) or {}
    except sdk.Failed:
        return {}


def _categories(sdk):
    """Every family the store can hold, straight from the layout."""
    try:
        return sdk.plugins.list(source="families") or []
    except sdk.Failed:
        return sorted(_LABELS)


def _stocked(sdk, source):
    """Categories with something in them — a button for an empty family is a
    dead end the user has to back out of."""
    counts = _counts(sdk, source)
    stocked = [item for item in _categories(sdk) if counts.get(item)]
    return stocked or _categories(sdk)


def _overview(sdk, source):
    counts = _counts(sdk, source)
    header = (
        "Installed files by category:"
        if source in ("installed", "removable")
        else "Available files by category:"
    )
    rows = [
        (_label(category), counts.get(category, 0), _BLURB.get(category, ""))
        for category in _categories(sdk)
        if counts.get(category)
    ]
    return header + "\n\n" + sdk.md.table(
        ["Category", "Count", "What"], rows)


def _counts(sdk, source):
    try:
        items = sdk.plugins.list(source=source)
    except sdk.Failed:
        return {}
    counts = {}
    for item in items:
        family = item["family"]
        counts[family] = counts.get(family, 0) + 1
    return counts


def _items_table(sdk, items):
    rows = [(_item_label(item), item["path"]) for item in items]
    return sdk.md.table(["Name", "Path"], rows)


def _item_label(item):
    return item["id"] + (" (helper)" if item.get("helper") else "")


def _label(category):
    return _LABELS.get(category) or (category or "").replace("_", " ").title()
