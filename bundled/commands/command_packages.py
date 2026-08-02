"""Slash command plugin for `/packages`."""

from guest.bases import BaseCommand
from guest.forms import FormStep


ACTIONS = ["available", "installed", "install", "uninstall", "update"]
ACTION_LABELS = [
    "Browse available",
    "Browse installed",
    "Install",
    "Uninstall",
    "Update installed",
]
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
# What a family's members are called in a heading. Only the exceptions: a
# parser is not a plugin and an LLM backend is not one either, so calling them
# "parser plugins" is wrong in the one place the wording is load-bearing.
_NOUN = {
    "bundles": "bundles",
    "parsers": "parsers",
    "llm": "LLM backends",
    "scripts": "scripts",
}


class PackagesCommand(BaseCommand):
    """Browse and manage packages through the kernel-owned store."""

    name = "packages"
    description = "Browse, install, or uninstall store files by category"
    category = "Capabilities"
    agent_prompt = (
        "## Package changes\n"
        "Installing or uninstalling a package changes the live catalogs: new "
        "tools and commands appear on the next turn, not instantly. After an "
        "install, re-check the tool catalog before concluding a capability is "
        "missing or broken."
    )
    # Browsing is a read; the other three change what this system can do. The
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
        """Build dependent steps from the answers collected so far."""
        steps = [FormStep(
            "action", "Choose a package action.", True,
            enum=ACTIONS, enum_labels=ACTION_LABELS)]
        action = args.get("action")
        if action in {"available", "installed"}:
            categories = _categories(sdk)
            steps.append(FormStep(
                "category", _category_prompt(sdk, action), True,
                enum=categories,
                enum_labels=[_label(item) for item in categories], columns=2))
        elif action == "install":
            steps.append(FormStep(
                "package_id",
                "Enter the plugin, helper, or bundle stem to install.",
                True,
            ))
        elif action == "uninstall":
            items = sdk.plugins.list(source="removable")
            steps.append(FormStep(
                "package_id",
                "Choose the plugin, helper, or bundle stem to uninstall.",
                True,
                enum=[item["id"] for item in items],
                columns=2,
            ))
        return steps

    def run(self, sdk, args):
        """Execute the selected package action."""
        action = args.get("action") or "installed"
        try:
            if action == "available":
                return _format_available(sdk, args.get("category"))
            if action == "installed":
                return _format_installed(sdk, args.get("category"))
            if action == "install":
                return sdk.plugins.install(args.get("package_id", ""))
            if action == "uninstall":
                return sdk.plugins.uninstall(args.get("package_id", ""))
            if action == "update":
                return sdk.plugins.update()
            return f"Unknown action: {action}"
        except sdk.Failed as exc:
            return f"Package {action} failed: {exc.error}"


def _category_prompt(sdk, action):
    return _overview(sdk, action) + "\n\nChoose a category."


def _categories(sdk):
    """Every family the store can hold, straight from the layout."""
    try:
        return sdk.plugins.list(source="families") or []
    except sdk.Failed:
        return sorted(_LABELS)


def _overview(sdk, action):
    counts = _counts(sdk, action)
    header = (
        "Installed files by category:"
        if action == "installed"
        else "Available files by category:"
    )
    rows = [
        (_label(category), counts.get(category, 0), _BLURB.get(category, ""))
        for category in _categories(sdk)
    ]
    return header + "\n\n" + _md_table(
        ["Category", "Count", "What"], rows)


def _counts(sdk, action):
    items = sdk.plugins.list(source=action)
    counts = {}
    for item in items:
        family = item["family"]
        counts[family] = counts.get(family, 0) + 1
    return counts


def _format_available(sdk, category):
    if not category:
        return (
            _overview(sdk, "available")
            + "\n\nChoose a category with /packages available <category>."
        )
    items = sdk.plugins.list(source="available", category=category)
    if not items:
        return f"No available {_label(category).lower()} files."
    return "\n\n".join([
        _heading("Available", category),
        _items_table(items),
        "Install with `/packages install <name>`.",
    ])


def _format_installed(sdk, category):
    if not category:
        return (
            _overview(sdk, "installed")
            + "\n\nChoose a category with /packages installed <category>."
        )
    items = sdk.plugins.list(source="installed", category=category)
    if not items:
        return f"No {_label(category).lower()} files installed."
    return "\n\n".join([
        _heading("Installed", category),
        _items_table(items),
        "Uninstall with `/packages uninstall <name>`.",
    ])


def _items_table(items):
    rows = [
        (
            item["id"] + (" (helper)" if item.get("helper") else ""),
            item["path"],
        )
        for item in items
    ]
    return _md_table(["Name", "Path"], rows)


def _md_table(headers, rows):
    """Pure copy of the kernel's markdown table wire format."""
    def cell(value):
        return str("" if value is None else value).replace(
            "\n", " ").replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _heading(prefix, category):
    noun = _NOUN.get(category)
    if noun:
        return f"{prefix} {noun}:"
    label = _label(category).lower()
    return (
        f"{prefix} "
        f"{label[:-1] if label.endswith('s') else label} plugins:"
    )


def _label(category):
    return _LABELS.get(category) or (category or "").replace("_", " ").title()
