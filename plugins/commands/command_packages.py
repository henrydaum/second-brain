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
CATEGORIES = [
    "tools", "tasks", "services", "commands", "frontends", "bundles"]
CATEGORY_LABELS = [
    "Tools", "Tasks", "Services", "Commands", "Frontends", "Bundles"]
_BLURB = {
    "tools": "agent-callable tools",
    "tasks": "pipeline tasks",
    "services": "persistent backends and helpers",
    "commands": "slash commands",
    "frontends": "chat frontends and helpers",
    "bundles": "named groups of store files",
}


class PackagesCommand(BaseCommand):
    """Browse and manage packages through the kernel-owned store."""

    name = "packages"
    description = "Browse, install, or uninstall store files by category"
    category = "System"
    agent_prompt = (
        "Installing or uninstalling a package changes the live catalogs: new "
        "tools and commands appear on the next turn, not instantly. After an "
        "install, re-check the tool catalog before concluding a capability is "
        "missing or broken."
    )
    requests = [
        "plugin.list", "plugin.install", "plugin.uninstall", "plugin.update"]

    def form(self, sdk, args):
        """Build dependent steps from the answers collected so far."""
        steps = [FormStep(
            "action", "Choose a package action.", True,
            enum=ACTIONS, enum_labels=ACTION_LABELS)]
        action = args.get("action")
        if action in {"available", "installed"}:
            steps.append(FormStep(
                "category", _category_prompt(sdk, action), True,
                enum=CATEGORIES, enum_labels=CATEGORY_LABELS, columns=2))
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


def _overview(sdk, action):
    counts = _counts(sdk, action)
    header = (
        "Installed files by category:"
        if action == "installed"
        else "Available files by category:"
    )
    rows = [
        (label, counts.get(category, 0), _BLURB[category])
        for category, label in zip(CATEGORIES, CATEGORY_LABELS)
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
    if category == "bundles":
        return f"{prefix} bundles:"
    label = _label(category).lower()
    return (
        f"{prefix} "
        f"{label[:-1] if label.endswith('s') else label} plugins:"
    )


def _label(category):
    if category in CATEGORIES:
        return CATEGORY_LABELS[CATEGORIES.index(category)]
    return category or ""
