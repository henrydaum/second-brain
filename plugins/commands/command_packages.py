"""Slash command plugin for `/packages`."""

from collections import Counter

from plugins.BaseCommand import BaseCommand
from plugins.commands.helpers import package_manager
from plugins.commands.helpers.store_backend import StoreBackendError
from state_machine.conversation import FormStep


ACTIONS = ["available", "installed", "install", "uninstall"]
ACTION_LABELS = ["Browse available", "Browse installed", "Install", "Uninstall"]
CATEGORIES = ["tools", "tasks", "services", "commands", "frontends", "bundles"]
CATEGORY_LABELS = ["Tools", "Tasks", "Services", "Commands", "Frontends", "Bundles"]
_BLURB = {
    "tools": "agent-callable tools",
    "tasks": "pipeline tasks",
    "services": "persistent backends and helpers",
    "commands": "slash commands",
    "frontends": "chat frontends and helpers",
    "bundles": "named groups of store files",
}


class PackagesCommand(BaseCommand):
    """Browse, install, and uninstall tree-store plugins/helpers."""
    name = "packages"
    description = "Browse, install, or uninstall store files by category"
    category = "System"

    def form(self, args, context):
        steps = [FormStep("action", "Choose a package action.", True, enum=ACTIONS, enum_labels=ACTION_LABELS)]
        action = args.get("action")
        if action in {"available", "installed"}:
            steps.append(FormStep("category", _category_prompt(context, action), True, enum=CATEGORIES, enum_labels=CATEGORY_LABELS, columns=2))
        elif action == "install":
            steps.append(FormStep("package_id", "Enter the plugin, helper, or bundle stem to install.", True))
        elif action == "uninstall":
            items = package_manager.removable_packages() + package_manager.search_bundles(context.root_dir)
            steps.append(FormStep("package_id", "Choose the plugin, helper, or bundle stem to uninstall.", True, enum=[p["id"] for p in items], columns=2))
        return steps

    def run(self, args, context):
        action = args.get("action") or "installed"
        try:
            if action == "available":
                return _format_available(context, args.get("category"))
            if action == "installed":
                return _format_installed(args.get("category"))
            if action == "install":
                return package_manager.install_package(context.root_dir, args.get("package_id", ""), context, progress=_progress).text()
            if action == "uninstall":
                return package_manager.uninstall_package(args.get("package_id", ""), context, progress=_progress, root_dir=context.root_dir).text()
            return f"Unknown action: {action}"
        except (package_manager.PackageError, StoreBackendError) as e:
            return f"Package {action} failed: {e}"


def _progress(message: str) -> None:
    print(message, flush=True)


def _category_prompt(context, action: str) -> str:
    return _overview(context, action) + "\n\nChoose a category."


def _overview(context, action: str) -> str:
    counts = _counts(context, action)
    header = "Installed files by category:" if action == "installed" else "Available files by category:"
    name_w = max(len(label) for label in CATEGORY_LABELS)
    count_w = max(len(str(n)) for n in counts.values()) if counts else 1
    lines = [header]
    for cat, label in zip(CATEGORIES, CATEGORY_LABELS):
        lines.append(f"  {label:<{name_w}}  {counts.get(cat, 0):>{count_w}}   - {_BLURB[cat]}")
    return "\n".join(lines)


def _counts(context, action: str) -> Counter:
    items = package_manager.installed_packages() if action == "installed" else _available_items(context)
    return Counter(item["family"] for item in items)


def _available_items(context) -> list[dict]:
    installed_paths = {item["path"] for item in package_manager.installed_packages()}
    return [item for item in package_manager.search_packages(context.root_dir) if item["path"] not in installed_paths]


def _format_available(context, category: str | None) -> str:
    if not category:
        return _overview(context, "available") + "\n\nChoose a category with /packages available <category>."
    items = [item for item in _available_items(context) if item["family"] == category]
    if not items:
        return f"No available {_label(category).lower()} files."
    lines = [f"Available {_label(category).lower()} files:"]
    lines.extend(_line(item) for item in items)
    lines += ["", "Install with /packages install <stem>."]
    return "\n".join(lines)


def _format_installed(category: str | None) -> str:
    if not category:
        return _overview(None, "installed") + "\n\nChoose a category with /packages installed <category>."
    items = [item for item in package_manager.installed_packages() if item["family"] == category]
    if not items:
        return f"No {_label(category).lower()} files installed."
    lines = [f"Installed {_label(category).lower()} files:"]
    lines.extend(_line(item) for item in items)
    lines += ["", "Uninstall with /packages uninstall <stem>."]
    return "\n".join(lines)


def _line(item: dict) -> str:
    helper = " [helper]" if item.get("helper") else ""
    return f"  *{item['id']}*{helper} - {item['path']}"


def _label(category: str) -> str:
    return CATEGORY_LABELS[CATEGORIES.index(category)] if category in CATEGORIES else (category or "")
