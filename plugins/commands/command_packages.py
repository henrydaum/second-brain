"""Slash command plugin for `/packages`."""

from collections import Counter

from plugins.BaseCommand import BaseCommand
from plugins.commands.helpers import package_manager
from plugins.frontends.helpers.formatters import md_table
from plugins.commands.helpers.store_backend import StoreBackendError
from state_machine.conversation import FormStep


ACTIONS = ["available", "installed", "install", "uninstall", "update"]
ACTION_LABELS = ["Browse available", "Browse installed", "Install", "Uninstall", "Update installed"]
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
    agent_prompt = (
        "Installing or uninstalling a package changes the live catalogs: new "
        "tools and commands appear on the next turn, not instantly. After an "
        "install, re-check the tool catalog before concluding a capability is "
        "missing or broken."
    )

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
        progress = _progress_for(context)
        try:
            if action == "available":
                return _format_available(context, args.get("category"))
            if action == "installed":
                return _format_installed(args.get("category"))
            if action == "install":
                return package_manager.install_package(context.root_dir, args.get("package_id", ""), context, progress=progress).text()
            if action == "uninstall":
                return package_manager.uninstall_package(args.get("package_id", ""), context, progress=progress, root_dir=context.root_dir).text()
            if action == "update":
                return package_manager.update_packages(context.root_dir, context, progress=progress).text()
            return f"Unknown action: {action}"
        except (package_manager.PackageError, StoreBackendError) as e:
            return f"Package {action} failed: {e}"


def _progress_for(context):
    """Progress sink that reaches the user's frontend, not just stdout.

    Long steps (pip install can run minutes) need to surface in whatever
    frontend issued the command; CHAT_MESSAGE_PUSHED via push_message does
    that. Headless/test contexts fall back to printing.
    """
    runtime = getattr(context, "runtime", None)
    session_key = getattr(context, "session_key", None)
    if runtime is not None and session_key:
        return lambda message: runtime.push_message(session_key, message, source="packages")
    return lambda message: print(message, flush=True)


def _category_prompt(context, action: str) -> str:
    return _overview(context, action) + "\n\nChoose a category."


def _overview(context, action: str) -> str:
    counts = _counts(context, action)
    header = "Installed files by category:" if action == "installed" else "Available files by category:"
    rows = [(label, counts.get(cat, 0), _BLURB[cat]) for cat, label in zip(CATEGORIES, CATEGORY_LABELS)]
    return header + "\n\n" + md_table(["Category", "Count", "What"], rows)


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
    return "\n\n".join([_heading("Available", category),
                        _items_table(items), "Install with `/packages install <name>`."])


def _format_installed(category: str | None) -> str:
    if not category:
        return _overview(None, "installed") + "\n\nChoose a category with /packages installed <category>."
    items = [item for item in package_manager.installed_packages() if item["family"] == category]
    if not items:
        return f"No {_label(category).lower()} files installed."
    return "\n\n".join([_heading("Installed", category),
                        _items_table(items), "Uninstall with `/packages uninstall <name>`."])


def _items_table(items: list[dict]) -> str:
    rows = [(item["id"] + (" (helper)" if item.get("helper") else ""), item["path"]) for item in items]
    return md_table(["Name", "Path"], rows)


def _heading(prefix: str, category: str) -> str:
    if category == "bundles":
        return f"{prefix} bundles:"
    label = _label(category).lower()
    return f"{prefix} {label[:-1] if label.endswith('s') else label} plugins:"


def _label(category: str) -> str:
    return CATEGORY_LABELS[CATEGORIES.index(category)] if category in CATEGORIES else (category or "")
