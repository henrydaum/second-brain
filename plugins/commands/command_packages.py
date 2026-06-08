"""Slash command plugin for `/packages`."""

import re
from collections import Counter

from plugins.BaseCommand import BaseCommand
from plugins.commands.helpers import package_manager
from plugins.commands.helpers.store_backend import StoreBackendError
from plugins.helpers.plugin_paths import PLUGIN_FAMILIES
from state_machine.conversation import FormStep


ACTIONS = ["available", "installed", "install", "uninstall"]
ACTION_LABELS = ["Browse available", "Browse installed", "Install package", "Uninstall package"]
CLEANUP_MODES = ["all", "none", "specific"]
CLEANUP_MODE_LABELS = ["All", "None", "Specific"]

# The seven structural families a package can belong to (5 plugin families +
# meta-bundles + shared helpers), keyed the same way the store index and the
# publisher's ``family_for`` are. This is the browse taxonomy.
CATEGORIES = ["tools", "tasks", "services", "commands", "frontends", "bundles", "helpers"]
CATEGORY_LABELS = ["Tools", "Tasks", "Services", "Commands", "Frontends", "Bundles", "Helpers"]

_CATEGORY_BLURB = {
    "tools": "agent-callable tools",
    "tasks": "pipeline tasks",
    "services": "persistent backends",
    "commands": "slash commands",
    "frontends": "chat frontends",
    "bundles": "curated plugin sets",
    "helpers": "shared modules",
}


class PackagesCommand(BaseCommand):
    """Install and uninstall package-store plugins."""
    name = "packages"
    description = "Browse, install, or uninstall packages by category"
    category = "System"

    def form(self, args, context):
        steps = [FormStep("action", "Choose a package action.", True, enum=ACTIONS, enum_labels=ACTION_LABELS)]
        action = args.get("action")
        if action in {"available", "installed"}:
            steps.append(FormStep("category", _category_prompt(context, action), True, enum=CATEGORIES, enum_labels=CATEGORY_LABELS, columns=2))
        if action == "install":
            steps.append(FormStep("package_id", "Enter the package id.", True))
        if action == "uninstall":
            steps.append(FormStep("package_id", "Choose the package to uninstall.", True, enum=[p["id"] for p in package_manager.removable_packages()], columns=2))
            if args.get("package_id"):
                plan = package_manager.build_uninstall_plan(args["package_id"])
                if plan.needs_confirm:
                    steps.append(FormStep("confirm", _confirm_prompt(plan), True, "boolean", default=False))
                    if not _truthy(args.get("confirm")):
                        return steps  # settle the confirm before asking about cleanup
                if any(pkg.cleanup["settings"] for pkg in plan.packages):
                    steps.append(FormStep("cleanup_config", "Remove package-owned config settings?", True, enum=CLEANUP_MODES, enum_labels=CLEANUP_MODE_LABELS, default="all"))
                if any(pkg.cleanup["tables"] for pkg in plan.packages):
                    steps.append(FormStep("cleanup_tables", "Drop package-owned SQL tables?", True, enum=CLEANUP_MODES, enum_labels=CLEANUP_MODE_LABELS, default="all"))
                if any(plan.pip_removals.values()):
                    steps.append(FormStep("cleanup_pip", "Uninstall safe package-owned Python deps?", True, enum=CLEANUP_MODES, enum_labels=CLEANUP_MODE_LABELS, default="all"))
                if _cleanup_mode(args.get("cleanup_config")) == "specific":
                    for pkg in plan.packages:
                        if pkg.cleanup["settings"]:
                            steps.append(FormStep(_cleanup_arg("config", pkg.id), _cleanup_prompt(pkg.id, "Remove config settings", "Config settings", pkg.cleanup["settings"]), True, "boolean"))
                if _cleanup_mode(args.get("cleanup_tables")) == "specific":
                    for pkg in plan.packages:
                        if pkg.cleanup["tables"]:
                            steps.append(FormStep(_cleanup_arg("tables", pkg.id), _cleanup_prompt(pkg.id, "Drop SQL tables", "Tables", pkg.cleanup["tables"]), True, "boolean"))
                if _cleanup_mode(args.get("cleanup_pip")) == "specific":
                    for pkg in plan.packages:
                        if plan.pip_removals.get(pkg.id):
                            steps.append(FormStep(_cleanup_arg("pip", pkg.id), package_manager.cleanup_pip_prompt(plan.pip_removals[pkg.id]), True, "boolean"))
        return steps

    def run(self, args, context):
        action = args.get("action") or "installed"
        try:
            if action == "available":
                return _format_available(context, args.get("category"))
            if action == "installed":
                return _format_installed(context, args.get("category"))
            if action == "install":
                return package_manager.install_package(context.root_dir, args.get("package_id", ""), context, progress=_progress).text()
            if action == "uninstall":
                plan = package_manager.build_uninstall_plan(args.get("package_id", ""))
                if plan.needs_confirm and not _truthy(args.get("confirm")):
                    return "Uninstall cancelled."
                return package_manager.execute_uninstall_plan(plan, context, _cleanup_choices(plan, args), progress=_progress).text()
            return f"Unknown action: {action}"
        except (package_manager.PackageError, StoreBackendError) as e:
            if action == "install" and _is_missing_manifest(e, args.get("package_id", "")):
                return f"Package {action} failed: {args.get('package_id', '')!r} not found."
            return f"Package {action} failed: {e}"


def _progress(message: str) -> None:
    print(message, flush=True)


# ── Category browsing ────────────────────────────────────────────────

def _item_family(item: dict) -> str:
    """The category a store-index entry belongs to (carried in the index)."""
    family = item.get("family")
    return family if family in CATEGORIES else "helpers"


def _receipt_family(receipt: dict) -> str:
    """An installed package's browse category, read from its id prefix.

    ``bundle_*`` ⇒ bundles, a plugin family prefix (``tool_``/``task_``/…) ⇒
    that family, anything else ⇒ a shared-helper package.
    """
    pid = receipt.get("id", "")
    if pid.startswith("bundle_"):
        return "bundles"
    for plugin_type, (family, prefix) in PLUGIN_FAMILIES.items():
        if pid.startswith(prefix):
            return family
    return "helpers"


def _category_counts(context, action: str) -> Counter:
    if action == "installed":
        return Counter(_receipt_family(r) for r in package_manager.installed_packages())
    installed = {p.get("id") for p in package_manager.installed_packages()}
    items = package_manager.search_packages(context.root_dir)
    return Counter(_item_family(i) for i in items if i.get("id") not in installed)


def _category_overview(context, action: str) -> str:
    counts = _category_counts(context, action)
    header = "Installed packages by category:" if action == "installed" else "Available packages by category:"
    # Column widths derived from the data so the table stays aligned if labels
    # or counts change: labels left-aligned, counts right-aligned in their own
    # column (``{value:<w}`` / ``{value:>w}`` are Python's alignment specs).
    name_w = max(len(label) for label in CATEGORY_LABELS)
    count_w = max(len(str(n)) for n in counts.values()) if counts else 1
    lines = [header]
    for cat, label in zip(CATEGORIES, CATEGORY_LABELS):
        blurb = _CATEGORY_BLURB.get(cat, "")
        lines.append(f"  {label:<{name_w}}  {counts.get(cat, 0):>{count_w}}{('   — ' + blurb) if blurb else ''}")
    return "\n".join(lines)


def _category_prompt(context, action: str) -> str:
    return _category_overview(context, action) + "\n\nChoose a category."


def _category_label(category: str) -> str:
    return CATEGORY_LABELS[CATEGORIES.index(category)] if category in CATEGORIES else (category or "")


def _format_available(context, category: str | None) -> str:
    if not category:
        return _category_overview(context, "available") + "\n\nChoose a category with /packages available <category>."
    installed = {p.get("id") for p in package_manager.installed_packages()}
    items = [i for i in package_manager.search_packages(context.root_dir)
             if _item_family(i) == category and i.get("id") not in installed]
    label = _category_label(category).lower()
    if not items:
        return f"No available {label} packages — everything published is already installed."
    lines = [f"Available {label} packages:"]
    lines.extend(_available_line(i) for i in items)
    lines.append("")
    lines.append("Install with /packages install <id>.")
    return "\n".join(lines)


def _format_installed(context, category: str | None) -> str:
    if not category:
        return _category_overview(context, "installed") + "\n\nChoose a category with /packages installed <category>."
    receipts = [r for r in package_manager.installed_packages() if _receipt_family(r) == category]
    label = _category_label(category).lower()
    if not receipts:
        return f"No {label} packages installed."
    lines = [f"Installed {label} packages:"]
    for item in receipts:
        deps = item.get("requires") or []
        suffix = f" (requires: {', '.join(deps)})" if deps else ""
        lines.append(f"  *{item.get('id')}*{suffix}")
    lines.append("")
    lines.append("Uninstall with /packages uninstall <id>.")
    return "\n".join(lines)


def _available_line(item: dict) -> str:
    desc = item.get("description") or ""
    return f"  *{item.get('id', '')}*{(' — ' + desc) if desc else ''}"


# ── Uninstall cleanup form plumbing ──────────────────────────────────

def _confirm_prompt(plan) -> str:
    """Warn before a removal that would break something still installed."""
    dependents = ", ".join(plan.broken_dependents)
    if plan.raw_files:
        return f"{', '.join(plan.raw_files)} is still listed by package(s): {dependents}. Delete the file anyway?"
    return f"{plan.target} is still required by: {dependents}. Uninstall anyway — those plugins may break?"


def _cleanup_arg(kind: str, package_id: str) -> str:
    return "cleanup_" + kind + "__" + re.sub(r"[^a-zA-Z0-9_]", "_", package_id)


def _cleanup_prompt(package_id: str, action: str, label: str, values: list[str]) -> str:
    return f"{action} for {package_id}?\n\n{label}: " + ", ".join(values)


def _cleanup_choices(plan, args) -> dict[str, dict[str, bool]]:
    return {
        "config": _cleanup_kind_choices(plan, args, "config", lambda pkg: bool(pkg.cleanup["settings"])),
        "tables": _cleanup_kind_choices(plan, args, "tables", lambda pkg: bool(pkg.cleanup["tables"])),
        "pip": _cleanup_kind_choices(plan, args, "pip", lambda pkg: bool(plan.pip_removals.get(pkg.id))),
    }


def _cleanup_kind_choices(plan, args, kind: str, relevant) -> dict[str, bool]:
    mode = _cleanup_mode(args.get(f"cleanup_{kind}"))
    if mode == "all":
        return {pkg.id: bool(relevant(pkg)) for pkg in plan.packages}
    if mode == "none":
        return {pkg.id: False for pkg in plan.packages}
    return {pkg.id: _truthy(args.get(_cleanup_arg(kind, pkg.id))) for pkg in plan.packages}


def _cleanup_mode(value) -> str:
    if value in (None, ""):
        return "all"
    if isinstance(value, bool):
        return "all" if value else "none"
    text = str(value).strip().lower()
    if text in CLEANUP_MODES:
        return text
    return "all" if _truthy(text) else "none"


def _truthy(value) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in {"true", "yes", "1", "y"}


def _is_missing_manifest(error: Exception, package_id: str) -> bool:
    return bool(package_id) and f"packages/{package_id}/manifest.json" in str(error) and "does not exist" in str(error)
