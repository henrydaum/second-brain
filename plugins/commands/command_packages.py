"""Slash command plugin for `/packages`."""

import re

from plugins.BaseCommand import BaseCommand
from plugins.commands.helpers import package_manager
from plugins.commands.helpers.store_backend import StoreBackendError
from state_machine.conversation import FormStep


ACTIONS = ["search", "categories", "list", "info", "install", "uninstall"]
ACTION_LABELS = ["Search available", "Browse categories", "List installed", "Package info", "Install package", "Uninstall package"]

_FAMILY_BLURB = {
    "bundle": "curated sets — install one, get many",
    "service": "long-lived backends (LLMs, OCR, embeddings, integrations)",
    "tool": "agent-callable tools",
    "task": "pipeline tasks",
    "command": "slash commands",
    "frontend": "chat frontends",
    "parser": "file parsers",
    "helper": "shared helper modules pulled in by other packages",
}


class PackagesCommand(BaseCommand):
    """Install and uninstall package-store plugins."""
    name = "packages"
    description = "Search, install, list, inspect, or uninstall packages"
    category = "System"

    def form(self, args, context):
        steps = [FormStep("action", "Choose a package action.", True, enum=ACTIONS, enum_labels=ACTION_LABELS)]
        action = args.get("action")
        if action == "search":
            steps.append(FormStep("query", "Enter a package search query.", False, default=""))
        if action in {"install", "info"}:
            steps.append(FormStep("package_id", "Enter the package id.", True))
        if action == "uninstall":
            steps.append(FormStep("package_id", "Choose the package to uninstall.", True, enum=[p["id"] for p in package_manager.installed_packages()], columns=2))
            if args.get("package_id"):
                for package_id, cleanup in package_manager.uninstall_cleanup_plans(args["package_id"]):
                    steps.append(FormStep(_cleanup_arg(package_id), package_manager.cleanup_prompt(cleanup), True, "boolean"))
        return steps

    def run(self, args, context):
        action = args.get("action") or "list"
        try:
            if action == "search":
                items = package_manager.search_packages(context.root_dir, args.get("query", ""))
                installed = {p.get("id") for p in package_manager.installed_packages()}
                return _format_index(items, installed)
            if action == "categories":
                return _format_categories(package_manager.search_packages(context.root_dir))
            if action == "list":
                return _format_installed(package_manager.installed_packages())
            if action == "info":
                return _format_manifest(package_manager.package_info(context.root_dir, args.get("package_id", "")))
            if action == "install":
                return package_manager.install_package(context.root_dir, args.get("package_id", ""), context).text()
            if action == "uninstall":
                approvals = {pkg: bool(args.get(_cleanup_arg(pkg))) for pkg, _cleanup in package_manager.uninstall_cleanup_plans(args.get("package_id", "")) if _cleanup_arg(pkg) in args}
                return package_manager.uninstall_package(args.get("package_id", ""), context, cleanup_approvals=approvals).text()
            return f"Unknown action: {action}"
        except (package_manager.PackageError, StoreBackendError) as e:
            return f"Package {action} failed: {e}"


def _is_bundle(item: dict) -> bool:
    return "bundle" in (item.get("tags") or [])


def _cleanup_arg(package_id: str) -> str:
    return "cleanup__" + re.sub(r"[^a-zA-Z0-9_]", "_", package_id)


def _format_index(items: list[dict], installed: set[str] = frozenset()) -> str:
    if not items:
        return "No packages found."

    def line(item: dict) -> str:
        mark = "✓" if item.get("id") in installed else " "
        desc = item.get("description") or ""
        tags = [t for t in (item.get("tags") or []) if t != "bundle"]
        tagstr = ("   " + " ".join(f"#{t}" for t in tags)) if tags else ""
        return f"  {mark} {item.get('id', '')}{(' — ' + desc) if desc else ''}{tagstr}"

    # Bundles are the curated front door — list them first and apart.
    bundles = [i for i in items if _is_bundle(i)]
    rest = [i for i in items if not _is_bundle(i)]
    out: list[str] = []
    if bundles:
        out.append("Bundles — install one to get a curated set:")
        out.extend(line(i) for i in bundles)
    if rest:
        if out:
            out.append("")
        out.append("Packages:")
        out.extend(line(i) for i in rest)
    out.append("")
    out.append("✓ = installed.  Tip: /packages categories to browse by family, or "
               "search a family (`tool`, `parser`, `bundle`) or any term (`email`, `ocr`).")
    return "\n".join(out)


def _format_categories(items: list[dict]) -> str:
    from collections import Counter
    counts = Counter(package_manager.package_family(i) for i in items)
    if not counts:
        return "No packages found."
    # Whatever families exist, derived from the names: bundle first, then by
    # descending count, then alphabetical. New families appear here for free.
    families = sorted(counts, key=lambda f: (f != "bundle", -counts[f], f))
    out = ["Categories — /packages search <name> to list one:"]
    for family in families:
        blurb = _FAMILY_BLURB.get(family, "")
        out.append(f"  {family:9} {counts[family]:>3}{('  — ' + blurb) if blurb else ''}")
    return "\n".join(out)


def _format_installed(items: list[dict]) -> str:
    if not items:
        return "No packages installed.\nUse /packages search to browse available packages, then /packages install <id>."
    lines = ["Installed packages:"]
    for item in items:
        mode = "" if item.get("requested") else " [dependency]"
        deps = item.get("requires") or []
        suffix = f" (requires: {', '.join(deps)})" if deps else ""
        lines.append(f"  {item.get('id')}{mode}{suffix}")
    return "\n".join(lines)


def _format_manifest(manifest: dict) -> str:
    lines = [manifest.get("id", ""), manifest.get("description", "")]
    requires = manifest.get("requires") or []
    if requires:
        lines.append(f"Requires: {', '.join(requires)}")
    files = manifest.get("files") or []
    if files:
        lines.append("Files:")
        lines.extend(f"  {path}" for path in files)
    entrypoints = manifest.get("entrypoints")
    if entrypoints:
        lines.append("Entrypoints:")
        lines.extend(f"  {path}" for path in entrypoints)
    return "\n".join(line for line in lines if line)
