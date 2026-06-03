"""Slash command plugin for `/packages`."""

from plugins.BaseCommand import BaseCommand
from plugins.commands.helpers import package_manager
from plugins.commands.helpers.store_backend import StoreBackendError
from state_machine.conversation import FormStep


ACTIONS = ["search", "list", "info", "install", "uninstall"]


class PackagesCommand(BaseCommand):
    """Install and uninstall package-store plugins."""
    name = "packages"
    description = "Search, install, list, inspect, or uninstall packages"
    category = "System"

    def form(self, args, context):
        steps = [FormStep("action", "Choose a package action.", True, enum=ACTIONS)]
        action = args.get("action")
        if action == "search":
            steps.append(FormStep("query", "Enter a package search query.", False, default=""))
        if action in {"install", "info"}:
            steps.append(FormStep("package_id", "Enter the package id.", True))
        if action == "uninstall":
            steps.append(FormStep("package_id", "Choose the package to uninstall.", True, enum=[p["id"] for p in package_manager.installed_packages()], columns=2))
        return steps

    def run(self, args, context):
        action = args.get("action") or "list"
        try:
            if action == "search":
                return _format_index(package_manager.search_packages(context.root_dir, args.get("query", "")))
            if action == "list":
                return _format_installed(package_manager.installed_packages())
            if action == "info":
                return _format_manifest(package_manager.package_info(context.root_dir, args.get("package_id", "")))
            if action == "install":
                return package_manager.install_package(context.root_dir, args.get("package_id", ""), context).text()
            if action == "uninstall":
                return package_manager.uninstall_package(args.get("package_id", ""), context).text()
            return f"Unknown action: {action}"
        except (package_manager.PackageError, StoreBackendError) as e:
            return f"Package {action} failed: {e}"


def _format_index(items: list[dict]) -> str:
    if not items:
        return "No packages found."
    lines = ["Packages:"]
    for item in items:
        desc = item.get("description") or ""
        suffix = f" - {desc}" if desc else ""
        lines.append(f"  {item.get('id', '')}{suffix}")
    return "\n".join(lines)


def _format_installed(items: list[dict]) -> str:
    if not items:
        return "No packages installed."
    lines = ["Installed packages:"]
    for item in items:
        mode = "requested" if item.get("requested") else "auto"
        deps = item.get("requires") or []
        suffix = f" (requires: {', '.join(deps)})" if deps else ""
        lines.append(f"  {item.get('id')} [{mode}]{suffix}")
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
