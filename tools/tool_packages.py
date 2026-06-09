"""Package-store tool — let the agent browse and manage store packages.

Bridges the agent to the same package manager that backs /packages, so a
missing capability becomes a recoverable state: search the store, show the
user what an install would do, and (with explicit approval) install it.
Mutating actions are gated through context.approve_command, and the tool is
not background-safe — an unattended session can never self-install.
"""


dependencies_files = []
dependencies_pip = []

import logging

from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("PackagesTool")


class ManagePackages(BaseTool):
    """Search, inspect, install, uninstall, and update store packages."""

    name = "manage_packages"
    description = (
        "Browse and manage the Second Brain package store. Actions:\n"
        "- search: list available store packages, optionally filtered by a query.\n"
        "- info: show one package's file path, family, and dependencies.\n"
        "- installed: list currently installed packages.\n"
        "- install: install a package (and its dependencies) by stem. Requires user approval.\n"
        "- uninstall: remove an installed package by stem. Requires user approval.\n"
        "- update: re-install every installed file whose store copy changed. Requires user approval.\n\n"
        "Use this when the user asks for a capability you don't currently have "
        "(a tool, parser, frontend, or service) — search the store first, then "
        "propose the install. Never install without telling the user what it adds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "info", "installed", "install", "uninstall", "update"],
                "description": "What to do against the package store.",
            },
            "target": {
                "type": "string",
                "description": (
                    "Package stem for info/install/uninstall (e.g. 'tool_web_search', "
                    "'parse_pdf', 'frontend_telegram'). For search, an optional "
                    "substring query."
                ),
            },
        },
        "required": ["action"],
    }
    requires_services = []
    max_calls = 5
    background_safe = False  # installs need a human watching

    agent_prompt = (
        "## The package store\n"
        "Second Brain is a small kernel plus installable packages. If the user asks for "
        "something you have no tool for (web search, PDF parsing, email, scheduling, "
        "shell access, etc.), don't just say you can't — use manage_packages to search "
        "the store for a package that adds the capability, tell the user what you found "
        "and what it would install, and install it once they agree (an approval prompt "
        "will confirm). Newly installed tools become available on your next turn."
    )

    def run(self, context, **kwargs) -> ToolResult:
        from plugins.commands.helpers import package_manager
        from plugins.commands.helpers.store_backend import StoreBackendError

        action = (kwargs.get("action") or "").strip()
        target = (kwargs.get("target") or "").strip()
        root_dir = getattr(context, "root_dir", None) or _default_root()

        try:
            if action == "search":
                return self._search(package_manager, root_dir, target)
            if action == "info":
                return self._info(package_manager, root_dir, target)
            if action == "installed":
                return self._installed(package_manager)
            if action == "install":
                return self._install(package_manager, context, root_dir, target)
            if action == "uninstall":
                return self._uninstall(package_manager, context, root_dir, target)
            if action == "update":
                return self._update(package_manager, context, root_dir)
            return ToolResult.failed(f"Unknown action: {action!r}.")
        except (package_manager.PackageError, StoreBackendError) as e:
            return ToolResult.failed(f"Package {action or 'store'} operation failed: {e}")

    # ── read-only actions ────────────────────────────────────────────────

    def _search(self, pm, root_dir, query) -> ToolResult:
        items = pm.search_packages(root_dir, query)
        installed = {item["path"] for item in pm.installed_packages()}
        lines = [
            f"{item['id']} [{item['family']}]"
            + (" [helper]" if item.get("helper") else "")
            + (" [installed]" if item["path"] in installed else "")
            for item in items
        ]
        summary = (
            f"{len(items)} store package(s)" + (f" matching {query!r}" if query else "") + ":\n"
            + "\n".join(lines)
            if lines else f"No store packages found{f' matching {query!r}' if query else ''}."
        )
        return ToolResult(data={"items": items}, llm_summary=summary)

    def _info(self, pm, root_dir, target) -> ToolResult:
        if not target:
            return ToolResult.failed("'target' is required for info.")
        info = pm.package_info(root_dir, target)
        summary = (
            f"{info['id']} — {info['path']} (family: {info['family']})\n"
            f"File dependencies: {', '.join(info['dependencies_files']) or 'none'}\n"
            f"Pip dependencies: {', '.join(info['dependencies_pip']) or 'none'}"
        )
        return ToolResult(data=info, llm_summary=summary)

    def _installed(self, pm) -> ToolResult:
        items = pm.installed_packages()
        lines = [f"{item['id']} [{item['family']}]" + (" [helper]" if item.get("helper") else "") for item in items]
        summary = f"{len(items)} installed file(s):\n" + "\n".join(lines) if lines else "No packages installed."
        return ToolResult(data={"items": items}, llm_summary=summary)

    # ── mutating actions (approval-gated) ────────────────────────────────

    def _install(self, pm, context, root_dir, target) -> ToolResult:
        if not target:
            return ToolResult.failed("'target' is required for install.")
        plan = pm.build_install_plan(root_dir, target)
        detail = "Files:\n" + "\n".join(f"  {f.path}" for f in plan.files)
        if plan.pip_packages:
            detail += "\nPython packages (pip install):\n" + "\n".join(f"  {p}" for p in plan.pip_packages)
        if not self._approved(context, f"Install package '{target}'", detail):
            return self._denied(context, f"install of '{target}'")
        result = pm.execute_install_plan(plan, context)
        return ToolResult(
            data={"lines": result.lines},
            llm_summary=result.text() + "\n\nNewly installed tools/commands become available on your next turn.",
        )

    def _uninstall(self, pm, context, root_dir, target) -> ToolResult:
        if not target:
            return ToolResult.failed("'target' is required for uninstall.")
        plan = pm.build_uninstall_plan(target, root_dir=root_dir)
        detail = "Files to remove:\n" + "\n".join(f"  {rel}" for rel in plan.remove_files)
        if plan.pip_packages:
            detail += "\nPython packages to pip-uninstall:\n" + "\n".join(f"  {p}" for p in plan.pip_packages)
        if not self._approved(context, f"Uninstall package '{target}'", detail):
            return self._denied(context, f"uninstall of '{target}'")
        result = pm.execute_uninstall_plan(plan, context)
        return ToolResult(data={"lines": result.lines}, llm_summary=result.text())

    def _update(self, pm, context, root_dir) -> ToolResult:
        outdated = pm.outdated_packages(root_dir)
        if not outdated:
            return ToolResult(llm_summary="All installed packages are up to date.")
        detail = "Files with store updates:\n" + "\n".join(f"  {rel}" for rel in outdated)
        if not self._approved(context, f"Update {len(outdated)} installed package file(s)", detail):
            return self._denied(context, "package update")
        result = pm.update_packages(root_dir, context)
        return ToolResult(data={"lines": result.lines}, llm_summary=result.text())

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _approved(context, title: str, detail: str) -> bool:
        approve = getattr(context, "approve_command", None)
        if approve is None:
            return False  # no live session to ask — treat as deny
        try:
            return bool(approve(title, detail))
        except Exception as e:
            logger.error(f"Approval callback failed: {e}")
            return False

    @staticmethod
    def _denied(context, what: str) -> ToolResult:
        reason = getattr(context, "approval_denial_reason", "") or "User denied the request."
        return ToolResult.failed(
            f"{reason} The {what} was not performed. STOP — do not retry. Ask the user how to proceed."
        )


def _default_root():
    from paths import ROOT_DIR
    return ROOT_DIR
