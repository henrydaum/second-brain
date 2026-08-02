"""Browse and manage packages through guarded SDK requests.

Read-only catalogue requests run directly. Install, uninstall, and update are
classified unsafe by the kernel; the Request itself is the approval boundary,
so this tool never asks through a second permission mechanism.
"""

dependencies_files = []
dependencies_pip = []
requests = [
    "plugin.list", "plugin.install", "plugin.uninstall", "plugin.update",
]

from guest.bases import BaseTool


class ManagePackages(BaseTool):
    """Search, inspect, install, uninstall, and update store packages."""

    name = "manage_packages"
    description = (
        "Browse and manage the Second Brain package store. Search/info/list are "
        "read-only. Install, uninstall, and update require kernel approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "info", "installed", "install", "uninstall", "update"],
            },
            "target": {
                "type": "string",
                "description": "Package or bundle ID; optional substring for search.",
            },
        },
        "required": ["action"],
    }
    requires_services = []
    agent_prompt = (
        "## The package store\n"
        "If a requested capability is missing, use manage_packages to search "
        "the store, explain what you found, and install only through the "
        "approval-gated install action. New tools appear on the next turn."
    )

    def run(self, sdk, **kwargs):
        action = str(kwargs.get("action") or "").strip().lower()
        target = str(kwargs.get("target") or "").strip()
        try:
            if action == "search":
                return self._search(sdk, target)
            if action == "info":
                return self._info(sdk, target)
            if action == "installed":
                return self._installed(sdk)
            if action == "install":
                return self._mutate(sdk, action, target)
            if action == "uninstall":
                return self._mutate(sdk, action, target)
            if action == "update":
                return self._mutate(sdk, action, "")
            return sdk.fail(f"Unknown action: {action!r}.")
        except sdk.Denied as error:
            return sdk.fail(
                f"Package {action} was denied: {error}. The operation was not "
                "performed. STOP and do not retry."
            )
        except sdk.Failed as error:
            return sdk.fail(f"Package {action or 'store'} operation failed: {error}")

    def _catalogue(self, sdk):
        available = list(sdk.plugins.list(source="available") or [])
        installed = list(sdk.plugins.list(source="installed") or [])
        installed_paths = {item.get("path") for item in installed}
        combined = []
        for item in available:
            combined.append({**item, "installed": False})
        for item in installed:
            combined.append({**item, "installed": True})
        # Bundles are not files in the installed tree and are exposed through
        # the removable catalogue even before installation.
        for item in sdk.plugins.list(source="removable") or []:
            if item.get("family") == "bundles" and item.get("path") not in installed_paths:
                combined.append({**item, "installed": False})
        unique = {}
        for item in combined:
            unique[(item.get("id"), item.get("path"))] = item
        return sorted(
            unique.values(),
            key=lambda item: (item.get("family", ""), item.get("id", "")),
        )

    def _search(self, sdk, query):
        lowered = query.lower()
        items = [
            item for item in self._catalogue(sdk)
            if not lowered or lowered in str(item.get("id", "")).lower()
            or lowered in str(item.get("path", "")).lower()
        ]
        lines = [
            f"{item.get('id')} [{item.get('family', 'unknown')}]"
            + (" [helper]" if item.get("helper") else "")
            + (" [installed]" if item.get("installed") else "")
            for item in items
        ]
        summary = (
            f"{len(items)} store package(s)"
            + (f" matching {query!r}" if query else "")
            + (":\n" + "\n".join(lines) if lines else ".")
        )
        return sdk.ok({"items": items}, llm_summary=summary)

    def _info(self, sdk, target):
        if not target:
            return sdk.fail("'target' is required for info.")
        matches = [
            item for item in self._catalogue(sdk)
            if target in {str(item.get("id") or ""), str(item.get("path") or "")}
        ]
        if not matches:
            return sdk.fail(f"No package named {target!r}.")
        if len(matches) > 1:
            return sdk.fail(
                f"Package name {target!r} is ambiguous: "
                + ", ".join(str(item.get("path")) for item in matches)
            )
        item = matches[0]
        summary = (
            f"{item.get('id')} — {item.get('path')} "
            f"(family: {item.get('family')}; "
            f"{'installed' if item.get('installed') else 'available'})"
        )
        return sdk.ok(item, llm_summary=summary)

    def _installed(self, sdk):
        items = list(sdk.plugins.list(source="installed") or [])
        lines = [f"{item.get('id')} [{item.get('family', 'unknown')}]" for item in items]
        summary = (
            f"{len(items)} installed file(s):\n" + "\n".join(lines)
            if lines else "No packages installed."
        )
        return sdk.ok({"items": items}, llm_summary=summary)

    def _mutate(self, sdk, action, target):
        if action != "update" and not target:
            return sdk.fail(f"'target' is required for {action}.")
        if action == "install":
            output = sdk.plugins.install(target)
        elif action == "uninstall":
            output = sdk.plugins.uninstall(target)
        else:
            output = sdk.plugins.update()
        text = str(output or f"Package {action} completed.")
        if action == "install":
            text += "\n\nNew tools and commands become available on the next turn."
        return sdk.ok({"output": output}, llm_summary=text)
