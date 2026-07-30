"""Slash command plugin for `/locations`."""

from guest.bases import BaseCommand


#: The three trees, plus the two roots they hang off. Named for where the code
#: came from, which is the only thing the tree name has ever carried.
KINDS = ["root", "bundled", "installed", "workspace"]


class LocationsCommand(BaseCommand):
    """Slash-command handler for `/locations`."""
    name = "locations"
    description = "Show project and plugin directories"
    category = "System"
    requests = ["paths.get", "fs.list"]

    def form(self, sdk, args):
        """Handle form."""
        return [{
            "name": "kind",
            "prompt": "Choose which location map to show.",
            "required": True,
            "enum": KINDS,
        }]

    def run(self, sdk, args):
        """Execute `/locations` for the active session."""
        project = sdk.paths.get("project")
        data_dir = sdk.paths.get("data")
        bundled = sdk.paths.get("bundled")
        installed = sdk.paths.get("installed")
        workspace = sdk.paths.get("workspace")
        locations = {
            "root": (project, data_dir),
            "bundled": (bundled, bundled),
            "installed": (installed, installed),
            "workspace": (workspace, workspace),
        }
        root, data = locations.get(
            args.get("kind") or "root", locations["root"])
        return _format_locations(
            root, _tree(sdk, root), data, _tree(sdk, data))


def _tree(sdk, path):
    """Internal helper to handle tree."""
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed as exc:
        # Matched on a fragment rather than the whole formatted message: the
        # previous version compared against an exact f-string containing the
        # path, so rewording the handler silently turned "(missing)" into a
        # raised error.
        if "no such directory or file" in exc.error:
            return ["(missing)"]
        raise
    entries.sort(key=lambda entry: (
        not entry["is_dir"], entry["name"].lower()))
    return [
        entry["name"] + ("/" if entry["is_dir"] else "")
        for entry in entries
    ]


def _format_locations(root_path, root_tree, data_path, data_tree):
    """Render the same fenced location map as the native command."""
    def section(label, path, tree):
        listing = "\n".join(tree) if tree else "(empty)"
        return f"**{label}**\n`{path}`\n```\n{listing}\n```"

    return "\n\n".join([
        section("Project root", root_path, root_tree),
        section("Data directory", data_path, data_tree),
    ])
