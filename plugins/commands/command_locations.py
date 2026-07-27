"""Slash command plugin for `/locations`."""

from guest.bases import BaseCommand


KINDS = ["root", "plugins", "sandbox", "installed"]


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
        sandbox_plugins = sdk.paths.get("sandbox_plugins")
        installed_plugins = sdk.paths.get("installed_plugins")
        locations = {
            "root": (project, data_dir),
            "plugins": (_join(project, "plugins"), data_dir),
            "sandbox": (sandbox_plugins, sandbox_plugins),
            "installed": (installed_plugins, installed_plugins),
        }
        root, data = locations.get(
            args.get("kind") or "root", locations["root"])
        return _format_locations(
            root, _tree(sdk, root), data, _tree(sdk, data))


def _join(root, name):
    """Join an application root without consulting the guest environment."""
    separator = "\\" if "\\" in root else "/"
    return root.rstrip("/\\") + separator + name


def _tree(sdk, path):
    """Internal helper to handle tree."""
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed as exc:
        if exc.error == f"not a directory: {path}":
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
