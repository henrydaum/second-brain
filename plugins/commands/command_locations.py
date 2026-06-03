"""Slash command plugin for `/locations`."""

from pathlib import Path

from paths import DATA_DIR, INSTALLED_PLUGINS, ROOT_DIR, SANDBOX_PLUGINS
from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.formatters import format_locations
from state_machine.conversation import FormStep


KINDS = {
    "root": (ROOT_DIR, DATA_DIR),
    "plugins": (ROOT_DIR / "plugins", DATA_DIR),
    "sandbox": (SANDBOX_PLUGINS, SANDBOX_PLUGINS),
    "installed": (INSTALLED_PLUGINS, INSTALLED_PLUGINS),
}


class LocationsCommand(BaseCommand):
    """Slash-command handler for `/locations`."""
    name = "locations"
    description = "Show project and plugin directories"
    category = "System"

    def form(self, args, context):
        """Handle form."""
        return [FormStep("kind", "Choose which location map to show.", True, enum=list(KINDS))]

    def run(self, args, context):
        """Execute `/locations` for the active session."""
        root, data = KINDS.get(args.get("kind") or "root", KINDS["root"])
        return format_locations({"root_path": str(root), "root_tree": _tree(root), "data_path": str(data), "data_tree": _tree(data)})


def _tree(path):
    """Internal helper to handle tree."""
    path = Path(path)
    if not path.exists():
        return ["(missing)"]
    return [p.name + ("/" if p.is_dir() else "") for p in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))]
