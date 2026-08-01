"""Slash command plugin for `/locations`."""

from guest.bases import BaseCommand


#: The three code trees, plus the two directories they hang off. Named for
#: where the code came from, which is the only thing a tree name has ever
#: carried.
KINDS = ["root", "bundled", "installed", "workspace"]
KIND_LABELS = [
    "Project root and data directory",
    "Bundled — ships with the app",
    "Installed — from the package store",
    "Workspace — written by the agent",
]
_TREE_BLURB = {
    "bundled": "Ships with the app. Read-only in practice.",
    "installed": "What the package store has put here.",
    "workspace": "What the agent has written. Free to edit, always sandboxed.",
}


class LocationsCommand(BaseCommand):
    """Slash-command handler for `/locations`."""
    name = "locations"
    description = "Show project and plugin directories"
    category = "System"
    requests = ["paths.get", "fs.list", "plugin.list"]

    def form(self, sdk, args):
        """Handle form."""
        return [{
            "name": "kind",
            "prompt": "Choose which location map to show.",
            "required": True,
            "enum": KINDS,
            "enum_labels": KIND_LABELS,
        }]

    def run(self, sdk, args):
        """Execute `/locations` for the active session."""
        kind = args.get("kind") or "root"
        if kind == "root":
            return _format_root(
                sdk, sdk.paths.get("project"), sdk.paths.get("data"))
        if kind not in KINDS:
            return f"Unknown location: {kind}"
        return _format_tree(sdk, kind, sdk.paths.get(kind))


def _format_root(sdk, project, data):
    """The two directories the whole layout hangs off."""
    return "\n\n".join([
        _section("Project root", project, _entries(sdk, project)),
        _section("Data directory", data, _entries(sdk, data)),
    ])


def _format_tree(sdk, kind, path):
    """One tree, one section per declared root.

    Every kind but ``root`` used to map to ``(path, path)`` and get rendered
    through the two-directory shape above — so ``/locations bundled`` printed
    the same directory twice, once labelled "Project root" and once "Data
    directory", both of them wrong. And because the listing was one level deep,
    the three trees came out as three near-identical lists of folder names with
    nothing to tell them apart.

    The roots are what actually distinguishes a tree, so they are what this
    shows: all of them, including the empty ones. An empty ``tools/`` is
    information — it says this tree *may* hold tools and does not — which a
    listing that simply omits it cannot express.
    """
    sections = [
        f"**{kind}**\n`{path}`\n{_TREE_BLURB.get(kind, '')}".rstrip()]
    for root in _roots(sdk):
        inside = _join(path, root)
        sections.append(_section(
            f"{root}/", inside,
            _entries(sdk, inside, missing="(not created)")))
    return "\n\n".join(sections)


def _join(root, name):
    """Join a path in the separator style the platform already gave us.

    A command cannot import ``os.path`` — and does not need to, since the
    kernel handed it an absolute path whose separator is the answer.
    """
    separator = "\\" if "\\" in root else "/"
    return root.rstrip("/\\") + separator + name


def _roots(sdk):
    """The declared roots, in the layout's own order.

    Asked of the kernel rather than restated here: ``trees.py`` is the layout
    authority and a command cannot import it, so a hardcoded copy would be a
    second declaration that silently goes stale the day a ninth root is added.
    """
    try:
        families = sdk.plugins.list(source="families") or []
    except sdk.Failed:
        return []
    # ``bundles`` comes back too; it is a store family rather than a tree
    # root and no tree has a directory for it.
    return [name for name in families if name != "bundles"]


def _section(label, path, listing):
    """One fenced block. Fenced because a rich renderer folds the single
    newlines of a bare listing into one paragraph."""
    body = "\n".join(listing) if listing else "(empty)"
    return f"**{label}**\n`{path}`\n```\n{body}\n```"


def _entries(sdk, path, missing="(missing)"):
    """Names in one directory, folders first."""
    try:
        entries = sdk.fs.list(path, details=True)
    except sdk.Failed as exc:
        # Matched on a fragment rather than the whole formatted message: the
        # previous version compared against an exact f-string containing the
        # path, so rewording the handler silently turned "(missing)" into a
        # raised error.
        if "no such directory or file" in exc.error:
            return [missing]
        raise
    entries.sort(key=lambda entry: (
        not entry["is_dir"], entry["name"].lower()))
    return [
        entry["name"] + ("/" if entry["is_dir"] else "")
        for entry in entries
        # This is a map of what is *here*, and a bytecode cache is not. Left
        # in, it appeared in almost every root and was the only content of
        # several, so an empty root read as an occupied one.
        if entry["name"] != "__pycache__"
    ] or []
