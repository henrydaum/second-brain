"""Slash command plugin for `/permissions` — see and withdraw standing grants.

The whole approval design rests on a grant being **enumerable and
withdrawable**: an answer nobody can find is an answer nobody can take back.
Three settings hold every standing grant the system has, and before this they
were three keys among thirty in ``/config``, discoverable only by knowing
their names.

One verb, deliberately. Adding by hand stays in ``/config``; this exists to
show what is granted and to revoke it, and keeping it to that is what makes
the listing scannable. The two footnotes are load-bearing for the same
reason the tables are — "what can it reach" is not answered by the lists
alone, because scratch and the workspace tree are always free and the app's
own files are never grantable.
"""

from guest.bases import BaseCommand
from guest.forms import FormStep

_NETWORK = "Network"
_FOLDERS = "Writable folders"
_COMMANDS = "Commands"
_REVOKE_ALL = "Revoke every entry in this list"

#: Category label -> the setting behind it. The only place the three are
#: named together outside the policy that reads them.
SETTINGS = {
    _NETWORK: "net_allowed_hosts",
    _FOLDERS: "fs_writable_dirs",
    _COMMANDS: "shell_allowed_prefixes",
}


class PermissionsCommand(BaseCommand):
    """Review and revoke what you have allowed without being asked again."""

    name = "permissions"
    description = "See and withdraw standing permissions"
    category = "Capabilities"
    # Revoking narrows and needs no dialog of its own, but the handler writes
    # a kernel setting either way, and ``config.write`` is consequential —
    # ``tests/test_command_approval_declarations.py`` derives that from
    # ``policy.ALWAYS_UNSAFE`` rather than from anyone remembering. Asked up
    # front, where the scope can be stated, rather than mid-run.
    approval_actions = ("Revoke",)
    approval_actor_id = "user"
    requests = ["config.read", "config.write"]

    agent_prompt = (
        "## Standing permissions\n"
        "The user can review and withdraw standing permissions with "
        "/permissions. If a request of yours is refused and they ask why, "
        "that is where the answer lives — do not offer to edit the settings "
        "yourself."
    )

    def form(self, sdk, args):
        """Pick a list, then an entry, then confirm."""
        granted = {label: _entries(sdk, key) for label, key in SETTINGS.items()}
        filled = [label for label, items in granted.items() if items]
        if not filled:
            return []

        steps = []
        category = args.get("category")
        # One populated list is not a choice worth making. Skipping the step
        # also skips it on the *re-entry* after an answer, so the form does not
        # re-ask what it already knows.
        if len(filled) == 1:
            category = filled[0]
        else:
            steps.append(FormStep(
                "category", "Which permissions do you want to review?", True,
                enum=filled,
                enum_labels=[f"{label} ({len(granted[label])})"
                             for label in filled],
                columns=1))
            if category not in filled:
                return steps

        entries = granted[category]
        steps.append(FormStep(
            "entry", f"{category} — choose an entry to revoke.", True,
            enum=list(entries) + [_REVOKE_ALL], columns=1))
        return steps

    def run(self, sdk, args):
        """Show the standing grants, or revoke one."""
        granted = {label: _entries(sdk, key) for label, key in SETTINGS.items()}
        category = args.get("category")
        entry = args.get("entry")

        if not category or not entry:
            return _overview(sdk, granted)

        key = SETTINGS.get(category)
        if key is None:
            return f"Unknown permission list: {category}"
        entries = granted[category]

        if entry == _REVOKE_ALL:
            if not entries:
                return f"{category}: nothing to revoke."
            _write(sdk, key, [])
            return (f"Revoked all {len(entries)} {category.lower()} "
                    f"permissions.")
        if entry not in entries:
            return f"{entry} is not in {category.lower()}."
        _write(sdk, key, [item for item in entries if item != entry])
        return f"Revoked: {entry}"


def _entries(sdk, key) -> list:
    """One setting's entries as a list of strings.

    ``config_manager`` normalizes every kernel list key on load, so this is
    already a list in practice. The string branch is kept because both
    settings document a comma-separated form and a hand-edited file is read
    before it is ever normalized again — showing one long row where there are
    four grants would be a listing that lies.
    """
    raw = sdk.config.read(key) or []
    if isinstance(raw, str):
        raw = raw.split(",")
    return [text for item in raw if (text := str(item).strip())]


def _write(sdk, key, entries) -> None:
    """Persist a revoked list. The kernel announces the change itself."""
    sdk.config.write(key, entries)


def _overview(sdk, granted) -> str:
    """The landing view: every standing grant, and what is not one."""
    total = sum(len(items) for items in granted.values())
    if not total:
        return (
            "**No standing permissions.**\n\n"
            "Every request to reach the network, write outside the agent's "
            "own tree, or run a command is asked about individually. The "
            "approval dialog offers to remember an answer; anything you keep "
            "that way appears here.\n\n" + _always())

    parts = [
        "Nothing here was granted by a plugin — every entry is an answer "
        "you gave.",
        _table(sdk, _NETWORK, granted[_NETWORK], "Host", _COVERS_HOST),
        _table(sdk, _FOLDERS, granted[_FOLDERS], "Folder", _COVERS_FOLDER),
        _table(sdk, _COMMANDS, granted[_COMMANDS], "Command", _COVERS_COMMAND),
        _always(),
    ]
    return "\n\n".join(part for part in parts if part)


#: What each kind of entry covers, restated where it is being reviewed.
#: Per-category rather than per-entry, because the rule does not vary by
#: entry: ``policy._host_allowed`` matches every listed host on a dot
#: boundary, so ``api.search.brave.com`` covers its subdomains exactly as
#: ``example.com`` does. Two phrasings of one rule would read as a
#: distinction, and the folder line exists to restate the sharpest edge of
#: that grant — deletes are in it.
_COVERS_HOST = "this host and its subdomains"
_COVERS_FOLDER = "create, edit, move and delete, including subfolders"
_COVERS_COMMAND = "with any arguments"


def _table(sdk, title, entries, heading, covers) -> str:
    """One category's table, or nothing when it holds nothing."""
    if not entries:
        return ""
    rows = [[entry, covers] for entry in entries]
    return (f"**{title}** ({len(entries)})\n"
            + sdk.md.table([heading, "Covers"], rows))


def _always() -> str:
    """What the lists do not say, which is most of what people want to know.

    A permissions view that shows only the grants answers "what did I allow"
    and leaves "what can it reach" wrong in both directions — the agent is
    freer than this suggests inside its own tree, and more constrained
    everywhere near the app's own files.
    """
    return (
        "**Always allowed, without any grant:** scratch space, and the "
        "agent's own workspace tree — everything there runs in a subprocess, "
        "so writing code into it changes what the agent can *ask*, never what "
        "it may affect.\n\n"
        "**Never grantable, whatever is listed above:** Second Brain's own "
        "program files and installed packages. A folder here that contains "
        "them does not open them — otherwise the agent could edit the rules "
        "that decide what it may do.\n\n"
        "Add entries with `/config`; this view revokes them."
    )
