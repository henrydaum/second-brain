"""Every kernel command, checked as a family rather than one file at a time.

This replaces fourteen ``test_command_<name>_sandbox.py`` files that each
rigged a bespoke context to run one command and assert its exact output
string. That shape cost a full test file per command, broke on any wording
change, and still only covered whichever branch its author happened to rig.

What actually needs pinning is the same for all of them, and it is structural:
a migrated command must validate clean, adapt into something discovery will
register, and declare only Requests that exist. Behaviour that is genuinely
worth a test belongs with the *mechanism* it exercises (approval gating in
``test_command_approval_declarations``, mid-run asking in
``test_command_can_ask_mid_run``, the Request handlers themselves in the
``test_sandbox_*`` suite), not with the command that happened to trip over it.
"""

import pytest

from plugins.plugin_paths import iter_plugin_dirs
from sandbox import bridge
from sandbox.guest.requests import ALL_TYPES
from sandbox.validator import validate_file

_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
_KERNEL_COMMANDS = sorted((_ROOT / "bundled" / "commands").glob("command_*.py"))


def _ids(paths):
    return [path.stem for path in paths]


def test_the_kernel_still_ships_its_commands():
    """A glob that silently matched nothing would make every test below pass."""
    assert len(_KERNEL_COMMANDS) >= 10


@pytest.mark.parametrize("path", _KERNEL_COMMANDS, ids=_ids(_KERNEL_COMMANDS))
def test_kernel_command_is_migrated_and_conforms(path):
    """Kernel commands are all migrated; each must validate clean."""
    assert bridge.is_sandboxed(path), (
        f"{path.name} is not a migrated command — the kernel tree is migrated "
        "in full, so this is either a regression or a new command that still "
        "needs porting to the SDK"
    )
    report = validate_file(path)
    assert report.ok, f"{path.name} will not load in a box:\n{report.render()}"


@pytest.mark.parametrize("path", _KERNEL_COMMANDS, ids=_ids(_KERNEL_COMMANDS))
def test_kernel_command_adapts_into_something_discovery_registers(path):
    """The bridge must produce a native-looking class discovery can find.

    ``__module__`` is asserted because discovery rejects classes that do not
    belong to the module it just loaded — the one failure mode that makes a
    perfectly valid command silently invisible.
    """
    module = bridge.adapt(path, family="command")
    assert module is not None, f"{path.name} did not adapt"

    classes = [
        value for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__
    ]
    assert len(classes) == 1, f"{path.name} adapted to {len(classes)} classes"
    adapter = classes[0]

    assert getattr(adapter, "name", ""), f"{path.name} adapts with no name"
    assert getattr(adapter, "description", ""), (
        f"/{adapter.name} has no description; it would render blank in /commands"
    )
    assert callable(getattr(adapter, "run", None))
    assert getattr(adapter, "_sandboxed", False)
    assert getattr(adapter, "approval_prompt", "")


@pytest.mark.parametrize("path", _KERNEL_COMMANDS, ids=_ids(_KERNEL_COMMANDS))
def test_kernel_command_declares_only_real_requests(path):
    """``requests`` is load-bearing — it is what one approval buys.

    A typo used to read as documentation; now it silently narrows the grant,
    so a name outside the catalogue is a bug rather than a spelling mistake.
    """
    declared = validate_file(path).declarations.get("requests") or ()
    unknown = sorted(set(declared) - set(ALL_TYPES))
    assert not unknown, f"{path.name} declares non-existent Requests: {unknown}"


def test_command_names_are_unique_across_every_tree():
    """Two commands answering to one slash name is a coin flip at load order."""
    seen = {}
    clashes = []
    for family, directory in iter_plugin_dirs():
        if family != "command" or not directory.exists():
            continue
        for path in sorted(directory.glob("command_*.py")):
            name = validate_file(path).declarations.get("name") \
                or path.stem.split("_", 1)[-1]
            if name in seen and seen[name] != path:
                clashes.append(f"/{name}: {seen[name].name} and {path.name}")
            seen[name] = path
    assert not clashes, "duplicate command names:\n  " + "\n  ".join(clashes)
