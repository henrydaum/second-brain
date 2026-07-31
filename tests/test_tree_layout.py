"""The layout as a *claim*, not a by-product of what happens to be installed.

A declared root that only appears once something lands in it does not tell
anybody where things go. Three things broke on that, all visible at once:
``/locations`` showed the three trees with three different folder lists and no
way to tell which difference meant anything; an agent writing its first tool
had to know to create ``workspace/tools/`` first; and ``scripts/`` — the safe
alternative to ``proc.run`` — existed in no tree at all, because the only code
that ever made a directory was the watcher, and the watcher skips unwatched
roots.

Then uninstalling anything deleted them again.
"""

import pytest

import trees
from bundled.commands.helpers import package_manager
from tests.support import retarget_trees


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Three empty local trees at a throwaway location."""
    return retarget_trees(monkeypatch, tmp_path)


def test_every_tree_holds_every_declared_root(layout):
    """The whole point: four trees, one shape."""
    trees.materialize()

    expected = {root.name for root in trees.ROOTS}
    for name, path in layout.items():
        assert {p.name for p in path.iterdir() if p.is_dir()} == expected, (
            f"{name} does not hold the declared roots")


def test_the_unwatched_root_is_materialized_too(layout):
    """``scripts/`` is ``watched=False``, which is why it existed nowhere.

    The watcher was the only thing creating directories and it iterates
    ``watched_only=True``, so the one root the agent is *meant* to reach for
    instead of a shell was the one root that never appeared.
    """
    trees.materialize()

    for path in layout.values():
        assert (path / "scripts").is_dir()


def test_materialize_is_idempotent_and_keeps_what_is_there(layout):
    """It runs every boot, so a second run must be free and harmless."""
    trees.materialize()
    keeper = layout["workspace"] / "tools" / "tool_keep.py"
    keeper.write_text("VALUE = 1\n", encoding="utf-8")

    assert trees.materialize() == ()
    assert keeper.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_uninstalling_does_not_undo_the_layout(layout, monkeypatch):
    """``_remove_empty_dirs`` ran on every uninstall and swept the roots away.

    That is what made the trees disagree in the first place: which folders
    ``installed/`` had depended on what you happened to have installed and in
    what order you had removed things.
    """
    installed = layout["installed"]
    monkeypatch.setattr(package_manager, "INSTALLED_PLUGINS", installed)
    trees.materialize()
    (installed / "tools" / "leftovers").mkdir(parents=True)

    package_manager._remove_empty_dirs()

    assert (installed / "tools").is_dir()
    assert (installed / "scripts").is_dir()
    # Anything that is *not* a declared root is still tidied away.
    assert not (installed / "tools" / "leftovers").exists()


def test_the_family_list_is_derived_from_the_layout():
    """``/packages`` hardcoded six categories and zipped its labels against
    them, silently discarding the ``llm`` and ``parsers`` counts it had
    already computed — two whole categories invisible in a menu built out of
    their own data."""
    from sandbox.handlers.kernel import _plugin_list

    families = _plugin_list(object(), {"source": "families"}).data

    assert set(families) >= {root.name for root in trees.ROOTS}
    assert "llm" in families and "parsers" in families
    # Not tree roots, but the store carries them and a menu has to offer them.
    assert set(families) >= set(package_manager.EXTRA_FAMILIES)
