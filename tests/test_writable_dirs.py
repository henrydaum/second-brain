"""``fs_writable_dirs``: where the user has opened the filesystem, and where
no list of theirs can reach.

The setting is the filesystem twin of ``net_allowed_hosts`` — a person deciding
what code may touch, kept in config so the code being decided about cannot
write it. What is worth testing is almost entirely the carve-out: the natural
value for this setting on a developer's machine is the folder they keep
projects in, and on this machine that folder contains Second Brain itself.

Most tests run against a **synthetic layout** — ``paths.ROOT_DIR`` and
``paths.DATA_DIR`` pointed inside ``tmp_path`` — so they describe the rule
rather than this checkout. ``_protected`` re-reads both from the module on
every call, which is what makes that possible. Two tests deliberately use the
real ones, because "the app protects *itself*" is the claim that matters and
a synthetic layout cannot make it.
"""

import pytest

import paths
from runtime.context import set_kernel_parts
from sandbox.guest.requests import FS_DELETE, FS_MOVE, FS_WRITE, Request
from sandbox.policy import SAFE, UNSAFE, Chain, classify


def _allow(*dirs):
    """Set the user's writable list for the duration of a test."""
    set_kernel_parts(config={"fs_writable_dirs": [str(d) for d in dirs]})


@pytest.fixture(autouse=True)
def _empty_list():
    """Every test states its own list; none inherits the last one."""
    _allow()
    yield
    _allow()


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """Move the app somewhere else, and hand back somebody's project folder.

    ``tmp_path`` is inside the repo under this project's pytest settings, so
    without this the real carve-out would (correctly) refuse every grant and
    every test below would pass for the wrong reason.

    **``DATA_DIR`` is deliberately left alone.** ``trees`` snapshots it into
    ``WORKSPACE`` at *import* time, so patching it poisons the workspace path
    permanently for the rest of the session if ``trees`` happens not to have
    been imported yet — a failure that depends on test ordering and points
    nowhere near here. Nothing needs it patched: ``tmp_path`` is on the repo's
    drive and the real ``DATA_DIR`` is not above it.
    """
    app, project = tmp_path / "app", tmp_path / "work"
    for directory in (app, project):
        directory.mkdir()
    monkeypatch.setattr(paths, "ROOT_DIR", app)
    return project


def _write(path):
    """Classify a write to ``path`` with an ordinary chain."""
    return classify(Request(FS_WRITE, {"path": str(path)}),
                    Chain(root="repl").push("tool_edit"))


def _writable(path) -> bool:
    """Whether writing there needs no dialog."""
    return _write(path).level == SAFE


# ── the grant ─────────────────────────────────────────────────────────

def test_a_listed_directory_is_writable_including_below_it(layout):
    """Subfolders are covered — a project is a tree, not a folder."""
    _allow(layout)
    assert _writable(layout / "notes.md")
    assert _writable(layout / "src" / "deep" / "mod.py")


def test_nothing_is_writable_by_default(layout):
    """The list ships empty, so this changes nothing until somebody edits it."""
    assert not _writable(layout / "notes.md")


def test_an_unlisted_sibling_is_not_writable(layout):
    """The grant is the tree named, not its parent."""
    _allow(layout / "project")
    assert _writable(layout / "project" / "a.txt")
    assert not _writable(layout / "other" / "a.txt")


def test_deletes_and_moves_are_included(layout):
    """The decision was to match scratch semantics, deletes and all."""
    _allow(layout)
    chain = Chain(root="repl").push("tool_edit")
    assert classify(Request(FS_DELETE, {"path": str(layout / "a")}),
                    chain).level == SAFE
    assert classify(Request(FS_MOVE, {"src": str(layout / "a"),
                                      "dst": str(layout / "b")}),
                    chain).level == SAFE


def test_a_move_needs_both_ends(layout):
    """Dragging a file out of an allowed folder changes somewhere unlisted."""
    _allow(layout / "project")
    moved = Request(FS_MOVE, {"src": str(layout / "project" / "a"),
                              "dst": str(layout / "elsewhere" / "a")})
    assert classify(moved, Chain(root="repl").push("tool_edit")).level == UNSAFE


def test_the_ledger_can_tell_the_three_grants_apart(layout):
    """Scratch, the agent's own tree, and somebody's actual work."""
    _allow(layout)
    assert _write(layout / "a.txt").reason == "write in a directory you allowed"


# ── the carve-out, which is the whole point ───────────────────────────

def test_listing_the_app_itself_grants_nothing():
    """The failure this must be caught for, stated as plainly as possible.

    Deliberately against the *real* ``ROOT_DIR``: if a user grant could reach
    ``sandbox/policy.py``, the agent could edit the classifier that decides
    what it is allowed to do — the self-modification breach the whole design
    exists to prevent, arriving through a setting meant as a convenience.
    """
    _allow(paths.ROOT_DIR)
    assert not _writable(paths.ROOT_DIR / "sandbox" / "policy.py")
    assert not _writable(paths.ROOT_DIR / "main.pyw")
    assert not _writable(paths.ROOT_DIR / "bundled" / "commands" / "x.py")


def test_a_parent_directory_is_the_case_that_filtering_the_list_would_miss():
    """Why the check is against the target and not against the listed folder.

    Nobody lists the app's own directory on purpose. They list the folder they
    keep code in, and on a developer's machine the app is one of the things in
    it — so the grant arrives from a path that looks nothing like the app's.
    """
    _allow(paths.ROOT_DIR.parent)
    assert not _writable(paths.ROOT_DIR / "sandbox" / "policy.py")
    # ...while a genuine sibling of the app is exactly what was intended.
    assert _writable(paths.ROOT_DIR.parent / "some-other-project" / "main.py")


def test_installed_packages_are_not_reachable_either():
    """A free write there would be a way around ``plugin.install``."""
    _allow(paths.DATA_DIR)
    assert not _writable(paths.DATA_DIR / "installed" / "tools" / "t.py")
    assert not _writable(paths.DATA_DIR / "config.json")


def test_the_agents_own_tree_stays_free_inside_the_protected_area():
    """``workspace`` is the deliberate hole, and does not depend on the list.

    It is a scratch root in its own right, so free authorship there survives
    both an empty setting and a listed ``DATA_DIR``.
    """
    workspace = paths.DATA_DIR / "workspace" / "tools" / "tool_new.py"
    assert _writable(workspace)
    _allow(paths.DATA_DIR)
    assert _writable(workspace)


def test_a_symlink_out_of_an_allowed_directory_does_not_carry_the_grant(layout):
    """Resolution happens before comparison, which is the safe order."""
    outside, allowed = layout / "outside", layout / "allowed"
    outside.mkdir()
    allowed.mkdir()
    try:
        (allowed / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need privileges on this platform")

    _allow(allowed)
    assert _writable(allowed / "real.txt")
    assert not _writable(allowed / "escape" / "reached.txt")


# ── failing closed ────────────────────────────────────────────────────

def test_a_malformed_entry_is_skipped_rather_than_widening(layout):
    """One unusable path must not make the rest of the list unusable."""
    set_kernel_parts(config={"fs_writable_dirs": ["", "   ", str(layout)]})
    assert _writable(layout / "a.txt")


def test_a_comma_separated_string_is_accepted_like_the_host_list(layout):
    """Same shape as ``net_allowed_hosts``, so the same typing works."""
    first, second = layout / "one", layout / "two"
    set_kernel_parts(config={"fs_writable_dirs": f"{first}, {second}"})
    assert _writable(first / "a.txt")
    assert _writable(second / "a.txt")
    assert not _writable(layout / "three" / "a.txt")
