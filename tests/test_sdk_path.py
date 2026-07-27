"""``sdk.path`` — pure path arithmetic, and the reasons it stays pure.

Guest code cannot import ``pathlib`` or ``os.path``; both reach the
environment and the validator refuses them. But joining and splitting a path
is *computation*, and by the SDK's own test (does it touch disk, network,
clock, or process?) that makes it a helper rather than a Request. These pin
the behaviour, and — more importantly — the two things it deliberately will
not do.
"""

import sys

import pytest

from sandbox.guest.sdk import _Path

path = _Path()
WIN = sys.platform == "win32"


def test_join_uses_the_platform_separator():
    assert path.join("a", "b", "c.py") == ("a\\b\\c.py" if WIN else "a/b/c.py")


def test_join_skips_empty_segments():
    """``join(root, "", name)`` is what a caller with an optional part writes."""
    assert path.join("a", "", None, "b") == ("a\\b" if WIN else "a/b")
    assert path.join() == ""


@pytest.mark.parametrize("func, expected", [
    ("parent", "/x/y"), ("name", "z.tar.gz"),
    ("stem", "z.tar"), ("suffix", ".gz"),
])
def test_splitting(func, expected):
    assert getattr(path, func)("/x/y/z.tar.gz") == expected


def test_absolute_resolves_against_the_base_it_is_given():
    assert path.absolute("x.py", base="/base") == (
        "\\base\\x.py" if WIN else "/base/x.py")


def test_absolute_collapses_dotdot_textually():
    """No disk access — which is exactly what keeps this a helper."""
    assert path.absolute("a/../b.py", base="/r") == (
        "\\r\\b.py" if WIN else "/r/b.py")


def test_a_relative_path_without_a_base_stays_relative():
    """The cwd is never consulted.

    Inside a box the working directory is ``sandbox/``, which means nothing to
    the plugin — silently resolving against it would produce a confidently
    wrong absolute path. Callers pass the base they mean.
    """
    assert not path.is_absolute(path.absolute("x/y.py"))


def test_within_is_true_for_the_root_itself():
    assert path.within("/data", "/data")


def test_within_does_not_confuse_a_sibling_prefix():
    """``/data`` must not appear to contain ``/database``.

    A plain ``startswith`` gets this wrong, and every caller of ``within`` is
    asking a containment question that guards something.
    """
    assert path.within("/data/a/b.py", "/data")
    assert not path.within("/database/x.py", "/data")


@pytest.mark.skipif(not WIN, reason="case folding is platform behaviour")
def test_normalize_folds_case_where_the_platform_does():
    assert path.normalize(r"C:\A.PY") == path.normalize(r"c:\a.py")


def test_normalize_does_not_resolve_symlinks(tmp_path):
    """Resolving a link is a disk read, so it would have to be a Request.

    The cost is that two names for one file compare unequal. For the caller
    that matters — read-before-edit tracking — that reads as "not seen yet",
    which is the strict direction and therefore the safe one.
    """
    target = tmp_path / "real.txt"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this system")

    assert path.normalize(link) != path.normalize(target)


def test_the_helper_reaches_nothing(tmp_path):
    """Answers do not depend on what is actually on disk."""
    absent = tmp_path / "nope" / "gone.py"
    assert path.suffix(absent) == ".py"
    assert path.within(absent, tmp_path)
    assert path.name(absent) == "gone.py"


def test_it_is_a_helper_not_a_request():
    """No channel, like ``sdk.text`` and ``sdk.md``.

    If this ever needed the kernel it would have to become a Request family;
    taking no channel is what proves it does not.
    """
    from sandbox.guest.sdk import SDK

    sdk = SDK(None)
    assert sdk.path.join("a", "b")      # works with no channel at all
