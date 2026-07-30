"""Reading numbers off a Request without hiding a bad one.

These coercions used to sit inside handlers' broad ``except Exception``, so a
guest passing ``limit="abc"`` was told the *handler* failed with an int()
error rather than that its argument was wrong.
"""

import pytest

from sandbox.guest.codes import ERROR_INVALID_ARGUMENT
from sandbox.handlers.args import float_arg, int_arg


def test_a_missing_argument_takes_the_default_quietly():
    assert int_arg({}, "limit", 50) == (50, None)
    assert int_arg({"limit": None}, "limit", 50) == (50, None)
    assert int_arg({"limit": ""}, "limit", 50) == (50, None)


def test_a_readable_value_is_used():
    assert int_arg({"limit": 7}, "limit", 50) == (7, None)
    assert int_arg({"limit": "7"}, "limit", 50) == (7, None)
    assert float_arg({"t": "1.5"}, "t", 9.0) == (1.5, None)


def test_a_value_outside_the_range_is_clamped_not_refused():
    """A plugin may ask for more; it does not get to grant itself more."""
    assert int_arg({"limit": 9999}, "limit", 50, lo=1, hi=200) == (200, None)
    assert int_arg({"limit": -5}, "limit", 50, lo=1, hi=200) == (1, None)
    # The default is clamped too, so a careless caller cannot exceed the ceiling.
    assert int_arg({}, "limit", 9999, hi=200) == (200, None)


@pytest.mark.parametrize("bad", ["abc", [1], {"a": 1}, object()])
def test_a_value_that_is_not_a_number_is_a_coded_failure(bad):
    value, failure = int_arg({"limit": bad}, "limit", 50)
    assert value == 50
    assert failure is not None and not failure.ok
    assert failure.code == ERROR_INVALID_ARGUMENT
    assert "limit must be a whole number" in failure.error
    # It is breakage, not policy — a plugin must not catch this as Denied.
    assert failure.denied is False


def test_a_float_argument_reports_its_own_name():
    _, failure = float_arg({"timeout": "soon"}, "timeout", 300.0)
    assert "timeout must be a number" in failure.error


def test_a_bad_argument_blames_the_argument_not_the_handler(tmp_path):
    """End to end: what the guest actually hears.

    Before the coercion moved out of the handler bodies, ``limit="abc"`` was
    reported as ``"list failed: invalid literal for int() with base 10"`` --
    an int() error attributed to the conversation lister.
    """
    from sandbox import Interpreter, run_in_process

    interp = Interpreter()
    try:
        def plugin(sdk, root):
            try:
                sdk.fs.search("x", root=root, limit="abc")
            except sdk.Failed as exc:
                return sdk.ok({"code": exc.result.code,
                               "error": exc.result.error})
            return sdk.ok("no failure")

        out = run_in_process(interp, plugin, name="searcher",
                             kwargs={"root": str(tmp_path)}).data
    finally:
        interp.shutdown()

    assert out["code"] == ERROR_INVALID_ARGUMENT
    assert "limit must be a whole number" in out["error"]
    assert "invalid literal" not in out["error"]
