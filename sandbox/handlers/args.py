"""Reading a number off a Request, and saying so when it is not one.

A guest's arguments are untrusted input like any other. ``int(args.get("limit"))``
raises on ``"abc"``, and where that sits inside a handler's broad ``except`` the
guest is told ``"list failed: invalid literal for int() with base 10: 'abc'"`` --
a message that names the wrong layer and reads like the kernel broke.

So the coercion is separated from the work, and answers with a Result the guest
can act on. Both helpers follow the existing ``_need`` idiom in ``kernel.py``:
the failure comes back beside the value, and the caller returns it.

    limit, bad = int_arg(args, "limit", 50, lo=1, hi=200)
    if bad is not None:
        return bad

Clamping is deliberate and silent -- asking for more than the ceiling gets the
ceiling, which is the same "a plugin may ask, it does not get to grant itself"
rule the timeouts follow. Only a value that is not a number at all is refused.
"""

from __future__ import annotations

from ..guest.codes import ERROR_INVALID_ARGUMENT
from ..guest.requests import Result


def _clamp(value, lo, hi):
    """Bound a number to the range the kernel is willing to serve."""
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def int_arg(args: dict, name: str, default: int,
            *, lo: int | None = None, hi: int | None = None):
    """``(value, None)``, or ``(default, Result)`` when it is not a number.

    An absent or empty argument takes ``default`` without complaint; only a
    value that is present and unreadable is a failure.
    """
    raw = args.get(name)
    if raw is None or raw == "":
        return _clamp(default, lo, hi), None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default, Result.failure(
            f"{name} must be a whole number, not {raw!r}",
            code=ERROR_INVALID_ARGUMENT)
    return _clamp(value, lo, hi), None


def float_arg(args: dict, name: str, default: float,
              *, lo: float | None = None, hi: float | None = None):
    """``(value, None)``, or ``(default, Result)`` when it is not a number."""
    raw = args.get(name)
    if raw is None or raw == "":
        return _clamp(default, lo, hi), None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default, Result.failure(
            f"{name} must be a number, not {raw!r}",
            code=ERROR_INVALID_ARGUMENT)
    return _clamp(value, lo, hi), None
