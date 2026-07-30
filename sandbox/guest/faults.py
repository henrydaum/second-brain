"""What broke, in the guest's own terms.

A plugin author asks "where in *my* code did this happen", and the frames above
their first line answer somebody else's question: the worker thread in
``sandbox/runner.py``, the serve loop in ``child.py``, the per-call thread in
``sandbox/boxes.py``. A stack opening with three frames of machinery reads like
a kernel bug, so those are dropped and what is left is the guest's own.

The formatted *string* crosses the boundary, never an exception object. That is
CPython's ``multiprocessing.pool`` trick, and it is the only option here anyway:
the wire is JSON, and unpickling an exception would execute code — precisely
what ``protocol.py`` refuses to do.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

#: How much of a stack crosses. Generous enough for a deep call with source
#: lines, small enough that it can neither approach ``protocol.MAX_MESSAGE_BYTES``
#: nor crowd out the model's tool-result budget.
MAX_TRACEBACK_CHARS = 4000

_GUEST_DIR = str(Path(__file__).resolve().parent)
_FOLD_CASE = sys.platform.startswith("win")


def _key(path: str) -> str:
    """A comparable form of a frame's filename.

    ``co_filename`` and ``__file__`` are not guaranteed to agree on case or on
    symlinks, and on Windows they routinely disagree on both. ``os.path.normcase``
    is the obvious reach and is exactly what the guest boundary forbids, so
    resolve instead — affordable, because this runs only after something broke.
    """
    if not path:
        return ""
    try:
        text = str(Path(path).resolve())
    except (OSError, ValueError):
        text = path
    return text.lower() if _FOLD_CASE else text


def clamp(text: str, limit: int = MAX_TRACEBACK_CHARS) -> str:
    """Cap a traceback by its middle.

    Both ends carry signal — the entry point and the raise site — so a tail trim
    throws away the half saying *what* and a head trim the half saying *where
    from*.
    """
    if not text or len(text) <= limit:
        return text
    head = limit // 2
    return (f"{text[:head]}\n...[{len(text) - limit} characters elided]...\n"
            f"{text[-(limit - head):]}")


def guest_traceback(exc: BaseException, *, drop: tuple = (),
                    limit: int = MAX_TRACEBACK_CHARS) -> str:
    """Format ``exc``'s stack as the guest's own frames.

    ``drop`` names host files whose frames are machinery — ``runner.py`` passes
    its own ``__file__``, ``boxes.py`` passes its own. Named by the caller
    because the guest may not import the host to find out.

    Everything *below* the first guest frame is kept, libraries included: a
    plugin whose HTTP client raised wants to see the client.
    """
    tb = getattr(exc, "__traceback__", None)
    if tb is None:
        # Constructed and never raised — ``child.py`` reports one for an
        # unexpected message kind. ``format_exc()`` answers "NoneType: None"
        # here, or worse a stale unrelated stack, and saying nothing beats both.
        return ""

    unwanted = {_key(p) for p in drop}
    guest = _key(_GUEST_DIR)
    frames = traceback.extract_tb(tb)
    kept = [f for f in frames
            if _key(f.filename) not in unwanted
            and not _key(f.filename).startswith(guest)]
    # Nothing survived: the fault was raised entirely inside the boundary — the
    # loader refusing a file, the protocol refusing a message — so the
    # machinery's stack is the only one there is, and it is the right one.
    return clamp("".join([
        "Traceback (most recent call last):\n",
        *traceback.format_list(kept or frames),
        *traceback.format_exception_only(type(exc), exc),
    ]), limit)
