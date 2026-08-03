"""Who is asking, carried across a Request that re-enters the sandbox.

The design argument — why the chain has to be a stack, why it is ambient
rather than a parameter, and why the token reset is load-bearing — lives in
CLAUDE.md under "The chain only became a stack once it could survive
re-entry". One home for it; this module is the mechanism.

The value carries the host context alongside the chain because both answer
"who is asking", one link apart: a service called from a session reads that
person's rows, a service acting on its own reads the kernel's.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: The Request being serviced on this thread, or None outside a handler.
_CURRENT: ContextVar = ContextVar("sandbox_provenance", default=None)

#: What survives into a filename. Windows rejects ``:`` outright, and a session
#: key is full of them.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_LABEL = 40
_FALLBACK = "sb-box"


@dataclass(frozen=True)
class Caller:
    """The execution whose Request is being serviced on this thread.

    ``chain`` is its provenance and ``context`` the host object its own
    Requests are answered from. A callee adopts both: the chain so the callee
    appears *below* its caller rather than beside it, and the context so a
    service invoked from someone's session reads that person's rows rather
    than the kernel's.

    ``execution`` is the caller itself, and it is here for one reason:
    cancellation only reaches code that is *making* Requests. A handler that
    starts nested work and then blocks waiting for it — ``script.run`` is the
    case — is not making Requests while it waits, so cancelling the caller sets
    a flag nobody reads and the nested work runs to its own ceiling. Handlers
    that block on something cancellable poll :meth:`abandoned` and tear it
    down.
    """
    chain: object
    context: object = None
    execution: object = None

    @property
    def abandoned(self) -> bool:
        """Whether the caller has been cancelled and stopped wanting an answer.

        False when there is no execution to ask, so a handler written against
        this reads as "carry on" wherever provenance is not being tracked —
        which is every test that calls a handler directly.
        """
        return bool(getattr(self.execution, "cancelled", False))


@contextmanager
def serving(chain, context=None, execution=None):
    """Mark this thread as servicing one execution's Request.

    Reset in a ``finally`` without exception: a pool worker that kept the value
    would hand it to the next Request that happened to land on it.
    """
    token = _CURRENT.set(
        Caller(chain=chain, context=context, execution=execution))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current() -> Caller | None:
    """Who is asking, or None if this thread is not inside a handler."""
    return _CURRENT.get()


def scratch_prefix() -> str:
    """A ``tempfile`` prefix naming whoever is asking for the scratch.

    Everything in ``workspace/temp`` used to be called ``sb-box-<random>``,
    which threw away the one fact that makes the folder readable. Opening it
    and finding a directory of unexplained diagrams is the whole problem: the
    kernel knew it was ``extract_container`` unpacking an archive, and named
    the evidence after nobody.

    The innermost link, not the root. The root is what *caused* the work —
    ``agent``, ``user``, a session key — which is the right answer for an
    approval dialog and the wrong one here, where three different tasks
    triggered by the same turn would all come out identical. ``links[-1]`` is
    the thing that actually asked.

    Safe by construction rather than by trust: the label is a chain link, and
    a link can be a session key (``telegram:7912761600:...``), whose colons
    Windows rejects outright. So it is filtered to a filename alphabet and
    capped, and anything left empty falls back — a caller that cannot be named
    must still get its scratch.
    """
    caller = current()
    links = getattr(getattr(caller, "chain", None), "links", ()) or ()
    label = _UNSAFE.sub("_", str(links[-1]) if links else "").strip("._-")
    return f"{label[:_MAX_LABEL] or _FALLBACK}-"
