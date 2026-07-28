"""Who is asking, carried across a Request that re-enters the sandbox.

``Chain`` is the call stack, and ``policy.classify`` leans on it twice: once
for depth, once for cycles. Both were dead, and the reason is that a chain only
ever got one link. ``Chain.push`` was called at the outermost run and nowhere
else, so a tool reached through ``tool.call`` started a *fresh* chain rooted at
whatever caused the outer call — and the callee's provenance had no memory of
its caller at all.

That mattered in three places. ``tool.call`` and ``service.call`` are
classified SAFE on the stated grounds that "the callee's own Requests are
classified with the caller still in the chain, so routing through a tool
launders nothing", which was simply not true. The cycle detector could never
fire, so a tool reaching itself recursed until a pool ran dry. And the approval
dialog — built entirely around showing a person the chain, because
"service_web wants to make an HTTP request" is unanswerable — only ever had one
link to show.

**Why an ambient value rather than a parameter.** The re-entry happens inside a
*handler*, and a handler's signature is ``(ctx, args)``: there are around a
hundred of them and only three care. Threading a chain through all of them to
serve three would be the wrong trade. What is true instead is that the whole
nested call is synchronous on one thread — ``_execute`` calls the handler, the
handler calls into the registry, the registry reaches the adapter — so the
thread *is* the call stack, and a context variable is the honest way to say so.

**Why the token dance is not optional.** ``ThreadPoolExecutor`` reuses its
threads and does not reset their context between tasks, unlike asyncio. A value
set and not reset would still be there when an unrelated Request landed on the
same worker, and it would be believed. Every set is paired with a reset in a
``finally``; :func:`serving` is the only way in, so there is one place to get
that right.

The value carries the host context alongside the chain for the same reason and
by the same route: a service called from a session should answer from *that*
session's world, and a service acting on its own should answer from the
kernel's. Both are "who is asking", one link apart.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

#: The Request being serviced on this thread, or None outside a handler.
_CURRENT: ContextVar = ContextVar("sandbox_provenance", default=None)


@dataclass(frozen=True)
class Caller:
    """The execution whose Request is being serviced on this thread.

    ``chain`` is its provenance and ``context`` the host object its own
    Requests are answered from. A callee adopts both: the chain so the callee
    appears *below* its caller rather than beside it, and the context so a
    service invoked from someone's session reads that person's rows rather
    than the kernel's.
    """
    chain: object
    context: object = None


@contextmanager
def serving(chain, context=None):
    """Mark this thread as servicing one execution's Request.

    Reset in a ``finally`` without exception: a pool worker that kept the value
    would hand it to the next Request that happened to land on it.
    """
    token = _CURRENT.set(Caller(chain=chain, context=context))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current() -> Caller | None:
    """Who is asking, or None if this thread is not inside a handler."""
    return _CURRENT.get()
