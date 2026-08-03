"""A counter that ticks whenever sandboxed code changes something.

The one question it answers: *has anything changed since I last computed this?*

``_cached_prompt`` in ``bridge.py`` is the only reader today. A plugin's
``agent_prompt`` may be a method reading live state — the installed store's
``tool_run_script`` lists the scripts directory, ``tool_sql_query`` the database's
tables — and the system prompt is rebuilt on *every* LLM call, not once per turn.
Recomputing per call costs a fresh box per ephemeral plugin (a module import at
minimum, a subprocess spawn for anything foreign); never recomputing is what the
cache used to do, and it left the agent reading a listing that predated the file
it had just written.

An epoch buys both: a read-only stretch — read, search, think, call the model
again, which is most of a turn — costs **zero** recomputes, and a single
``fs.write`` costs exactly one.

``Interpreter._settle`` is the single funnel every serviced Request passes
through, which is what makes one counter sufficient. The bump lives there beside
the ledger sink and answers a near-identical question, so the two draw the same
line in the same vocabulary.

Deliberately global rather than per-family. Scoping by ``Request.family`` would
spare a database-reading plugin a recompute after an ``fs.write``, but nothing
declares *which* family a prompt method reads, so the dependency would have to be
inferred — and over-invalidating costs one box call while under-invalidating is
silently wrong. The coarse counter is the safe direction.
"""

from __future__ import annotations

import threading

from .guest.requests import LLM_DELTA, READ_ONLY

#: Requests that do not tick the counter.
#:
#: Reads, by definition — they change nothing. Plus ``llm.delta``, which is a
#: *write* and is not in ``READ_ONLY``, but which a streaming backend sends once
#: per token: thousands of bumps per reply would invalidate every cached prompt
#: on every call and quietly undo the whole point of caching. The ledger's
#: sandbox sink excludes it from recording for the same shape of reason and
#: draws the line the same way (``runtime/ledger.py``).
UNCOUNTED = READ_ONLY | {LLM_DELTA}

_lock = threading.Lock()
_epoch = 0


def bump() -> None:
    """Record that something changed."""
    global _epoch
    with _lock:
        _epoch += 1


def value() -> int:
    """The current epoch. Compare against a stamp taken earlier."""
    with _lock:
        return _epoch


def counts(request, result) -> bool:
    """Whether this serviced Request should tick the counter.

    A refusal ran nothing, so it changed nothing — and under ``lockdown`` every
    denial would otherwise force a pointless recompute of every live prompt.
    Reading ``result.ok`` rather than the decision keeps a *failed* effect
    counted as no change, which is the common case (a write to a path that does
    not exist) and wrong only for a partial write — worth one stale prompt
    against a bump on every denied Request.
    """
    return (getattr(request, "type", "") not in UNCOUNTED
            and bool(getattr(result, "ok", False)))
