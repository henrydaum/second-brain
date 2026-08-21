"""When a plugin's system-prompt contribution goes stale — the one table.

A plugin contributes to the agent system prompt through ``agent_prompt``, and
the prompt is rebuilt on every **LLM call**, not once per turn. For the method
shape that means a real call into the plugin's box — a module import at
minimum, a subprocess spawn for anything foreign — so the answer has to be
cached, and the cache has to know when the world moved.

This module is that knowledge, and it is a *declaration* rather than an
inference. A plugin names the cue its text follows::

    agent_prompt_refresh = "session"

    def agent_prompt(self, sdk):
        mode = (sdk.session.get() or {}).get("mode")
        return "..." if mode == "lockdown" else ""

What it replaces is one global counter that ticked whenever sandboxed code
changed anything. That counter was right for what it could know — nothing
declared what a prompt method read, so the dependency had to be inferred, and
over-invalidating costs one box call while under-invalidating is silently
wrong. But it made a prompt that only follows the permission mode recompute on
every ``fs.write`` in the process, to re-read something that changes once a
conversation.

The ladder
----------

One total order, least to most frequent. Each rung adds a component to the
cache key, so a plugin at rank L is keyed on the tuple of every component at
rank ``<= L``. That is what makes the ladder a *threshold* rather than a set of
unrelated events: a rarer change invalidates finer-grained plugins, never the
reverse. A ``config`` write therefore refreshes a ``turn``-cued prompt, and a
turn boundary does not refresh a ``config``-cued one.

``never``    the static-string shape. Derived, never declared — see DECLARABLE.
``load``     the loaded plugin population changed. A plugin's *own* reload
             needs no counter, because discovery builds a fresh adapter class
             and instantiates it, so the cache goes with the old object; what
             the counter is for is the other reading, and the one an author
             means by ``load`` — *which plugins exist*. A package install
             writes its files from kernel code and never passes the write
             counter's funnel at all, so without this rung a prompt describing
             what is installed was invalidated only by coincidence.
``config``   a counter, fired once from ``config_manager._emit_config_changed``.
``session``  **not an event.** It is a fact — see SESSION_FACTS — read straight
             off the ``PromptContext`` being built. No fire site can be
             forgotten because there is none, which is the property worth
             protecting: a turn-scoped ``yolo`` cleared by writing the session
             field directly is covered anyway, because the next context simply
             reports a different mode.
``turn``     a counter, fired once from ``HookRegistry.start_turn``.
``write``    the counter this module grew out of, fired from
             ``Interpreter._settle`` for any successful Request that changed
             something. **The default.**
``call``     never cached. Its stamp never compares equal to itself.

The default is safe by *superset*, not by identity
--------------------------------------------------

Before cues, a method's key was ``(epoch, session_key, security_mode)``. A
``write``-cued key is every rung, so the default now also moves on a turn
boundary, a config save, an install, and three more session facts. Every one of
those is a strictly wider invalidation than before, so nothing can go stale
that did not before — but it does cost one extra recompute per turn for a
plugin nothing wrote for. That is the price of the ladder being a ladder: a
rung that did not invalidate the rungs below it would make this a set of
unrelated triggers with no basis for ordering them. A plugin that feels the
cost has a one-word answer available to it: declare the rung it actually
follows.

Placement
---------

``STABLE_THROUGH`` is the line between the two blocks the prompt is assembled
from. Cues at or below it do not move within a conversation, so their text
rides in the semi-stable block of the position-0 ``system`` message; the rest
ride in the dynamic ``[SYSTEM CONTEXT UPDATE]`` block. Within each block
``agent/system_prompt.py`` sorts by rank, rarest first.

That sort pays in the prefix and is cosmetic in the dynamic block, which is
rebuilt per call either way — worth knowing before anyone optimizes it further.

The tier is enforced rather than promised: a cue below ``session`` is answered
with the plain kernel context, so ``sdk.session.get()`` tells it nothing. Note
what this does *not* claim. The position-0 message is already per-session — its
tool catalog is profile-scoped and its command catalog is filtered by the
session's frontend — so the prefix was never session-independent. What it is
is *stable for the life of one session*, and the tier is what keeps a plugin
from being the reason it stops being.
"""

from __future__ import annotations

import threading

from sandbox.guest.requests import (CONSOLE_WRITE, HTTP_CLOSE, HTTP_PUSH,
                                    HTTP_RESPOND, LLM_DELTA, READ_ONLY,
                                    UI_PROGRESS, UI_RENDER)

NEVER = "never"
LOAD = "load"
CONFIG = "config"
SESSION = "session"
TURN = "turn"
WRITE = "write"
CALL = "call"

#: The ladder itself, least to most frequent. Order *is* the meaning: rank
#: decides what a stamp contains, which block the text lands in, and where
#: within that block it sits.
LADDER = (NEVER, LOAD, CONFIG, SESSION, TURN, WRITE, CALL)
RANK = {cue: position for position, cue in enumerate(LADDER)}

#: What a plugin may actually write. ``never`` is derived from the *shape* — a
#: plain string is settled at load — so declaring it says nothing a method
#: could honour. Worse, a method declaring ``never`` is exactly the permanent
#: cache the write counter was written to kill: a tool that goes on describing
#: a directory as it stood when its adapter was built. Making it unspellable is
#: cheaper than making it an error somebody has to read.
DECLARABLE = LADDER[1:]

#: What an undeclared method gets. ``write`` is the widest rung short of never
#: caching, so a plugin that says nothing cannot be stale — and a prompt
#: listing a directory or a database's tables, which is what the method shape
#: mostly exists for, is wrong at any rarer cue and wrong invisibly.
DEFAULT = WRITE

#: Cues at or below this rank ride in the cacheable position-0 message.
STABLE_THROUGH = CONFIG

#: The session facts a ``session``-cued contribution is keyed on, and the
#: reason this is a named tuple of strings rather than four inlined getattrs:
#: what a sandboxed prompt can *see* of its session is whatever
#: ``handlers/kernel._session_get`` answers with, so these two have to agree or
#: a prompt reads something nothing invalidates. ``tests/test_prompt_cues.py``
#: walks that handler's keys against this set and TRANSIENT below.
#:
#: ``user_id`` earns its place the least obviously and matters the most:
#: ``bind_session_user`` rebinds a live session to another account, and a
#: prompt naming the user would otherwise still be naming the last one.
#: Written as the map so the agreement is checkable rather than asserted: the
#: key is what the handler answers with, the value is what ``PromptContext``
#: calls the same thing.
SESSION_GET_KEYS = {
    "key": "session_key",
    "conversation_id": "conversation_id",
    "user_id": "user_id",
    "agent_profile": "profile_name",
    "frontend": "frontend_name",
    "mode": "security_mode",
}
SESSION_FACTS = tuple(SESSION_GET_KEYS.values())

#: Session keys deliberately *not* facts. Each is transient inside the very
#: turn the prompt is being built for — ``busy`` is always true at prompt-build
#: time — so keying on them would move the stamp mid-turn for no reader.
#: ``debug`` is a live services dump answered only when asked for, and
#: ``service_flags`` is what it holds — the walk that checks this set against
#: the handler flattens nested dicts, which is the conservative direction: a
#: key it cannot place is one somebody has to decide about.
TRANSIENT = ("phase", "busy", "attended", "debug", "service_flags")

#: Showing the agent's output to a person. Writes, all of them, and none of
#: them in ``READ_ONLY`` — but **rendering is not a change**: they move text to
#: a screen and produce no state any system prompt could read back.
#:
#: Excluding them is load-bearing rather than tidy, because the volume is
#: per-token. A streaming backend sends one ``llm.delta`` per token, and the
#: frontend rendering that stream sends one ``console.write`` per token right
#: behind it (``bundled/frontends/frontend_repl.py``). Counting either would
#: tick thousands of times per reply, so every ``write``-cued ``agent_prompt``
#: would recompute on every model call and the caching would be undone
#: entirely — with no symptom beyond being slow, which is why the set is named
#: and pinned rather than left to the reading of ``READ_ONLY``.
#:
#: ``llm.delta`` alone was the first version of this and was not enough: the
#: two halves of one stream arrive as different Request types, and excluding
#: the backend's half while counting the frontend's fixed nothing.
#:
#: The ``http.*`` three are that same second half for a frontend whose screen
#: is somewhere else: an SSE ``http.push`` per token sits behind ``llm.delta``
#: exactly where ``console.write`` sits for the REPL. ``respond`` and ``close``
#: join them not by volume but by the same argument the set is named for —
#: they finish moving text to a person and produce no state a prompt could
#: read back. Only ``http.drain`` is left out, and it is a read.
#:
#: ``ui.progress`` joins on both counts: a command narrating a loop emits one
#: per iteration, and what it produces is a line on a status display.
RENDERING = {LLM_DELTA, CONSOLE_WRITE, UI_RENDER, UI_PROGRESS,
             HTTP_RESPOND, HTTP_PUSH, HTTP_CLOSE}

#: Requests that do not tick the ``write`` counter: reads, which change nothing
#: by definition, plus the rendering family above.
UNCOUNTED = READ_ONLY | RENDERING

#: The cues that are events rather than facts. ``never`` is a shape and
#: ``session`` is read off the context; neither needs a number, and neither
#: needs a call site that can be forgotten. ``call`` needs the opposite of a
#: number.
COUNTED = (LOAD, CONFIG, TURN, WRITE)

_lock = threading.Lock()
_counters = {cue: 0 for cue in COUNTED}
_serial = 0


def fire(cue: str) -> None:
    """Record that this cue's event happened."""
    with _lock:
        _counters[cue] += 1


def value(cue: str) -> int:
    """This cue's counter. Compare against a stamp taken earlier."""
    with _lock:
        return _counters[cue]


def counts(request, result) -> bool:
    """Whether this serviced Request should tick the ``write`` counter.

    A refusal ran nothing, so it changed nothing — and under ``lockdown`` every
    denial would otherwise force a pointless recompute of every write-cued
    prompt. Reading ``result.ok`` rather than the decision keeps a *failed*
    effect counted as no change, which is the common case (a write to a path
    that does not exist) and wrong only for a partial write — worth one stale
    prompt against a bump on every denied Request.
    """
    return (getattr(request, "type", "") not in UNCOUNTED
            and bool(getattr(result, "ok", False)))


def rank(cue: str) -> int:
    """The declared cue's rank, or the default's for anything unrecognised.

    Falling back rather than raising is the same call :data:`DEFAULT` makes: an
    unknown cue is caught by the validator at load, with the line and a
    suggestion, so by the time a prompt is being assembled the honest response
    to a name nobody knows is the conservative one.
    """
    return RANK.get(cue, RANK[DEFAULT])


def of(plugin) -> str:
    """The cue a plugin's ``agent_prompt`` follows.

    The *shape* decides first and cannot be overridden: a non-callable
    ``agent_prompt`` is ``never`` whatever it declared. That is not tidiness —
    a plugin declaring a string and a cue would otherwise park a fixed sentence
    in the volatile block forever, recomputed on every rung it named and
    identical every time. The validator refuses the combination at authoring
    time; this is the half that cannot be skipped by a file nobody validated.
    """
    if not callable(getattr(plugin, "agent_prompt", "")):
        return NEVER
    declared = getattr(plugin, "agent_prompt_refresh", None)
    if not isinstance(declared, str) or declared not in DECLARABLE:
        return DEFAULT
    return declared


def stable(cue: str) -> bool:
    """Whether this cue's text belongs in the cacheable position-0 message."""
    return rank(cue) <= RANK[STABLE_THROUGH]


def session_for(cue: str, ctx=None) -> str:
    """The session key to answer this contribution with — "" below ``session``.

    The enforcement half of the tier. A contribution that cannot see the
    session cannot accidentally depend on it, so what rides in the shared
    prefix stays true for as long as that prefix does. "" reaches
    ``Interpreter.context_for_session``, which answers the plain kernel
    context, so ``sdk.session.get()`` tells such a plugin nothing.
    """
    if rank(cue) < RANK[SESSION]:
        return ""
    return str(getattr(ctx, "session_key", "") or "")


def _next_serial() -> int:
    """A number nobody has seen before, so ``call`` is never cached.

    Expressing "always recompute" as a value rather than a branch keeps
    ``_cached_prompt`` to one comparison — the cue decides what a stamp holds,
    and nothing downstream has to know which cue it was handed.

    A counter rather than a sentinel object that compares unequal to
    everything, because such an object has to refuse ``__hash__`` to stay
    honest, and that would make one rung's stamp the only unhashable one. The
    obvious next use of a stamp is as a dict key — a cache holding an entry per
    live session instead of a single slot — and it would work for six rungs and
    raise ``TypeError`` on the seventh.
    """
    global _serial
    with _lock:
        _serial += 1
        return _serial


def stamp(cue: str, ctx=None) -> tuple:
    """The cache key for a contribution that refreshes on ``cue``.

    Exactly the components at or below the cue's rank, in ladder order. The cue
    itself leads, so a key says what it is keyed on — which costs one string
    and saves the reader of a failing test from working it out.

    ``ctx`` is the ``PromptContext`` being built. Its :data:`SESSION_FACTS`
    *are* the ``session`` component: read rather than counted, so a mode
    change, a conversation switch, a profile change or a user rebind is noticed
    with nothing anywhere having to announce it.
    """
    at = rank(cue)
    if at >= RANK[CALL]:
        return (cue, _next_serial())
    parts = [cue]
    if at >= RANK[LOAD]:
        parts.append(value(LOAD))
    if at >= RANK[CONFIG]:
        parts.append(value(CONFIG))
    if at >= RANK[SESSION]:
        parts.append(tuple(getattr(ctx, name, None) for name in SESSION_FACTS))
    if at >= RANK[TURN]:
        parts.append(value(TURN))
    if at >= RANK[WRITE]:
        parts.append(value(WRITE))
    return tuple(parts)
