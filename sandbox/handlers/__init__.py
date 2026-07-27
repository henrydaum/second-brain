"""Effect implementations — the only code that actually touches the world.

Each handler takes ``(ctx, args)`` and returns a :class:`Result`. Handlers run
on the interpreter's execution pool, never on the gate thread, so a slow
handler (a 30-second HTTP call) cannot delay classification for anyone else.

Handlers are the *bottom* of the system: sandboxed code cannot reach them
except through a classified Request.

``ctx`` is the kernel's own context object — in Second Brain, the
``SecondBrainContext`` that used to be handed *to* plugins. It does not
disappear under this design; it moves to the other side of the boundary. The
guest gets an ``sdk`` that can only ask, and the context is what answers. It is
duck-typed throughout so the sandbox stays independent of the kernel, and it
must never cross into the guest.

Split by what they need:

- :mod:`.fs_net` — stdlib only: paths, URLs, argv
- :mod:`.kernel` — everything reaching into ``ctx``

A Request with no handler is not an error in the catalogue; it means that
capability is not wired yet, and the caller gets an ordinary failure saying
so. The explicit ``UNWIRED`` inventory keeps those gaps visible.
"""

from ..guest.requests import ALL_TYPES, SELF_RESPOND
from . import fs_net, kernel

HANDLERS = {}
HANDLERS.update(fs_net.HANDLERS)
HANDLERS.update(kernel.HANDLERS)

# Requests in the catalogue that nothing services yet. Named rather than
# silently missing, so "not built" is distinguishable from "misspelled".
UNWIRED = sorted(ALL_TYPES - set(HANDLERS) - {SELF_RESPOND})

__all__ = ["HANDLERS", "UNWIRED"]
