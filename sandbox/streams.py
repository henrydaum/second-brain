"""The delta desk — where a backend's streamed text lands.

Third instance of a pattern the sandbox now leans on twice already: an escort's
``proceed`` closure (``hooks._PHONES``) and a frontend's adapter
(``frontends._DESKS``). All three answer the same question the same way.

The question is how to scope a Request that means "do this to the call I am
*already inside*". Classification cannot answer it — there is no property of
the arguments that distinguishes a backend legitimately streaming its own
response from a tool shouting text into someone else's. So the limit is
**reachability, not a verdict**: the kernel parks the sink under an
unguessable token for exactly the duration of one call, hands the token to the
box, and the handler trades it back. Code holding no token reaches no sink and
is refused, which is the right answer rather than a gap.

The sink is one-way. Nothing is returned, so this cannot be used to read
anything, and a chunk costs a frame rather than a round trip — which is what
makes streaming across a process boundary affordable at all.

Note what is *not* here: any way for the backend to learn that the user
cancelled. That is deliberate. Stopping a stream is the kernel's decision, and
it already has a mechanism — cancel the execution and the guest's next Request
raises ``Terminated``. A second, advisory channel saying "you may stop now"
would be a rule that careless code could ignore.
"""

from __future__ import annotations

import logging
import threading
import uuid

logger = logging.getLogger("Sandbox")

_SINKS: dict = {}
_SINK_LOCK = threading.Lock()


def park(sink) -> str:
    """Park one delta sink and return the token that reaches it.

    ``sink`` is any callable taking the fragment of text.
    """
    token = uuid.uuid4().hex
    with _SINK_LOCK:
        _SINKS[token] = sink
    return token


def unpark(token: str) -> None:
    """Discard a sink. Always called, however the call ended."""
    if not token:
        return
    with _SINK_LOCK:
        _SINKS.pop(token, None)


def sink(token: str):
    """The sink a token reaches, or None if it reaches nothing."""
    with _SINK_LOCK:
        return _SINKS.get(token or "")


def deliver(token: str, text: str) -> bool:
    """Hand one fragment to the sink a token names.

    Returns whether it reached anything. A raising sink is logged and
    swallowed: the sink is a rendering path, and a frontend that cannot draw
    a character must not be able to fail the model call that produced it.
    """
    target = sink(token)
    if target is None:
        return False
    try:
        target(text)
    except Exception:
        logger.exception("delta sink raised; continuing")
    return True
