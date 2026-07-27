"""The two-way translation for frontends, and the desk a token reaches.

A frontend is the only family that acts *for a person*. Everything else in the
sandbox asks the kernel to do something on its own account; a frontend says
"someone typed this" — and the kernel believes it, because believing the
frontend is what a frontend is for. That makes "which frontend is asking?" the
whole security question, and it is answered structurally rather than by policy.

**The desk.** When a frontend's box opens, its adapter is parked here under a
one-shot token and the token is handed into the box. Every ``sdk.frontend.*``
Request carries it back, and the handler reaches *that adapter and no other*.
Code holding no token — a tool, a service, a script — resolves to nothing and
is refused. So a frontend cannot submit on another frontend's behalf, cannot
bind a session it does not own, and cannot outlive its own box: the desk is
cleared at stop, and a leaked token then reaches nothing.

This is the same move ``sandbox/hooks.py`` makes for an escort's ``proceed``,
with one difference worth naming. An escort's phone is parked for the duration
of *one doorway visit*, because that is how long the call it refers to exists.
A frontend's desk lasts as long as its box, because a frontend is not visiting
anything — it is resident, and its authority is its identity rather than a
moment.

**Rendering is a projection.** ``BaseFrontend`` hands its subclasses live
kernel objects; a box can only be handed data. The nine ``render_*`` methods
collapse to one ``render(kind, payload)`` call here, because nine wire methods
for one concept is surface with no payoff — and a guest that only understands
``messages`` is still a working frontend.
"""

from __future__ import annotations

import logging
import threading
import uuid

logger = logging.getLogger("Sandbox")

# token -> the native adapter it stands for.
_DESKS: dict = {}
_DESK_LOCK = threading.Lock()

# The kinds a ``render`` call can carry. Named here because both halves have to
# agree and only this module knows both — the guest documents them, the native
# adapter emits them, and a typo on either side would silently show nothing.
KINDS = ("messages", "attachments", "form_field", "approval", "buttons",
         "error", "typing", "tool_status", "stream_delta")


def park(adapter) -> str:
    """Give a frontend a desk and return the token that reaches it."""
    token = uuid.uuid4().hex
    with _DESK_LOCK:
        _DESKS[token] = adapter
    return token


def unpark(token: str) -> None:
    """Clear the desk. Anything still holding the token now reaches nothing."""
    with _DESK_LOCK:
        _DESKS.pop(token or "", None)


def desk(token: str):
    """The adapter a token stands for, or None if it stands for nothing."""
    with _DESK_LOCK:
        return _DESKS.get(token or "")


# ──────────────────────────────────────────────────────────────────────
# What a box may be shown.
# ──────────────────────────────────────────────────────────────────────

def project_approval(request) -> dict:
    """Reduce an approval request to what a box can be shown and answer.

    A ``StateMachineApprovalRequest`` carries a live ``threading.Event`` the
    state machine waits on, and the pending action it would authorize. Neither
    crosses: the box gets the id and the question, and answers by id through
    ``sdk.frontend.resolve``. Holding the id is enough to answer, and it is
    *only* enough to answer — which is the difference between showing someone
    a decision and handing them the thing being decided.
    """
    return {
        "id": getattr(request, "id", ""),
        "title": getattr(request, "title", ""),
        "body": getattr(request, "body", ""),
        "type": getattr(request, "type", "boolean"),
        "enum": list(getattr(request, "enum", None) or []) or None,
        "default": getattr(request, "default", None),
    }


def project_payload(kind: str, payload):
    """Reduce one render payload to something that can cross.

    Most are already plain — the markdown-on-the-wire convention means a
    message is a string and a form field is a dict. ``approval`` is the
    exception, and the fallback keeps a malformed payload from taking the
    whole render down.
    """
    if kind == "approval":
        return project_approval(payload)
    if isinstance(payload, (str, int, float, bool, type(None), dict)):
        return payload
    if isinstance(payload, (list, tuple)):
        return list(payload)
    logger.debug("render payload for %s was not plain data: %r", kind,
                 type(payload))
    return None
