"""Per-conversation security mode — the standing answer to an approval dialog.

Three values, and they are exactly the three answers a person can give to the
question "may this run?" before being asked it:

===========  ==========================================================
``lockdown`` No. Refuse anything that would have raised a dialog.
``ask``      Ask me. Today's behaviour, and the default.
``yolo``     Yes. Approve anything that would have raised a dialog.
===========  ==========================================================

**A mode is a standing answer, not a new layer.** It is read at the one point
in :func:`sandbox.approval.build_approver` where the dialog would otherwise be
drawn, which is what gives each value its precise scope: ``yolo`` frees only
work that would have interrupted you, and ``lockdown`` refuses only what would
have reached you. Neither touches what the kernel had already settled without
asking — a plugin's own ``vet_permission`` verdict, or a service reading the
credential it was configured with. Lockdown means "stop asking me, the answer
is no"; it does not mean "break the plugins I already set up".

Two things it deliberately cannot do, both worth saying out loud because a
grant that overstates itself erodes trust in the dialog as fast as one that
understates it erodes safety:

- **Yolo is not root.** Every structural refusal stands — a cross-user read,
  DDL naming a kernel table, ``MAX_DEPTH``, the cycle detector, a desk or
  one-shot token, an unclassified Request. Those never produced a dialog, so
  there is no answer for a mode to stand in for.
- **Yolo never reaches unattended work.** The approver refuses a chain nobody
  is watching *before* it consults the mode, so a cron job or a subagent
  cannot spend a grant you gave for a foreground task.

This module is kernel rather than ``sandbox/`` because two layers read it: the
sandbox approver, and the state machine's command grant
(``state_machine/action.py``). It holds the vocabulary and nothing else — where
the value is *stored* is :class:`runtime.session.RuntimeSession`, and who may
change it is ``sandbox.policy.classify``.
"""

from __future__ import annotations

from typing import Any

LOCKDOWN = "lockdown"
ASK = "ask"
YOLO = "yolo"

#: Ordered loosest-last, which is the order every picker and table shows.
SECURITY_MODES = (LOCKDOWN, ASK, YOLO)
DEFAULT_SECURITY_MODE = ASK

#: What each mode is, in one line, for a person. The single source for the
#: ``/mode`` table and the ``session.set_mode`` dialog.
MODE_BLURBS = {
    LOCKDOWN: "Refuse anything that needs your approval, without asking.",
    ASK: "Ask you about anything that needs your approval. The default.",
    YOLO: "Approve anything that needs your approval, without asking.",
}

#: Scopes a mode can be set for. ``conversation`` lasts until the conversation
#: changes; ``turn`` is dropped when the agent turn ends. The second exists
#: because "yes, and stop asking for the rest of this" is a grant scoped to
#: *time* rather than to a destination, which the three standing grant lists
#: could never express.
CONVERSATION_SCOPE = "conversation"
TURN_SCOPE = "turn"
SCOPES = (CONVERSATION_SCOPE, TURN_SCOPE)


def security_mode(value: Any, default: str = DEFAULT_SECURITY_MODE) -> str:
    """Normalize an arbitrary input to one of :data:`SECURITY_MODES`.

    Never raises and never answers something outside the tuple: an unreadable
    mode has to degrade to the *default* rather than to the loosest value, and
    a normalizer that could raise would take down the approver that calls it.
    """
    raw = str(value or default).strip().lower()
    mode = {"lock": LOCKDOWN, "locked": LOCKDOWN, "deny": LOCKDOWN,
            "default": ASK, "normal": ASK, "manual": ASK, "prompt": ASK,
            "auto": YOLO, "bypass": YOLO, "allow": YOLO}.get(raw, raw)
    return mode if mode in SECURITY_MODES else default


def scope_name(value: Any, default: str = CONVERSATION_SCOPE) -> str:
    """Normalize a scope the same way, for the same reason."""
    raw = str(value or default).strip().lower()
    return raw if raw in SCOPES else default


def standing_answer(mode: Any) -> bool | None:
    """What this mode answers on the person's behalf, or ``None`` to ask them.

    The whole of the mode's authority, in one function. ``None`` is not a
    failure branch — it is ``ask``, the default, and it is what keeps this
    feature invisible until somebody turns it on.
    """
    mode = security_mode(mode)
    if mode == YOLO:
        return True
    if mode == LOCKDOWN:
        return False
    return None


def tightens(mode: Any) -> bool:
    """Whether moving *to* this mode can only ever narrow what may happen.

    ``lockdown`` is the tightest value, so arriving at it widens nothing
    whatever the current mode is — which is why an agent may set it without
    being asked. Every other value could widen, and none of them can be
    decided without knowing where we are now, so they are all treated as
    widening. ``sandbox.policy`` reads this as its polarity rule.
    """
    return security_mode(mode) == LOCKDOWN


def prompt_note(mode: Any) -> str:
    """The line the agent is shown about the mode it is working under.

    Empty for ``ask``, because that is the default and a prompt that restates
    the default on every turn is tokens spent saying nothing. Told to the
    model rather than left implicit so that a refusal is *legible*: an agent
    that does not know it is in lockdown reads a denial as a transient failure
    and retries it, which is the worst available reading.
    """
    mode = security_mode(mode)
    if mode == LOCKDOWN:
        return (
            "## Lockdown\n"
            "This conversation is in lockdown. Anything that would need the "
            "user's approval is refused outright, and asking again will not "
            "change that — do not retry a refused action or look for another "
            "route to the same effect. Capabilities the policy classifies as "
            "safe, including internally mediated work, remain available. If "
            "you need something that is refused, say so plainly and tell the "
            "user that `/mode ask` lifts the standing refusal."
        )
    if mode == YOLO:
        return (
            "## YOLO mode\n"
            "This conversation is in YOLO mode: the user has pre-approved "
            "anything that would normally raise an approval dialog, so you "
            "will not be interrupted. Structural refusals and capability "
            "limits still apply. Nobody is checking each action — weigh "
            "consequential and irreversible steps yourself, and say what you "
            "are about to do before you do it."
        )
    return ""
