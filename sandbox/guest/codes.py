"""Why a Request failed, in a form code can branch on.

A ``Result`` carries a sentence, and a sentence is for a person. Anything that
has to *decide* something needs a token, and until this module existed there
was exactly one, encoded as a prefix on the sentence: ``Result.denied`` asked
whether ``error`` started with the word "denied". That works right up until a
handler reports ``"denied by the remote host"``, at which point a web server's
refusal is indistinguishable from the kernel's and guest code catches
``sdk.Denied`` for it.

So the signal moves onto its own field. The names follow
``state_machine/errors.py``, which had the same problem and solved it the same
way; ``guest/llm.py`` already puts an ``error_code`` on ``LLMResponse`` and
crosses the wire with it.

**An empty code is not a bug.** Most failures are read by a person and never
branched on, and a code invented for one of those is decoration that has to be
kept accurate forever. Add one when a *second* reader appears — that is the
rule that keeps this a vocabulary rather than a rename of 166 call sites.

**``retryable`` stays orthogonal.** A timeout is usually worth retrying and a
bad argument never is, but the flag is set deliberately by whoever knows, not
derived from the code. ``runner_subprocess`` already pairs a timeout with
``retryable=True``; nothing should start inferring one from the other.
"""

from __future__ import annotations

# ── Denials: policy said no. ──────────────────────────────────────────
# These are refusals, not breakage. Everything here is in DENIAL_CODES,
# which is what `Result.denied` and therefore `sdk.Denied` are built on.
ERROR_DENIED = "denied"                        # generic; the default refusal
ERROR_NOT_PERMITTED = "not_permitted"          # outside an allowed path or host
ERROR_APPROVAL_DECLINED = "approval_declined"  # a person said no
ERROR_CANCELLED = "cancelled"                  # the execution was cancelled
ERROR_SHUTTING_DOWN = "shutting_down"          # the sandbox is going away

# ── Breakage: something failed. ───────────────────────────────────────
ERROR_INVALID_ARGUMENT = "invalid_argument"
ERROR_NOT_FOUND = "not_found"
ERROR_UNAVAILABLE = "unavailable"              # a subsystem is absent
ERROR_TIMEOUT = "timeout"
ERROR_HANDLER_ERROR = "handler_error"          # the interpreter's net fired
ERROR_NO_HANDLER = "no_handler"                # a vocabulary entry with no wiring
ERROR_GUEST_FAULT = "guest_fault"              # sandboxed code raised
ERROR_GUEST_EXITED = "guest_exited"            # the child died
ERROR_CONFLICT = "conflict"                    # already exists, or wrong state

#: The codes that mean "refused" rather than "broke". ``Result.denied`` is
#: membership in this set, and ``sdk.Denied`` is raised for exactly these — so
#: adding a name here widens what guest code catches as a policy refusal.
DENIAL_CODES = frozenset({
    ERROR_DENIED,
    ERROR_NOT_PERMITTED,
    ERROR_APPROVAL_DECLINED,
    ERROR_CANCELLED,
    ERROR_SHUTTING_DOWN,
})

#: Every code this module defines. Used by tests to pin the vocabulary closed.
ALL_CODES = frozenset({
    value for name, value in list(globals().items())
    if name.startswith("ERROR_")
})
