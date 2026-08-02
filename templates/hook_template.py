"""
HOOK TEMPLATE
=============
A hook stands at a doorway in the agent turn and gets a say in what happens.
Reference for authoring one; not imported by the running system.

Read docs/SDK.md for the Request surface and sandbox/guest/hooks.py for the payload
and verdict definitions. This file covers what is specific to hooks.

Before writing: read docs/SDK.md, then this entire template and
templates/service_template.py, because hooks live on services. For details not
defined here, inspect sandbox/guest/hooks.py (moments, payloads, and verdicts),
runtime/hooks.py (kernel traversal), and sandbox/bridge.py (service adapter).
Validate the service file before registering it.

  Where they live:  inside a SERVICE — see templates/service_template.py
  Declared by:      hooks = {moment: method_name}
  Signature:        method(self, sdk, ctx, payload)

Every agent turn is the same short ritual: the turn starts, the model thinks,
the agent acts, think/act repeats, the turn ends. The hook system puts a
labeled doorway at every moment of that ritual, and the one rule is: NOTHING
influences a turn except by standing at a doorway. If nobody is registered,
the kernel walks straight through and behaves exactly as it would with no
hooks at all.


A HOOK IS INBOUND — WHICH IS WHY IT IS DECLARED
------------------------------------------------
Everywhere else in the SDK your code asks and the kernel answers. A hook is
the reverse: the kernel calls you. There is nothing to call at load time, so
there is nothing to register — you declare it, and the kernel reads the
declaration without importing your file:

    class Doorman(BaseService):
        name = "doorman"
        hooks = {"end_turn": "check_done"}

Uninstalling the service removes its hooks with it. A hook cannot leak,
because you never registered it in the first place.

Hooks live in services because something has to be RESIDENT for the kernel to
call into. A tool is gone the moment it answers; a service is still there.


THE SIX MOMENTS (in the order a turn meets them)
------------------------------------------------
  moment            kind       what standing there means
  ----------------  ---------  ------------------------------------------------
  "turn_start"      adjuster   The agent is about to begin. Slip a note into
                               its pocket via sdk.session.add_prompt(...).
  "shape_scope"     adjuster   Here are the tool NAMES the agent will see;
                               return the ones to keep.
  "vet_permission"  verdict    Something sensitive wants to happen; say yes,
                               say no, or stay silent.
  "llm_call"      escort     Own the round trip to the model: rewrite the
                               request, place the call, inspect the answer,
                               go around again if you don't like it.
  "end_turn"        verdict    The doorman at the exit: let the agent leave,
                               send it back with a note, or demand one last
                               tool call first.
  "turn_finish"     observer   The turn is over; look at what happened.


ABSTAIN BY RETURNING None
-------------------------
It is the default and it is what composes. A hook that speaks only when it
must works alongside every other plugin; one that always answers silences
everyone behind it, because the first non-None verdict wins.

A hook that RAISES is logged and skipped — it can never break a turn. So is
one whose service is unloaded, or whose box has died. The worst a hook can do
is fall silent.


KEEP THEM FAST
--------------
Hooks run synchronously on the drive thread, inside every turn they touch.
Each one adds a box round trip (~70µs in-process, ~120µs subprocessed) to the
latency of every reply. That is cheap, but it is not free, and it is paid on
every turn whether or not you end up having an opinion. Do the expensive thing
in a task and read its result here.


WHAT CROSSES, AND WHAT DOES NOT
-------------------------------
A payload is a PROJECTION, not the kernel object encoded. You get what a hook
can act on:

  ctx           session_key, user_id, conversation_id, attended
  TurnEnding    final_text, reason, doorman_fires
  TurnOutcome   ok, cancelled, final_text
  PermissionQuery
                tool_name, command, stage, origin, request, chain
  Scope         tools — a list of NAMES
  ModelRequest  llm (a name), messages, tools, tool_choice, params

Two consequences worth knowing:

  - A scope shaper can HIDE and REORDER, never synthesize. Names you invent
    are ignored. To add a tool, use sdk.session.add_tool(...).
  - request.llm is the backend's NAME, not the backend. Assign another loaded
    backend's name to swap brains for that one call.


THE FIRE BUDGET
---------------
Doormen may intervene at most DOORMAN_FIRE_LIMIT times per turn. Past that
they are no longer consulted and the agent always gets to leave — a stubborn
doorman can never trap a turn. Write doormen to abstain once satisfied (check
ending.doorman_fires or your own state), not to rely on the cap.
"""

from guest.bases import BaseService
from guest.hooks import PermissionVerdict, RequireTool, SendBack


class Doorman(BaseService):
    """One service standing at four doorways.

    A real service usually stands at one. These are together to show the
    shapes side by side.
    """

    name = "doorman"
    description = "Turn policy: gates sensitive calls and enforces an exit check."
    exports = ["report"]

    hooks = {
        "turn_start": "on_start",
        "vet_permission": "gate",
        "end_turn": "check_done",
        "turn_finish": "learn",
    }

    def start(self, sdk):
        """A service still needs a lifecycle; hooks ride on top of it."""
        self._refusals = 0
        return True

    def report(self, sdk):
        """Exported so other plugins can ask what this one has been doing."""
        return {"refusals": self._refusals}

    # ── adjuster ────────────────────────────────────────────────────

    def on_start(self, sdk, ctx, payload):
        """Slip a note into the agent's pocket before it begins.

        The payload is None here — the turn has not produced anything yet.
        What you get is the identity in ctx, and what you do is an effect.
        """
        if ctx.user_id and not ctx.attended:
            sdk.session.add_prompt("Nobody is watching this session. Do not "
                                   "ask questions; make a decision and act.")
        return None

    # ── verdict ─────────────────────────────────────────────────────

    def gate(self, sdk, ctx, query):
        """Say yes, say no, or stay silent.

        Two stages knock here. "approval" means something sensitive wants to
        happen and abstaining falls through to asking the user.
        "unattended_call" means it was asked for with nobody present, and
        abstaining means refuse.
        """
        if query.origin == "request" and query.request.get("type") == "proc.run":
            # A sandboxed Request arrives whole: typed, classified, and
            # carrying the chain of provenance that caused it.
            self._refusals += 1
            return PermissionVerdict(allow=False,
                                     reason="shell commands are off in this profile")

        # Anything this hook has no opinion about, it says nothing about.
        return None

    def check_done(self, sdk, ctx, ending):
        """The doorman at the exit.

        Abstain once satisfied. Relying on the fire budget to stop you is how
        a doorman becomes someone else's bug.
        """
        if ending.doorman_fires >= 1:
            return None                      # said our piece already

        if ending.reason == "budget_exhausted":
            return SendBack("You are out of budget. Summarize what you found.",
                            ephemeral=True, allow_tools=False)

        if not (ending.final_text or "").strip():
            return RequireTool("send_text", note="Say something before you go.")

        return None

    # ── observer ────────────────────────────────────────────────────

    def learn(self, sdk, ctx, outcome):
        """The turn is over. Look, touch nothing.

        Fires once per LOGICAL turn, foreground and background alike. Heavy
        work belongs in a task listening on SESSION_TURN_COMPLETED, not here.
        """
        if not outcome.ok and not outcome.cancelled:
            sdk.log(f"turn failed in {ctx.session_key}", level="warning")
        return None


class Retrier(BaseService):
    """The escort — the one doorway where you decide WHEN the kernel acts."""

    name = "retrier"
    description = "Retries an empty model answer once."
    hooks = {"llm_call": "escort", "shape_scope": "narrow"}

    def start(self, sdk):
        """Nothing to hold."""
        return True

    def escort(self, sdk, ctx, request):
        """Hold both the request and the phone.

        sdk.llm.proceed() places the call. Call it more than once to retry,
        or not at all to answer for yourself — a cached reply never troubles
        the model. Returning None abstains, and the kernel dials for you.
        """
        response = sdk.llm.proceed(request)

        if not (response.content or "").strip() and not response.has_tool_calls:
            # Rewrite and go around again. The change reaches the live call.
            request.messages = request.messages + [
                {"role": "user", "content": "You replied with nothing. Answer."}]
            response = sdk.llm.proceed(request)

        return response

    def narrow(self, sdk, ctx, scope):
        """Hide tools rather than build a toolbox.

        You are handed names and you return names. Anything you invent is
        ignored — narrowing is safe, widening is not, so widening is somebody
        else's Request (sdk.session.add_tool).
        """
        return [name for name in scope.tools if not name.startswith("admin_")]
