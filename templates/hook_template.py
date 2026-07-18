"""
HOOK TEMPLATE
=============
This file is a self-contained reference for writing runtime hooks.
It is NOT imported by the running system — it exists for LLM consumption only.

Every agent turn is the same short ritual: the turn starts, the model thinks,
the agent acts, think/act repeats, the turn ends. The hook system
(runtime/hooks.py) puts a labeled doorway at every moment of that ritual, and
the one rule is: NOTHING influences a turn except by standing at a doorway.
If nobody is registered at a doorway, the kernel walks straight through and
behaves exactly as it would with no hooks at all.

Hooks are not their own plugin family. They ride inside an EXTENSION SERVICE
(see templates/service_template.py): a tiny service whose bind_runtime()
registers functions at doorways via runtime.hooks.add(moment, fn), and whose
unload() walks them away via runtime.hooks.remove(fn). Uninstalling the
service removes its hooks with it.


THE SIX MOMENTS (in the order a turn meets them)
------------------------------------------------
  moment            kind       what standing there means
  ----------------  ---------  ------------------------------------------------
  "turn_start"      adjuster   The agent is about to begin; slip a note into
                               its pocket (mutate the session: prompt extras,
                               staged attachments, queued actions).
  "shape_scope"     adjuster   Here is the toolbox the agent will see; hand
                               back a changed one.
  "vet_permission"  verdict    The agent wants to run something sensitive;
                               say yes, say no, or stay silent.
  "model_call"      escort     Own the round trip to the model: rewrite the
                               request, place the call yourself, inspect the
                               answer, go around again if you don't like it.
  "end_turn"        verdict    The doorman at the exit: the agent says "I'm
                               done" — let it leave, send it back with a note,
                               or demand one last tool call first.
  "turn_finish"     observer   The turn is over; look at what happened.
                               Touch nothing.


THREE KINDS OF DOORWAY
----------------------
  OBSERVER — watches, touches nothing. (turn_finish)

  ADJUSTER — handed a thing, may return a changed thing, never sees what
  happens next. A VERDICT is an adjuster whose answer is a decision object;
  the first non-None verdict wins and later hooks are not consulted.
  (turn_start, shape_scope, vet_permission, end_turn)

  ESCORT — signature fn(ctx, payload, proceed). The escort holds both the
  request and the phone: it decides when to dial (proceed), sees the response
  before anyone else, and may redial with a changed request. Escorts nest
  like an onion — the first registered is the outermost wrapper. (model_call)


THE UNIFORM CONTRACT (identical at every doorway)
-------------------------------------------------
  1. Every hook receives (ctx, payload):
       ctx      — HookContext(session, runtime, moment)
       payload  — the moment-specific dataclass (table below)
  2. Return None to abstain; return a value to speak.
  3. A raising hook is logged and skipped — a hook can never break a turn.
     For escorts, "skipped" means transparent: a response already fetched via
     proceed is used (never re-fetched); raising before dialing lets the call
     proceed as if the escort were not there.
  4. Register with runtime.hooks.add(moment, fn); unregister with
     runtime.hooks.remove(fn) at unload — plugin unload must never leak a
     hook.
  5. Hooks run synchronously on the drive thread. Keep them fast — every
     millisecond is added to the reply latency of every turn they touch.


PAYLOADS AND VERDICTS (all defined in runtime/hooks.py)
-------------------------------------------------------
  HookContext(session, runtime, moment)
  ModelRequest(llm, messages, tools, tool_choice, params, attachments)
      — "model_call" payload. Mutate in place and call proceed(request), or
        build a new ModelRequest and proceed(new_request). params are extra
        provider kwargs, forwarded only when non-empty; tool_choice is only
        forwarded when the backend sets supports_tool_choice = True.
  TurnEnding(final_text, reason, doorman_fires)
      — "end_turn" payload. reason is "model_finished" (the model produced
        final text) or "budget_exhausted" (the loop ran out of tool budget).
        doorman_fires counts how often a doorman already intervened this turn.
  TurnOutcome(ok, cancelled, final_text)
      — "turn_finish" payload. Fires once per LOGICAL turn (a Redrive's two
        drives are one turn).
  PermissionQuery(tool_name, command, stage) / PermissionVerdict(allow, reason)
      — "vet_permission" payload and answer. stage says which question is
        being asked: "approval" (a tool wants to run a sensitive command;
        abstaining falls through to skip_permissions and then the human) or
        "unattended_call" (an interactive background_safe=False tool was
        invoked with no human present; abstaining falls through to the
        kernel default: refuse — answer allow to let it through).

  end_turn verdicts:
  Allow()                    — wave the agent through (and stop consulting
                               later doormen).
  SendBack(note, ephemeral=False, allow_tools=True)
      — send the agent back inside; the loop clears its final text and asks
        the model again with your note. ephemeral=True shows the note to the
        model without recording it in history; allow_tools=False makes the
        comeback call text-only.
  RequireTool(name, note="")
      — demand one specific tool before the agent may leave: one more model
        call offering only that tool, forced via tool_choice where the
        backend supports it, degrading to a prompt-level instruction where it
        doesn't. At budget exhaustion this degrades to its note (there is no
        budget left to run a tool with).
  Redrive()
      — end this drive without ending the logical turn; the runtime
        immediately re-drives it. Turn starters do NOT re-run. Prefer a
        model_call escort when all you want is a different brain — an escort
        swaps request.llm per call without restarting anything.

  THE FIRE BUDGET: doormen may intervene at most
  ConversationLoop.DOORMAN_FIRE_LIMIT times per turn. Past that they are no
  longer consulted and the agent always gets to leave — a stubborn doorman
  can never trap a turn. Write doormen to abstain once satisfied (check
  ending.doorman_fires or your own session state), not to rely on the cap.


THE AGENT ACTION QUEUE
----------------------
  session.pending_agent_actions is the agent-side mirror of the mid-turn
  user-message queue: append {"name": tool_name, "args": {...},
  "forced_by": "<your hook label>"} and the ConversationLoop drains it at its
  next loop boundary (never mid tool-call batch), running the tool through
  the same enact/ledger path as a model-chosen call. This is how a
  turn_start hook injects a tool call at the start of a turn. Ephemeral —
  never persisted across restarts.


THE LEDGER STAMP
----------------
  Every doorway-forced act is auditable: queued actions and doorman-forced
  tool calls carry data_json.hook = <your label> in their action_ledger rows,
  and every agent enact records the brain that actually drove it
  (data_json.llm) — after escorts, not before.


Hook authoring flow:
  1. Read this template, then runtime/hooks.py for the payload definitions.
  2. Create an extension service (see templates/service_template.py) in
     sandbox_plugins/services/service_<your_name>.py with lifecycle =
     EXTENSION; register your hooks in bind_runtime(), remove them in
     unload().
  3. Decide the doorway from the table above. Rule of thumb: react to the
     model's answer → model_call escort; enforce something before the turn
     may end → end_turn doorman; everything before the first model call →
     turn_start; visibility of tools → shape_scope; yes/no on a sensitive
     call → vet_permission; learn from finished turns → turn_finish.
  4. Abstain (return None) aggressively. A hook that speaks only when it
     must is a hook that composes with every other plugin.
  5. Test with the contract suite as a model: tests/test_hooks_moments.py
     shows a minimal loop + registry rig for every moment.
"""

# =====================================================================
# EXAMPLE 1: turn_start adjuster — inject memory recall before the drive
# =====================================================================

# from plugins.BaseService import EXTENSION, BaseService
#
#
# class RecallExtension(BaseService):
#     model_name = "Memory Recall"
#     lifecycle = EXTENSION
#
#     def bind_runtime(self, *, runtime=None, **_):
#         self.runtime = runtime
#         if runtime and getattr(runtime, "hooks", None):
#             runtime.hooks.add("turn_start", self._recall)
#
#     def unload(self):
#         if getattr(self, "runtime", None) and getattr(self.runtime, "hooks", None):
#             self.runtime.hooks.remove(self._recall)
#         self.loaded = False
#
#     def _recall(self, ctx, _payload):
#         # The latest user text is already in session.history; write what the
#         # agent should remember into the prompt overlay for this session.
#         latest = next((m["content"] for m in reversed(ctx.session.history)
#                        if m.get("role") == "user"), "")
#         memories = self._search(latest)  # your retrieval, kept fast
#         if memories:
#             ctx.session.system_prompt_extras["recall"] = memories
#         return None  # adjusters that mutate the session just abstain
#
#
# def build_services(config: dict) -> dict:
#     return {"recall_extension": RecallExtension()}


# =====================================================================
# EXAMPLE 2: model_call escort — swap brains, inspect, retry once
# =====================================================================

# class EscalateExtension(BaseService):
#     model_name = "Escalation"
#     lifecycle = EXTENSION
#
#     def bind_runtime(self, *, runtime=None, **_):
#         self.runtime = runtime
#         if runtime and getattr(runtime, "hooks", None):
#             runtime.hooks.add("model_call", self._escort)
#
#     def unload(self):
#         if getattr(self, "runtime", None) and getattr(self.runtime, "hooks", None):
#             self.runtime.hooks.remove(self._escort)
#         self.loaded = False
#
#     def _escort(self, ctx, request, proceed):
#         # Escorts own the round trip: rewrite the request, dial, inspect,
#         # redial. Swapping request.llm here replaces the old restart_turn
#         # dance — no re-drive needed to change brains mid-turn.
#         strong = (ctx.runtime.services or {}).get("llm_strong")
#         if strong is not None and self._should_escalate(ctx.session):
#             request.llm = strong
#         response = proceed(request)
#         if self._looks_lazy(response):
#             from runtime.hooks import ModelRequest
#             return proceed(ModelRequest(
#                 llm=request.llm,
#                 messages=[*request.messages,
#                           {"role": "user", "content": "Be thorough this time."}],
#                 tools=request.tools,
#             ))
#         return response


# =====================================================================
# EXAMPLE 3: end_turn doorman — insist on a report before the exit
# =====================================================================

# from runtime.hooks import RequireTool
#
#
# class NotifyExtension(BaseService):
#     model_name = "Notify Doorman"
#     lifecycle = EXTENSION
#
#     def bind_runtime(self, *, runtime=None, **_):
#         self.runtime = runtime
#         if runtime and getattr(runtime, "hooks", None):
#             runtime.hooks.add("end_turn", self._doorman)
#
#     def unload(self):
#         if getattr(self, "runtime", None) and getattr(self.runtime, "hooks", None):
#             self.runtime.hooks.remove(self._doorman)
#         self.loaded = False
#
#     def _doorman(self, ctx, ending):
#         # Only unattended (background) turns owe a report, and only once:
#         # abstain when attended, already satisfied, or already fired.
#         if ctx.runtime.is_attended(ctx.session.key):
#             return None
#         if ending.doorman_fires > 0 or self._already_reported(ctx.session):
#             return None
#         return RequireTool("tool_notify",
#                            note="Before finishing, file your report with the "
#                                 "tool_notify tool (importance + message).")


# =====================================================================
# EXAMPLE 4: vet_permission verdict — a policy gate
# =====================================================================

# from runtime.hooks import PermissionVerdict
#
#
# def deny_shell_for_guests(ctx, query):
#     """Gates decide, abstain, or defer: None lets the next gate (or the
#     kernel's own approval flow) decide."""
#     if query.tool_name != "run_shell":
#         return None
#     user = ctx.runtime.db.get_user(ctx.runtime.session_user_id(ctx.session.key))
#     if (user or {}).get("user_type") == "guest":
#         return PermissionVerdict(False, "Shell access is disabled for guest accounts.")
#     return None
