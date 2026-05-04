from __future__ import annotations

"""Concrete state-machine actions.

Every action follows the Poker Monster contract (see PokerMonsterRefactor.py):
`is_legal()` checks the current actor/phase, `execute()` mutates state, and
`enact()` combines both into a standardized ActionResult.

The set of actions plays the role of PokerMonster's `Card`/`Action` subclasses:
each one is a typed unit of behavior that the dispatch table in
`action_map.py` routes to based on the current phase. Multi-step flows live
on the phase stack (`cs.cache["phases"]`) — equivalent to PokerMonster's
`gs.cache` — and resolve when the original action is replayed with its
collected inputs.
"""

from typing import Any, Tuple, Optional
import logging

from state_machine.conversationClass import CallableSpec, PhaseFrame
from state_machine.conversation_phases import (
    PHASE_APPROVING_REQUEST,
    PHASE_CALLING_COMMAND,
    PHASE_CALLING_TOOL,
    PHASE_FILLING_COMMAND_FORM,
    PHASE_FILLING_TOOL_FORM,
    PHASE_PARSING_ATTACHMENT,
)
from state_machine.errors import (
    ERROR_ATTACHMENT_NOT_ALLOWED,
    ERROR_EXECUTION_FAILED,
    ERROR_INVALID_ACTION,
    ERROR_INVALID_INPUT,
    ERROR_UNKNOWN_COMMAND,
    ERROR_UNKNOWN_TOOL,
    ERROR_WRONG_ACTOR_TYPE,
    ERROR_WRONG_TURN,
    ActionError,
    ActionResult,
)

logger = logging.getLogger("actionClass")

class Action(object):
    """Base action with shared legality/error handling."""

    action_type = "action"

    def __init__(self, cs, actor_id: str | None = None, content: Any = None):
        self.cs = cs  # Conversation State
        self.actor_id = actor_id or cs.turn_priority
        self.content = content
        self.illegal_code = ERROR_INVALID_ACTION

    def is_legal(self) -> Tuple[bool, Optional[str]]:
        if self.actor_id not in self.cs.participants:
            return False, f"Unknown participant: {self.actor_id}."
        if self.actor_id != self.cs.turn_priority:
            self.illegal_code = ERROR_WRONG_TURN
            return False, "It is not this participant's turn."
        return True, None

    def execute(self) -> ActionResult:
        raise NotImplementedError("Subclass must implement execute()")

    def error(self, code: str, message: str, **details: Any) -> ActionError:
        return ActionError(code, message, details, self.cs.phase)

    def enact(self) -> ActionResult:
        legal, reason = self.is_legal()
        if not legal:
            err = self.error(self.illegal_code, reason or self.illegal_code)
            self.cs.last_error = err
            event = self.cs.event("error", self.actor_id, error=err.to_dict())
            result = ActionResult.fail(self.action_type, err)
            result.events.append(event)
            return result
        try:
            result = self.execute()
        except ActionError as err:
            self.cs.last_error = err
            event = self.cs.event("error", self.actor_id, error=err.to_dict())
            result = ActionResult.fail(self.action_type, err)
            result.events.append(event)
        except Exception as exc:
            logger.debug("Error executing %s for %s: %r", type(self).__name__, self.actor_id, self.content, exc_info=True)
            err = self.error(ERROR_EXECUTION_FAILED, str(exc) or type(exc).__name__)
            self.cs.last_error = err
            event = self.cs.event("error", self.actor_id, error=err.to_dict())
            result = ActionResult.fail(self.action_type, err)
            result.events.append(event)
        return result

class InvalidAction(Action):
    action_type = "invalid"

    def is_legal(self):
        return False, ERROR_INVALID_ACTION
    
    def execute(self):
        raise self.error(ERROR_INVALID_ACTION, "That action is not legal in this phase.", phase=self.cs.phase)


class SendText(Action):
    action_type = "send_text"

    def execute(self):
        text = self.content if isinstance(self.content, str) else (self.content or {}).get("text", "")
        event = self.cs.event("message", self.actor_id, text=text)
        # Self-contained priority hand-off: when a user finishes their turn by
        # sending text in the base phase, the other participant takes priority.
        # Mirrors PokerMonster's pattern of an action managing its own
        # turn_priority transitions instead of pushing that into a runner.
        actor = self.cs.participants.get(self.actor_id)
        if actor and actor.kind == "user":
            from state_machine.conversation_phases import BASE_PHASE
            if self.cs.phase == BASE_PHASE:
                self.cs.switch_priority(self.actor_id)
        return ActionResult(True, self.action_type, events=[event])


class EndTurn(Action):
    action_type = "end_turn"

    def execute(self):
        old = self.cs.turn_priority
        self.cs.reset_phase()
        self.cs.switch_priority(old)
        event = self.cs.event("turn_changed", old, from_actor=old, to_actor=self.cs.turn_priority)
        return ActionResult(True, self.action_type, events=[event])


class Cancel(Action):
    action_type = "cancel"

    def execute(self):
        frame = self.cs.pop_phase()
        event = self.cs.event("cancelled", self.actor_id, cancelled=frame.action_type if frame else None)
        return ActionResult(True, self.action_type, events=[event])


class _CallableAction(Action):
    """Shared flow for commands and direct tool calls.

    A callable can execute immediately, start a form, or suspend into an
    approval phase before being resumed.
    """

    registry = "commands"
    missing_code = ERROR_UNKNOWN_COMMAND
    calling_phase = PHASE_CALLING_COMMAND
    form_phase = PHASE_FILLING_COMMAND_FORM

    def payload(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"name": self.content, "args": {}}
        payload = dict(self.content or {})
        payload.setdefault("args", {})
        return payload

    def is_legal(self):
        legal, reason = super().is_legal()
        if not legal:
            return legal, reason
        if not self.cs.participants[self.actor_id].allows(self.action_type):
            self.illegal_code = ERROR_WRONG_ACTOR_TYPE
            return False, f"{self.cs.participants[self.actor_id].kind} cannot {self.action_type}."
        return True, None

    def spec(self, payload: dict[str, Any]) -> CallableSpec:
        name = payload.get("name")
        spec = self.cs.spec(self.actor_id, self.action_type, name)
        if not name or not spec:
            raise self.error(self.missing_code, f"Unknown {self.action_type.removeprefix('call_')}: {name!r}.", name=name)
        return spec

    def execute(self):
        payload, actor = self.payload(), self.actor_id
        spec = self.spec(payload)
        args = dict(payload.get("args") or {})
        # Missing args turn into a PhaseFrame; subsequent text/callback input
        # fills the frame until the original callable can resume.
        missing = [step for step in spec.form if step.name not in args]
        if missing:
            self.cs.push_phase(PhaseFrame(self.form_phase, self.action_type, actor, spec.name, {"args": args}, missing))
            event = self.cs.event("form_started", actor, name=spec.name, step=missing[0].name, prompt=missing[0].prompt)
            return ActionResult(True, self.action_type, "Input required.", events=[event], data={"step": missing[0].name})
        self._validate(spec, args)
        if spec.require_approval and not payload.get("_approved"):
            return self._approval(payload, spec)
        return self._run(spec, args)

    def _validate(self, spec: CallableSpec, args: dict[str, Any]) -> None:
        for step in spec.form:
            ok, reason = step.validate(args.get(step.name))
            if not ok:
                raise self.error(ERROR_INVALID_INPUT, reason or "Invalid input.", field=step.name)
        if spec.validator:
            ok, reason = spec.validator(args)
            if not ok:
                raise self.error(ERROR_INVALID_INPUT, reason or "Invalid input.")

    def _approval(self, payload: dict[str, Any], spec: CallableSpec):
        # Approval temporarily gives priority to the approver; approving later
        # reconstructs this same callable payload with `_approved=True`.
        approver = spec.approval_actor_id or self.cs.other_id(self.actor_id)
        self.cs.push_phase(PhaseFrame(PHASE_APPROVING_REQUEST, "answer_approval", approver, spec.name, {"pending": {"type": self.action_type, "actor_id": self.actor_id, "content": payload}}))
        self.cs.set_priority(approver)
        event = self.cs.event("approval_requested", self.actor_id, name=spec.name, approver=approver, payload=payload)
        return ActionResult(True, self.action_type, "Approval required.", events=[event])

    def _run(self, spec: CallableSpec, args: dict[str, Any]):
        old_phase = self.cs.phase
        self.cs.phase = self.calling_phase
        try:
            value = spec.handler(self.cs, self.actor_id, args) if spec.handler else None
            if isinstance(value, ActionResult) and not value.ok:
                raise value.error or self.error(ERROR_EXECUTION_FAILED, value.message or "Action failed.")
        finally:
            self.cs.reset_phase()
        event = self.cs.event(self.action_type, self.actor_id, name=spec.name, args=args, result=value, previous_phase=old_phase)
        return ActionResult(True, self.action_type, events=[event], data={"result": value})


class CallCommand(_CallableAction):
    action_type = "call_command"
    registry = "commands"
    missing_code = ERROR_UNKNOWN_COMMAND
    calling_phase = PHASE_CALLING_COMMAND
    form_phase = PHASE_FILLING_COMMAND_FORM


class CallTool(_CallableAction):
    action_type = "call_tool"
    registry = "tools"
    missing_code = ERROR_UNKNOWN_TOOL
    calling_phase = PHASE_CALLING_TOOL
    form_phase = PHASE_FILLING_TOOL_FORM


class SubmitFormText(Action):
    action_type = "submit_form_text"

    def execute(self):
        frame = self.cs.frame
        if not frame or not frame.step:
            raise self.error(ERROR_INVALID_ACTION, "No form is awaiting input.")
        text = self.content if isinstance(self.content, str) else (self.content or {}).get("text")
        ok, reason = frame.step.validate(text)
        if not ok:
            raise self.error(ERROR_INVALID_INPUT, reason or "Invalid input.", field=frame.step.name)
        frame.data.setdefault("args", {})[frame.step.name] = frame.step.coerce(text)
        frame.step_index += 1
        if frame.step:
            event = self.cs.event("form_step", self.actor_id, name=frame.name, step=frame.step.name, prompt=frame.step.prompt)
            return ActionResult(True, self.action_type, "Input required.", events=[event], data={"step": frame.step.name})
        pending = {"name": frame.name, "args": frame.data["args"]}
        actor, action_type = frame.actor_id, frame.action_type
        self.cs.pop_phase()
        from state_machine.action_map import create_action

        # Last form value received: pop the form phase and replay the original
        # command/tool action with its completed args.
        result = create_action(self.cs, action_type, pending, actor).enact()
        if not result.ok and action_type == "call_command":
            frame.step_index = max(0, len(frame.steps) - 1)
            self.cs.push_phase(frame)
            if result.error:
                result.error.retry_phase = self.cs.phase
        return result


class AnswerApproval(Action):
    action_type = "answer_approval"

    def execute(self):
        frame = self.cs.frame
        if not frame or frame.phase != PHASE_APPROVING_REQUEST:
            raise self.error(ERROR_INVALID_ACTION, "No approval is pending.")
        approved = self._approved()
        pending = frame.data["pending"]
        self.cs.pop_phase()
        self.cs.set_priority(pending["actor_id"])
        event = self.cs.event("approval_answered", self.actor_id, approved=approved, pending=pending)
        if not approved:
            self.cs.reset_phase()
            return ActionResult(True, self.action_type, "Denied.", events=[event], data={"approved": False})
        content = dict(pending["content"])
        content["_approved"] = True
        from state_machine.action_map import create_action

        result = create_action(self.cs, pending["type"], content, pending["actor_id"]).enact()
        result.events.insert(0, event)
        return result

    def _approved(self) -> bool:
        if isinstance(self.content, bool):
            return self.content
        if isinstance(self.content, dict) and "approved" in self.content:
            return bool(self.content["approved"])
        text = str(self.content.get("text") if isinstance(self.content, dict) else self.content).strip().lower()
        if text in {"y", "yes", "approve", "approved", "true"}:
            return True
        if text in {"n", "no", "deny", "denied", "false", "cancel"}:
            return False
        raise self.error(ERROR_INVALID_INPUT, "Approval needs yes or no.")


class SendAttachment(Action):
    action_type = "send_attachment"

    def is_legal(self):
        legal, reason = super().is_legal()
        if not legal:
            return legal, reason
        if not self.cs.participants[self.actor_id].allows(self.action_type):
            self.illegal_code = ERROR_WRONG_ACTOR_TYPE
            return False, f"{self.cs.participants[self.actor_id].kind} cannot send attachments."
        return True, None

    def execute(self):
        content = dict(self.content or {})
        ext = self.cs.attachment_extension(content)
        if self.cs.allowed_attachment_extensions and ext not in self.cs.allowed_attachment_extensions:
            raise self.error(ERROR_ATTACHMENT_NOT_ALLOWED, f".{ext} attachments are not allowed for this model.", extension=ext)
        self.cs.phase = PHASE_PARSING_ATTACHMENT
        parsed = self.cs.attachment_parser(content) if self.cs.attachment_parser else content
        self.cs.reset_phase()
        event = self.cs.event("attachment", self.actor_id, attachment=content, parsed=parsed)
        return ActionResult(True, self.action_type, events=[event], data={"parsed": parsed})
