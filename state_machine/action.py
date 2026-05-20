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

from __future__ import annotations

from typing import Any, Tuple, Optional
import logging

from state_machine.conversation import CallableSpec, FormStep, PhaseFrame
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


def _steps(spec: CallableSpec, args: dict[str, Any], cs=None) -> list[FormStep]:
    """Build the current form steps for a callable spec."""
    if not spec.form_factory:
        return spec.form
    return spec.form_factory(args, cs)


def _missing(spec: CallableSpec, args: dict[str, Any], cs=None) -> list[FormStep]:
    """Return the required form steps that are still missing."""
    return [s for s in _steps(spec, args, cs) if s.name not in args and (s.required or s.prompt_when_missing)]


def _emit_command_event(channel: str, cs, payload: dict[str, Any]) -> None:
    """Emit a slash-command lifecycle event when the event bus is available."""
    try:
        from events.event_bus import bus
        bus.emit(channel, {"session_key": cs.cache.get("session_key"), **payload})
    except Exception:
        pass


def _emit_command_progress(cs, frame) -> None:
    """Emit progress updates for an in-flight slash-command form."""
    call_id = (frame.data or {}).get("call_id")
    if frame.action_type == "call_command" and call_id:
        from events.event_channels import COMMAND_CALL_PROGRESSED
        _emit_command_event(COMMAND_CALL_PROGRESSED, cs, {"call_id": call_id, "command_name": frame.name, "args": dict((frame.data or {}).get("args") or {})})


def _record_form_field(frame) -> None:
    """Record the fields that have already been visited in a form."""
    frame.data.setdefault("form_history", []).append(frame.step.name)


def _rewind_form(cs, frame):
    """Move a form back one field and return the new active step."""
    history = frame.data.setdefault("form_history", [])
    if not history:
        return None
    args = frame.data.setdefault("args", {})
    args.pop(history.pop(), None)
    spec = cs.spec(frame.actor_id, frame.action_type, frame.name)
    missing = _missing(spec, args, cs) if spec else []
    frame.steps, frame.step_index = missing, 0
    _emit_command_progress(cs, frame)
    return missing[0] if missing else None


class Action(object):
    """Base action with shared legality/error handling."""

    action_type = "action"

    def __init__(self, cs, actor_id: str | None = None, content: Any = None):
        """Initialize the action."""
        self.cs = cs  # Conversation State
        self.actor_id = actor_id or cs.turn_priority
        self.content = content
        self.illegal_code = ERROR_INVALID_ACTION

    def is_legal(self) -> Tuple[bool, Optional[str]]:
        """Return whether this action is legal for the current actor and turn."""
        if self.actor_id not in self.cs.participants:
            return False, f"Unknown participant: {self.actor_id}."
        if self.actor_id != self.cs.turn_priority:
            self.illegal_code = ERROR_WRONG_TURN
            return False, "It is not this participant's turn."
        return True, None

    def execute(self) -> ActionResult:
        """Apply the action to the conversation state."""
        raise NotImplementedError("Subclass must implement execute()")

    def error(self, code: str, message: str, **details: Any) -> ActionError:
        """Build an ActionError anchored to the current phase."""
        return ActionError(code, message, details, self.cs.phase)

    def enact(self) -> ActionResult:
        """Run legality checks, execute the action, and normalize failures."""
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
    """Invalid action."""
    action_type = "invalid"

    def is_legal(self):
        """Always reject this placeholder action."""
        return False, ERROR_INVALID_ACTION
    
    def execute(self):
        """Raise the standardized invalid-action error."""
        raise self.error(ERROR_INVALID_ACTION, "That action is not legal in this phase.", phase=self.cs.phase)


class SendText(Action):
    """Send text."""
    action_type = "send_text"

    def execute(self):
        """Append a text message event and hand turn priority to the other side when needed."""
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
    """End turn."""
    action_type = "end_turn"

    def execute(self):
        """Reset phase state and hand turn priority to the other participant."""
        old = self.cs.turn_priority
        self.cs.reset_phase()
        self.cs.switch_priority(old)
        event = self.cs.event("turn_changed", old, from_actor=old, to_actor=self.cs.turn_priority)
        return ActionResult(True, self.action_type, events=[event])


class Cancel(Action):
    """Cancel."""
    action_type = "cancel"

    def execute(self):
        """Pop the active frame, restore priority, and emit cancellation events."""
        frame = self.cs.pop_phase()
        if frame is not None and frame.phase == PHASE_APPROVING_REQUEST:
            data = frame.data or {}
            pending = data.get("pending") or {}
            self.cs.set_priority(pending.get("actor_id") or data.get("previous_priority") or self.cs.other_id(self.actor_id) or self.actor_id)
        # If we were mid-form for a slash command, emit FINISHED so any UI
        # showing a pending hourglass can resolve it as cancelled.
        if frame is not None and frame.action_type == "call_command":
            call_id = (frame.data or {}).get("call_id")
            if call_id:
                try:
                    from events.event_bus import bus
                    from events.event_channels import COMMAND_CALL_FINISHED
                    bus.emit(COMMAND_CALL_FINISHED, {
                        "session_key": self.cs.cache.get("session_key"),
                        "call_id": call_id,
                        "command_name": frame.name,
                        "ok": False,
                        "error": "cancelled",
                    })
                except Exception:
                    pass
        event = self.cs.event("cancelled", self.actor_id, cancelled=frame.action_type if frame else None)
        return ActionResult(True, self.action_type, "Cancelled.", events=[event])


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
        """Normalize command/tool input into a name-plus-args payload."""
        if isinstance(self.content, str):
            return {"name": self.content, "args": {}}
        payload = dict(self.content or {})
        payload.setdefault("args", {})
        return payload

    def is_legal(self):
        """Return whether the current participant is allowed to call this callable."""
        legal, reason = super().is_legal()
        if not legal:
            return legal, reason
        if not self.cs.participants[self.actor_id].allows(self.action_type):
            self.illegal_code = ERROR_WRONG_ACTOR_TYPE
            return False, f"{self.cs.participants[self.actor_id].kind} cannot {self.action_type}."
        return True, None

    def spec(self, payload: dict[str, Any]) -> CallableSpec:
        """Resolve the callable spec for the requested command or tool."""
        name = payload.get("name")
        spec = self.cs.spec(self.actor_id, self.action_type, name)
        if not name or not spec:
            if self.action_type == "call_tool" and name in set(self.cs.cache.get("agent_scoped_tool_names") or []):
                raise self.error(self.missing_code, f"Tool not in agent scope: {name!r}.", name=name)
            raise self.error(self.missing_code, f"Unknown {self.action_type.removeprefix('call_')}: {name!r}.", name=name)
        return spec

    def execute(self):
        """Run the callable immediately or suspend into form/approval flow first."""
        payload, actor = self.payload(), self.actor_id
        spec = self.spec(payload)
        args = dict(payload.get("args") or {})
        raw_arg = "arg" in args
        resumed_call_id = payload.get("_call_id")
        if not resumed_call_id:
            self._supersede_pending_form()
        # Missing args turn into a PhaseFrame; subsequent text/callback input
        # fills the frame until the original callable can resume.
        missing = [] if raw_arg else _missing(spec, args, self.cs)
        if missing:
            # First invocation: emit STARTED so the UI can show a pending
            # indicator while the user fills the form. The call_id is pinned
            # to the phase frame so the matching FINISHED fires from the
            # eventual _run with the same id.
            call_id = resumed_call_id or self._emit_invocation_started(spec, args)
            frame_data = {"args": args}
            if call_id:
                frame_data["call_id"] = call_id
            self.cs.push_phase(PhaseFrame(self.form_phase, self.action_type, actor, spec.name, frame_data, missing))
            event = self.cs.event("form_started", actor, name=spec.name, step=missing[0].name, prompt=missing[0].prompt)
            return ActionResult(True, self.action_type, events=[event], data={"step": missing[0].name, "call_id": call_id})
        self._validate(spec, args)
        if spec.require_approval and not payload.get("_approved"):
            return self._approval(payload, spec)
        return self._run(spec, args, call_id=resumed_call_id)

    def _validate(self, spec: CallableSpec, args: dict[str, Any]) -> None:
        """Internal helper to validate collected args against the callable form."""
        if "arg" in args:
            return
        for step in _steps(spec, args, self.cs):
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
        """Internal helper to suspend a callable behind an approval frame."""
        approver = spec.approval_actor_id or self.cs.other_id(self.actor_id)
        self.cs.push_phase(PhaseFrame(PHASE_APPROVING_REQUEST, "answer_approval", approver, spec.name, {
            "type": "boolean",
            "title": spec.name,
            "prompt": f"Approve {spec.name}?",
            "required": True,
            "pending": {"type": self.action_type, "actor_id": self.actor_id, "content": payload},
        }))
        self.cs.set_priority(approver)
        event = self.cs.event("approval_requested", self.actor_id, name=spec.name, approver=approver, payload=payload)
        return ActionResult(True, self.action_type, "Approval required.", events=[event])

    def _run(self, spec: CallableSpec, args: dict[str, Any], *, call_id: str | None = None):
        """Internal helper to invoke the callable and translate its result into events."""
        old_phase = self.cs.phase
        self.cs.phase = self.calling_phase
        started = call_id or self._emit_invocation_started(spec, args)
        try:
            value = spec.handler(self.cs, self.actor_id, args) if spec.handler else None
            if isinstance(value, ActionResult) and not value.ok:
                raise value.error or self.error(ERROR_EXECUTION_FAILED, value.message or "Action failed.")
        except Exception as e:
            self._emit_command_finished(started, spec, False, str(e))
            raise
        finally:
            self.cs.reset_phase()
        self._emit_command_finished(started, spec, True, None)
        event = self.cs.event(self.action_type, self.actor_id, name=spec.name, args=args, previous_phase=old_phase)
        return ActionResult(True, self.action_type, events=[event], data={"result": value, "call_id": started})

    def _emit_invocation_started(self, spec: CallableSpec, args: dict[str, Any]):
        """Internal helper to announce the start of a slash-command invocation."""
        if self.action_type != "call_command":
            return None
        try:
            import uuid
            from events.event_bus import bus
            from events.event_channels import COMMAND_CALL_STARTED
            call_id = f"cmd:{spec.name}:{uuid.uuid4().hex[:8]}"
            bus.emit(COMMAND_CALL_STARTED, {"session_key": self.cs.cache.get("session_key"), "call_id": call_id, "command_name": spec.name, "args": args})
            return call_id
        except Exception:
            return None

    def _supersede_pending_form(self):
        """Internal helper to cancel any older pending form before starting a new one."""
        frame = self.cs.peek_phase() if hasattr(self.cs, "peek_phase") else self.cs.frame
        if not frame or frame.phase not in {PHASE_FILLING_COMMAND_FORM, PHASE_FILLING_TOOL_FORM}:
            return
        call_id = (frame.data or {}).get("call_id")
        if call_id:
            from events.event_channels import COMMAND_CALL_FINISHED
            _emit_command_event(COMMAND_CALL_FINISHED, self.cs, {"call_id": call_id, "command_name": frame.name, "ok": False, "error": "superseded"})
        self.cs.pop_phase()

    def _emit_command_finished(self, call_id, spec: CallableSpec, ok: bool, error: str | None):
        """Internal helper to announce the final status of a slash-command invocation."""
        if not call_id or self.action_type != "call_command":
            return
        try:
            from events.event_bus import bus
            from events.event_channels import COMMAND_CALL_FINISHED
            bus.emit(COMMAND_CALL_FINISHED, {"session_key": self.cs.cache.get("session_key"), "call_id": call_id, "command_name": spec.name, "ok": ok, "error": error})
        except Exception:
            pass


class CallCommand(_CallableAction):
    """Call command."""
    action_type = "call_command"
    registry = "commands"
    missing_code = ERROR_UNKNOWN_COMMAND
    calling_phase = PHASE_CALLING_COMMAND
    form_phase = PHASE_FILLING_COMMAND_FORM


class CallTool(_CallableAction):
    """Call tool."""
    action_type = "call_tool"
    registry = "tools"
    missing_code = ERROR_UNKNOWN_TOOL
    calling_phase = PHASE_CALLING_TOOL
    form_phase = PHASE_FILLING_TOOL_FORM


class SubmitFormText(Action):
    """Submit form text."""
    action_type = "submit_form_text"

    def execute(self):
        """Record one piece of form input and either advance or resume the callable."""
        frame = self.cs.frame
        if not frame or not frame.step:
            raise self.error(ERROR_INVALID_ACTION, "No form is awaiting input.")
        text = self.content if isinstance(self.content, str) else (self.content or {}).get("text")
        try:
            value = frame.step.coerce(text)
        except Exception as e:
            raise self.error(ERROR_INVALID_INPUT, f"{frame.step.name} must be {frame.step.type}: {e}", field=frame.step.name)
        if frame.step.validator:
            ok, reason = frame.step.validator(value)
            if not ok:
                raise self.error(ERROR_INVALID_INPUT, reason or "Invalid input.", field=frame.step.name)
        frame.data.setdefault("args", {})[frame.step.name] = value
        _record_form_field(frame)
        _emit_command_progress(self.cs, frame)
        frame.step_index += 1
        spec = self.cs.spec(frame.actor_id, frame.action_type, frame.name)
        missing = _missing(spec, frame.data["args"], self.cs) if spec else []
        if missing:
            frame.steps, frame.step_index = missing, 0
            event = self.cs.event("form_step", self.actor_id, name=frame.name, step=missing[0].name, prompt=missing[0].prompt)
            return ActionResult(True, self.action_type, events=[event], data={"step": frame.step.name})
        pending = {"name": frame.name, "args": frame.data["args"]}
        if frame.data.get("call_id"):
            pending["_call_id"] = frame.data["call_id"]
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
    """Resolve a pending typed-input request.

    Despite the historical name, this carries any typed value (string,
    integer, number, boolean, array, object, enum). For a frame whose
    `data["type"]` is "boolean" with a `pending` action, a truthy value
    re-enacts the gated action with `_approved=True`. For other types, the
    value is simply returned in the result data for the caller to consume.
    """

    action_type = "answer_approval"

    def execute(self):
        """Resolve the active approval frame and resume the pending callable when allowed."""
        frame = self.cs.frame
        if not frame or frame.phase != PHASE_APPROVING_REQUEST:
            raise self.error(ERROR_INVALID_ACTION, "No request is pending.")
        if isinstance(self.content, dict):
            got, expected = self.content.get("request_id"), (frame.data or {}).get("request_id")
            if got and expected and got != expected:
                raise self.error(ERROR_INVALID_INPUT, "That request is no longer active.", request_id=got)
        value = self._coerce(frame)
        pending = frame.data.get("pending")
        original_actor = (pending or {}).get("actor_id") or self.cs.other_id(self.actor_id) or self.actor_id
        self.cs.pop_phase()
        self.cs.set_priority(original_actor)
        event = self.cs.event("approval_answered", self.actor_id, value=value, approved=bool(value), pending=pending)

        if pending and frame.data.get("type", "boolean") == "boolean":
            if not value:
                self.cs.reset_phase()
                return ActionResult(True, self.action_type, "Denied.", events=[event], data={"approved": False, "value": False})
            content = dict(pending["content"])
            content["_approved"] = True
            from state_machine.action_map import create_action

            result = create_action(self.cs, pending["type"], content, pending["actor_id"]).enact()
            result.events.insert(0, event)
            return result

        # Free-form typed input: just return the value.
        self.cs.reset_phase()
        return ActionResult(True, self.action_type, "Received.", events=[event], data={"value": value, "approved": bool(value)})

    def _coerce(self, frame) -> Any:
        """Coerce raw content into the requested type using FormStep semantics."""
        type_ = frame.data.get("type", "boolean")
        enum = frame.data.get("enum")
        default = frame.data.get("default")
        required = frame.data.get("required", True)
        raw = self.content
        if isinstance(raw, dict):
            if "value" in raw:
                raw = raw["value"]
            elif "text" in raw:
                raw = raw["text"]

        # Booleans carry the historical lenient text parser ("yes", "y", etc.).
        if type_ == "boolean":
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in {"y", "yes", "approve", "approved", "true", "1"}:
                return True
            if text in {"n", "no", "deny", "denied", "false", "0", "cancel"}:
                return False
            raise self.error(ERROR_INVALID_INPUT, "Approval needs yes or no.")

        step = FormStep(name=frame.name or "input", required=required, type=type_, enum=enum, default=default)
        ok, reason = step.validate(raw)
        if not ok:
            raise self.error(ERROR_INVALID_INPUT, reason or "Invalid input.")
        return step.coerce(raw)


class SkipForm(Action):
    """Skip an optional form field by accepting its default and advancing.

    Mirrors `SubmitFormText`'s replay path: pop the form when complete and
    re-enact the original command/tool action with collected args.
    """

    action_type = "skip_form"

    def execute(self):
        """Skip the current optional form field and continue the callable flow."""
        frame = self.cs.frame
        if not frame or not frame.step:
            raise self.error(ERROR_INVALID_ACTION, "No form is awaiting input.")
        if frame.step.required:
            raise self.error(ERROR_INVALID_INPUT, "Cannot skip a required field.", field=frame.step.name)
        frame.data.setdefault("args", {})[frame.step.name] = frame.step.default
        _record_form_field(frame)
        _emit_command_progress(self.cs, frame)
        frame.step_index += 1
        spec = self.cs.spec(frame.actor_id, frame.action_type, frame.name)
        missing = _missing(spec, frame.data["args"], self.cs) if spec else []
        if missing:
            frame.steps, frame.step_index = missing, 0
            event = self.cs.event("form_step", self.actor_id, name=frame.name, step=missing[0].name, prompt=missing[0].prompt)
            return ActionResult(True, self.action_type, "Skipped.", events=[event], data={"step": frame.step.name})
        pending = {"name": frame.name, "args": frame.data["args"]}
        if frame.data.get("call_id"):
            pending["_call_id"] = frame.data["call_id"]
        actor, action_type = frame.actor_id, frame.action_type
        self.cs.pop_phase()
        from state_machine.action_map import create_action

        result = create_action(self.cs, action_type, pending, actor).enact()
        if not result.ok and action_type == "call_command":
            frame.step_index = max(0, len(frame.steps) - 1)
            self.cs.push_phase(frame)
            if result.error:
                result.error.retry_phase = self.cs.phase
        if result.ok:
            result.message = result.message or "Skipped."
        return result


class BackForm(Action):
    """Back form."""
    action_type = "back_form"

    def execute(self):
        """Move the active form back one collected field."""
        frame = self.cs.frame
        if not frame or not frame.step:
            raise self.error(ERROR_INVALID_ACTION, "No form is awaiting input.")
        step = _rewind_form(self.cs, frame)
        if step is None:
            raise self.error(ERROR_INVALID_ACTION, "Nothing to go back to.")
        event = self.cs.event("form_step", self.actor_id, name=frame.name, step=step.name, prompt=step.prompt)
        return ActionResult(True, self.action_type, "Back.", events=[event], data={"step": step.name})


class SendAttachment(Action):
    """Send attachment."""
    action_type = "send_attachment"

    def is_legal(self):
        """Return whether the current actor may submit an attachment in this phase."""
        legal, reason = super().is_legal()
        if not legal:
            return legal, reason
        if not self.cs.participants[self.actor_id].allows(self.action_type):
            self.illegal_code = ERROR_WRONG_ACTOR_TYPE
            return False, f"{self.cs.participants[self.actor_id].kind} cannot send attachments."
        return True, None

    def execute(self):
        """Queue an attachment for parsing and hand the turn to the agent."""
        content = dict(self.content or {})
        ext = self.cs.attachment_extension(content)
        if self.cs.allowed_attachment_extensions and ext not in self.cs.allowed_attachment_extensions:
            raise self.error(ERROR_ATTACHMENT_NOT_ALLOWED, f".{ext} attachments are not allowed for this model.", extension=ext)
        self.cs.phase = PHASE_PARSING_ATTACHMENT
        try:
            parsed = self.cs.attachment_parser(content) if self.cs.attachment_parser else content
        finally:
            self.cs.reset_phase()
        # If the parser produced an Attachment dataclass, queue it so the
        # next agent turn can hand it to the LLM service.
        attachment = (parsed or {}).get("attachment") if isinstance(parsed, dict) else None
        if attachment is not None:
            self.cs.pending_attachments.append(attachment)
        actor = self.cs.participants.get(self.actor_id)
        if actor and actor.kind == "user":
            self.cs.switch_priority(self.actor_id)
        event = self.cs.event("attachment", self.actor_id, attachment=content, parsed=parsed)
        return ActionResult(True, self.action_type, events=[event], data={"parsed": parsed})


# ──────────────────────────────────────────────────────────────────────
# Canvas actions
#
# First-class actions for canvas controls: skill execution, palette swap,
# slider/control mutation, regenerate, layer delete/move, size, reset.
# Each action mutates ``cs.canvas`` (the Canvas dataclass) and emits
# CANVAS_ACTION_STARTED/FINISHED so frontends can render pending state
# the same way they do for slash commands.
#
# Rendering itself (subprocess sandbox, PIL, PNG write) is delegated to
# ``plugins.tools.helpers.canvas_render`` so the state machine stays
# free of frontend / PIL / plugin imports.
# ──────────────────────────────────────────────────────────────────────


def _emit_canvas_event(channel: str, cs, payload: dict[str, Any]) -> None:
    try:
        from events.event_bus import bus
        bus.emit(channel, {"session_key": cs.cache.get("session_key"), **payload})
    except Exception:
        pass


class _CanvasAction(Action):
    """Shared lifecycle for canvas-mutating actions.

    Builds a *draft* ``Canvas`` (deep copy of ``cs.canvas``), passes it
    to ``_run(draft)``, and only assigns ``cs.canvas = draft`` on success.
    The live canvas is never observable mid-mutation; failures leave it
    untouched. Emits ``CANVAS_ACTION_STARTED`` / ``CANVAS_ACTION_FINISHED``
    around the call.

    Deps (skill_loader, db, config) come from ``cs.cache['canvas_deps']``,
    which the runtime populates at session construction — actions never
    reach into the runtime directly.
    """

    action_type = "canvas_action"

    def is_legal(self):
        # Canvas actions are loose about turn priority — a user click on
        # regenerate while the agent is mid-turn is rejected at the
        # runtime busy guard, not here. We still require a known actor.
        if self.actor_id not in self.cs.participants:
            return False, f"Unknown participant: {self.actor_id}."
        return True, None

    def _arg(self, key: str, default: Any = None) -> Any:
        if isinstance(self.content, dict):
            return self.content.get(key, default)
        return default

    def _session_key(self) -> str:
        return self.cs.cache.get("session_key") or "default"

    def _deps(self) -> Any:
        deps = self.cs.cache.get("canvas_deps")
        if deps is None:
            raise RuntimeError("canvas_deps missing from cs.cache; runtime did not wire them.")
        return deps

    def execute(self):
        import uuid
        from events.event_channels import CANVAS_ACTION_STARTED, CANVAS_ACTION_FINISHED
        from canvas.canvas import Canvas

        call_id = f"canvas:{self.action_type}:{uuid.uuid4().hex[:8]}"
        draft = Canvas.from_dict(self.cs.canvas.to_dict())
        _emit_canvas_event(CANVAS_ACTION_STARTED, self.cs, {
            "call_id": call_id,
            "action_name": self.action_type,
            "args": self._summary_args(),
        })
        try:
            data = self._run(draft) or {}
        except Exception as e:
            _emit_canvas_event(CANVAS_ACTION_FINISHED, self.cs, {
                "call_id": call_id,
                "action_name": self.action_type,
                "ok": False,
                "error": str(e),
            })
            raise self.error(ERROR_EXECUTION_FAILED, str(e) or type(e).__name__)
        # Atomic swap: only on success does the live canvas change.
        self.cs.canvas = draft
        _emit_canvas_event(CANVAS_ACTION_FINISHED, self.cs, {
            "call_id": call_id,
            "action_name": self.action_type,
            "ok": True,
        })
        event = self.cs.event(self.action_type, self.actor_id, **{k: v for k, v in (data.get("event_data") or {}).items()})
        return ActionResult(True, self.action_type, events=[event], data={"call_id": call_id, **{k: v for k, v in data.items() if k != "event_data"}})

    def _summary_args(self) -> dict[str, Any]:
        """Short, frontend-friendly arg summary for the STARTED event."""
        return dict(self.content) if isinstance(self.content, dict) else {}

    def _run(self, draft) -> dict[str, Any]:
        raise NotImplementedError


class RunSkill(_CanvasAction):
    """Execute a skill (creation or transform) and append/replace the chain.

    Builds the new chain entry with ``seed=None`` so ``replay_chain`` can
    sample from the seed pool — the second invocation of the same skill
    with the same params/palette/size becomes a cache hit instead of a
    fresh subprocess render.
    """

    action_type = "run_skill"

    def _run(self, draft) -> dict[str, Any]:
        from plugins.skills.helpers.skill_runner import (
            default_controls, make_chain_entry,
        )
        from plugins.tools.helpers import canvas_render
        from canvas.canvas import MAX_CHAIN_LENGTH

        slug = str(self._arg("slug") or "")
        params = dict(self._arg("params") or {})
        if not slug:
            raise ValueError("slug is required")
        deps = self._deps()
        skill = deps.skill_loader(slug)
        if not skill:
            raise ValueError(f"No skill named '{slug}'.")
        if skill.kind == "transform" and not draft.image_path:
            raise ValueError("Transform skills require a current canvas image.")
        if skill.kind == "transform" and len(draft.last_chain) >= MAX_CHAIN_LENGTH:
            raise ValueError(f"Chain is at the {MAX_CHAIN_LENGTH}-layer cap — delete a layer first.")

        controls = default_controls(skill)
        if any(c.get("type") == "palette" for c in (skill.controls or [])):
            controls["palette"] = draft.palette_id
        controls.update({k: v for k, v in params.items() if k in controls})
        entry = make_chain_entry(skill, params, 0, controls=controls)
        # seed=None → pool sample; explicit int → use as-is.
        entry["seed"] = int(params["seed"]) if params.get("seed") is not None else None
        draft.push_chain_entry(entry)
        snap = canvas_render.render_chain(
            self._session_key(), list(draft.last_chain),
            palette_id=draft.palette_id, size=int(draft.size),
            op=f"skill:{slug}", out_name=f"_skill_{slug}.png",
            canvas=draft, skill_loader=deps.skill_loader,
        )
        return {"canvas": snap, "event_data": {"slug": slug, "kind": skill.kind}}


class SetPalette(_CanvasAction):
    """Switch the session palette and replay the chain."""

    action_type = "set_canvas_palette"

    def _run(self, draft) -> dict[str, Any]:
        from plugins.helpers.palettes import palette_exists
        from plugins.tools.helpers import canvas_render, layered_canvas as lc

        palette_id = str(self._arg("palette_id") or "")
        if not palette_id or not palette_exists(palette_id):
            raise ValueError(f"unknown palette: {palette_id!r}")
        if palette_id == draft.palette_id:
            return {"canvas": lc.to_frontend_shape(draft), "event_data": {"palette_id": palette_id, "noop": True}}
        draft.apply_palette(palette_id)
        snap = lc.to_frontend_shape(draft)
        if draft.last_chain:
            snap = canvas_render.render_chain(
                self._session_key(), list(draft.last_chain),
                palette_id=palette_id, size=int(draft.size),
                op=f"palette:{palette_id}", out_name="_palette_replay.png",
                canvas=draft, skill_loader=self._deps().skill_loader,
            )
        return {"canvas": snap, "event_data": {"palette_id": palette_id}}


class SetSkillControl(_CanvasAction):
    """Update one control on a chain entry, then replay the chain."""

    action_type = "set_skill_control"

    def _run(self, draft) -> dict[str, Any]:
        import random
        from plugins.tools.helpers import canvas_render
        from plugins.skills.helpers.skill_controls import coerce_control_value

        chain_index = int(self._arg("chain_index") or 0)
        name = str(self._arg("name") or "")
        value = self._arg("value")
        action = str(self._arg("action") or "")

        chain = list(draft.last_chain)
        if not (0 <= chain_index < len(chain)):
            raise ValueError("That control no longer exists.")
        step = chain[chain_index]
        skill = self._deps().skill_loader(step.get("slug") or "")
        if not skill:
            raise ValueError("Skill for that control was deleted.")
        schema = {c.get("name"): c for c in (skill.controls or [])}
        spec = schema.get(name)
        if not spec:
            raise ValueError(f"Unknown control '{name}'.")
        if spec.get("type") == "button":
            act = action or spec.get("action") or "randomize"
            if act != "randomize":
                raise ValueError(f"Unknown button action '{act}'.")
            param = spec.get("param") or "seed"
            new_val = random.randint(1, 2_147_483_647)
            if param == "seed":
                draft.randomize_seed_at(chain_index, new_val)
            else:
                draft.apply_control(chain_index, param, new_val)
        elif spec.get("type") == "pan":
            v = value or {}
            xv = float(v.get("x", spec.get("x_default", 0.0)))
            yv = float(v.get("y", spec.get("y_default", 0.0)))
            draft.apply_control(chain_index, spec["x_param"], xv)
            draft.apply_control(chain_index, spec["y_param"], yv)
        else:
            draft.apply_control(chain_index, name, coerce_control_value(spec, value))
        snap = canvas_render.render_chain(
            self._session_key(), list(draft.last_chain),
            palette_id=draft.palette_id, size=int(draft.size),
            op=f"control:{step.get('slug')}.{name}", out_name="_control_replay.png",
            canvas=draft, skill_loader=self._deps().skill_loader,
        )
        return {"canvas": snap, "event_data": {"chain_index": chain_index, "name": name}}


class RegenerateCanvas(_CanvasAction):
    """Re-run the current chain with a fresh seed on every step."""

    action_type = "regenerate_canvas"

    def _run(self, draft) -> dict[str, Any]:
        import random
        from plugins.tools.helpers import canvas_render

        if not draft.last_chain:
            raise ValueError("Nothing to regenerate yet — make something first.")
        fresh = [random.randint(1, 2_147_483_647) for _ in draft.last_chain]
        draft.reseed_chain(fresh)
        snap = canvas_render.render_chain(
            self._session_key(), list(draft.last_chain),
            palette_id=draft.palette_id, size=int(draft.size),
            op="regenerate", out_name="_regenerate.png",
            canvas=draft, skill_loader=self._deps().skill_loader,
        )
        return {"canvas": snap, "event_data": {"layers": len(draft.last_chain)}}


class DeleteLayer(_CanvasAction):
    """Remove one chain entry and replay. Index 0 clears the canvas."""

    action_type = "delete_canvas_layer"

    def _run(self, draft) -> dict[str, Any]:
        from plugins.tools.helpers import canvas_render, layered_canvas as lc

        chain_index = int(self._arg("chain_index") or 0)
        if not (0 <= chain_index < len(draft.last_chain)):
            raise ValueError("That layer no longer exists.")
        if chain_index == 0:
            draft.reset()
            # Wipe the composite WebP on disk too.
            lc._wipe_composite(self._session_key())
            return {"canvas": lc.to_frontend_shape(draft), "event_data": {"chain_index": 0, "cleared": True}}
        draft.delete_entry(chain_index)
        snap = canvas_render.render_chain(
            self._session_key(), list(draft.last_chain),
            palette_id=draft.palette_id, size=int(draft.size),
            op=f"layer_delete:{chain_index}", out_name="_layer_delete.png",
            canvas=draft, skill_loader=self._deps().skill_loader,
        )
        return {"canvas": snap, "event_data": {"chain_index": chain_index}}


class MoveLayer(_CanvasAction):
    """Reorder one chain entry and replay."""

    action_type = "move_canvas_layer"

    def _run(self, draft) -> dict[str, Any]:
        from plugins.tools.helpers import canvas_render, layered_canvas as lc

        from_index = int(self._arg("from_index") or 0)
        to_index = int(self._arg("to_index") or 0)
        if from_index == to_index:
            return {"canvas": lc.to_frontend_shape(draft), "event_data": {"noop": True}}
        draft.move_entry(from_index, to_index)
        snap = canvas_render.render_chain(
            self._session_key(), list(draft.last_chain),
            palette_id=draft.palette_id, size=int(draft.size),
            op=f"layer_move:{from_index}->{to_index}", out_name="_layer_move.png",
            canvas=draft, skill_loader=self._deps().skill_loader,
        )
        return {"canvas": snap, "event_data": {"from_index": from_index, "to_index": to_index}}


class SetCanvasSize(_CanvasAction):
    """Update the centralized canvas size. Replays if there is a chain."""

    action_type = "set_canvas_size"

    def _run(self, draft) -> dict[str, Any]:
        from plugins.tools.helpers import canvas_render, layered_canvas as lc

        size = int(self._arg("size") or 0)
        draft.set_size(size)
        snap = lc.to_frontend_shape(draft)
        if draft.last_chain:
            snap = canvas_render.render_chain(
                self._session_key(), list(draft.last_chain),
                palette_id=draft.palette_id, size=int(draft.size),
                op=f"size:{draft.size}", out_name="_resize.png",
                canvas=draft, skill_loader=self._deps().skill_loader,
            )
        return {"canvas": snap, "event_data": {"size": draft.size}}


class ResetCanvas(_CanvasAction):
    """Wipe the canvas (state + composite image)."""

    action_type = "reset_canvas"

    def _run(self, draft) -> dict[str, Any]:
        from plugins.tools.helpers import layered_canvas as lc
        draft.reset()
        lc._wipe_composite(self._session_key())
        return {"canvas": lc.to_frontend_shape(draft), "event_data": {"cleared": True}}
