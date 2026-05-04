from __future__ import annotations

"""Small, serializable conversation-state primitives.

This file intentionally does not know about frontends, LLM providers, or the
database. It is the Poker Monster-style core: participants take actions, the
current phase decides what is legal, and multi-step flows live in `cache`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from state_machine.conversation_phases import BASE_PHASE, PHASE_AWAITING_INPUT

Validator = Callable[[Any], tuple[bool, str | None]]
Handler = Callable[["ConversationState", str, dict[str, Any]], Any]
FormFactory = Callable[[dict[str, Any]], list["FormStep"]]


@dataclass
class FormStep:
    """One requested value in a multi-step command/tool form."""

    name: str
    prompt: str = ""
    required: bool = True
    type: str = "string"
    enum: list[Any] | None = None
    default: Any = None
    validator: Validator | None = None
    prompt_when_missing: bool = False

    def coerce(self, value: Any) -> Any:
        # Form values arrive from text boxes, buttons, or future callbacks, so
        # normalize them before handlers see the collected args.
        if value in (None, "") and not self.required:
            return self.default
        if self.type in {"integer", "int"}:
            value = int(value)
        if self.type == "number":
            value = float(value)
        if self.type == "boolean":
            value = value if isinstance(value, bool) else str(value).strip().lower() in {"true", "yes", "1", "y"}
        if self.type in {"array", "object"} and isinstance(value, str):
            import json
            value = json.loads(value)
        if self.enum and value not in self.enum:
            raise ValueError(f"{self.name} must be one of: {', '.join(map(str, self.enum))}.")
        return value

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if self.required and (value is None or value == ""):
            return False, f"{self.name} is required."
        try:
            value = self.coerce(value)
        except Exception as e:
            return False, str(e)
        return self.validator(value) if self.validator else (True, None)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "prompt": self.prompt, "required": self.required, "type": self.type, "enum": self.enum, "default": self.default, "prompt_when_missing": self.prompt_when_missing}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FormStep":
        return cls(data["name"], data.get("prompt", ""), data.get("required", True), data.get("type", "string"), data.get("enum"), data.get("default"), prompt_when_missing=data.get("prompt_when_missing", False))


@dataclass
class CallableSpec:
    """Runtime description of something callable: slash command or tool."""

    name: str
    handler: Handler | None = None
    form: list[FormStep] = field(default_factory=list)
    require_approval: bool = False
    approval_actor_id: str | None = None
    validator: Validator | None = None
    form_factory: FormFactory | None = None


@dataclass
class Participant:
    """A conversation actor.

    `kind` controls default permissions, but explicit can_* flags let future
    user-user or agent-agent conversations override those defaults.
    """

    id: str
    kind: str
    name: str | None = None
    commands: dict[str, CallableSpec] = field(default_factory=dict)
    tools: dict[str, CallableSpec] = field(default_factory=dict)
    can_command: bool | None = None
    can_tool: bool | None = None
    can_attach: bool | None = None

    def allows(self, action: str) -> bool:
        defaults = {
            "call_command": self.kind == "user",
            "call_tool": self.kind == "agent",
            "send_attachment": self.kind == "user",
        }
        explicit = {"call_command": self.can_command, "call_tool": self.can_tool, "send_attachment": self.can_attach}.get(action)
        return defaults.get(action, True) if explicit is None else explicit


@dataclass
class PhaseFrame:
    """One suspended multi-step flow on the phase stack.

    The frame is deliberately serializable so in-progress forms/approvals can
    be stored in existing conversation_messages rows and restored later.
    """

    phase: str
    action_type: str
    actor_id: str
    name: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    steps: list[FormStep] = field(default_factory=list)
    step_index: int = 0
    previous_phase: str = BASE_PHASE

    @property
    def step(self) -> FormStep | None:
        return self.steps[self.step_index] if self.step_index < len(self.steps) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "action_type": self.action_type,
            "actor_id": self.actor_id,
            "name": self.name,
            "data": self.data,
            "steps": [s.to_dict() for s in self.steps],
            "step_index": self.step_index,
            "previous_phase": self.previous_phase,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseFrame":
        return cls(data["phase"], data["action_type"], data["actor_id"], data.get("name"), data.get("data") or {}, [FormStep.from_dict(s) for s in data.get("steps", [])], data.get("step_index", 0), data.get("previous_phase", BASE_PHASE))


class ConversationState:
    """The pure state machine: turn priority, current phase, and phase stack."""

    def __init__(
        self,
        participants: Iterable[Participant],
        turn_priority: str | None = None,
        phase: str = PHASE_AWAITING_INPUT,
        cache: dict[str, Any] | None = None,
        allowed_attachment_extensions: Iterable[str] = (),
        attachment_parser: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.participants = {p.id: p for p in participants}
        self.turn_order = list(self.participants)
        if not self.turn_order:
            raise ValueError("ConversationState needs at least one participant.")
        self.turn_priority = turn_priority or self.turn_order[0]
        if self.turn_priority not in self.participants:
            raise KeyError(f"Unknown turn priority: {self.turn_priority}")
        self.phase = phase
        self.cache = cache or {"phases": []}
        self.cache["phases"] = [PhaseFrame.from_dict(f) if isinstance(f, dict) else f for f in self.cache.get("phases", [])]
        self.history: list[dict[str, Any]] = []
        self.last_error = None
        self.allowed_attachment_extensions = {e.lower().lstrip(".") for e in allowed_attachment_extensions}
        self.attachment_parser = attachment_parser

    @property
    def active(self) -> Participant:
        return self.participants[self.turn_priority]

    @property
    def frame(self) -> PhaseFrame | None:
        frames = self.cache.setdefault("phases", [])
        return frames[-1] if frames else None

    def other_id(self, actor_id: str | None = None) -> str:
        actor_id = actor_id or self.turn_priority
        if len(self.turn_order) == 1:
            return actor_id
        return self.turn_order[(self.turn_order.index(actor_id) + 1) % len(self.turn_order)]

    def switch_priority(self, actor_id: str | None = None) -> None:
        self.turn_priority = self.other_id(actor_id)

    def set_priority(self, actor_id: str) -> None:
        if actor_id not in self.participants:
            raise KeyError(f"Unknown participant: {actor_id}")
        self.turn_priority = actor_id

    def push_phase(self, frame: PhaseFrame) -> None:
        # Push instead of overwrite so an approval/form can pause another
        # pending action and then resume it.
        frame.previous_phase = self.phase
        self.cache.setdefault("phases", []).append(frame)
        self.phase = frame.phase

    def pop_phase(self) -> PhaseFrame | None:
        frame = self.cache.setdefault("phases", []).pop() if self.cache.setdefault("phases", []) else None
        self.phase = self.frame.phase if self.frame else BASE_PHASE
        return frame

    def reset_phase(self) -> None:
        self.cache["phases"] = []
        self.phase = BASE_PHASE

    def event(self, type_: str, actor_id: str | None = None, **data: Any) -> dict[str, Any]:
        event = {"type": type_, "actor_id": actor_id or self.turn_priority, "phase": self.phase, **data}
        self.history.append(event)
        return event

    def enact(self, action_type: str, content: Any = None, actor_id: str | None = None):
        from state_machine.action_map import create_action

        return create_action(self, action_type, content, actor_id).enact()

    def spec(self, actor_id: str, action_type: str, name: str) -> CallableSpec | None:
        table = self.participants[actor_id].commands if action_type == "call_command" else self.participants[actor_id].tools
        return table.get(name)

    def attachment_extension(self, content: dict[str, Any]) -> str:
        return (content.get("extension") or Path(str(content.get("path", ""))).suffix).lower().lstrip(".")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_priority": self.turn_priority,
            "phase": self.phase,
            "cache": {**self.cache, "phases": [f.to_dict() if hasattr(f, "to_dict") else f for f in self.cache.get("phases", [])]},
            "history": self.history,
            "participants": [{"id": p.id, "kind": p.kind, "name": p.name} for p in self.participants.values()],
        }
