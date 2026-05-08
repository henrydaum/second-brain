from __future__ import annotations

from typing import Any

from state_machine.conversation import FormStep


def form_step_display(step: FormStep) -> dict[str, Any]:
    """Build frontend-neutral display metadata for one form step."""
    choices = _choices(step)
    return {
        "prompt": _prompt(step),
        "assist": _assist(step, bool(choices)),
        "choices": choices,
        "allow_skip": step.required is False,
        "allow_cancel": True,
        "input_mode": _input_mode(step),
    }


def _prompt(step: FormStep) -> str:
    prompt = (step.prompt or "").strip()
    return prompt or f"Enter {_humanize(step.name)}."


def _choices(step: FormStep) -> list[dict[str, Any]]:
    labels = step.enum_labels or []
    return [
        {"value": value, "label": str(labels[i]) if i < len(labels) else str(value)}
        for i, value in enumerate(step.enum or [])
    ]


def _assist(step: FormStep, has_choices: bool) -> str:
    bits: list[str] = []
    if has_choices:
        bits.append("Select an option.")
    elif step.type in {"integer", "int"}:
        bits.append("Reply with a whole number.")
    elif step.type == "number":
        bits.append("Reply with a number.")
    elif step.type == "array":
        bits.append("Send a JSON array.")
    elif step.type == "object":
        bits.append("Send a JSON object.")
    elif step.type in {"boolean", "bool"}:
        bits.append("Reply yes or no.")
    if step.required is False:
        bits.append(_skip_text(step))
    return " ".join(bits)


def _skip_text(step: FormStep) -> str:
    if step.default not in (None, ""):
        return f"Send /skip to use the default: {step.default}."
    return "Send /skip to leave this blank."


def _input_mode(step: FormStep) -> str:
    if step.enum:
        return "choice"
    if step.type in {"integer", "int", "number"}:
        return "number"
    if step.type in {"array", "object"}:
        return "json"
    if step.type in {"boolean", "bool"}:
        return "boolean"
    return "text"


def _humanize(name: str) -> str:
    return str(name or "value").replace("_", " ")
