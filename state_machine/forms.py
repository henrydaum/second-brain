"""State-machine support for forms."""

from __future__ import annotations

import json
from typing import Any

from state_machine.conversation import FormStep


def schema_to_form_steps(schema: dict | None, *, prompt_optional: bool = False) -> list[FormStep]:
    """Handle schema to form steps."""
    props = (schema or {}).get("properties", {})
    required = set((schema or {}).get("required", []))
    return [
        FormStep(name, _schema_prompt(name, info), name in required, info.get("type", "string"), info.get("enum"), default=info.get("default"), prompt_when_missing=prompt_optional and name not in required)
        for name, info in props.items()
    ]


def _schema_prompt(name: str, info: dict) -> str:
    """Internal helper to handle schema prompt."""
    label = str(name or "value").replace("_", " ")
    desc = str((info or {}).get("description") or "").strip()
    action = "Choose" if (info or {}).get("enum") or (info or {}).get("type") == "boolean" else "Enter"
    prompt = f"{action} {_article(label) if action == 'Enter' else label}."
    return f"{prompt}\n{desc}" if desc else prompt


def _article(label: str) -> str:
    """Internal helper to handle article."""
    return label if label.startswith(("a ", "an ", "the ")) else f"{'an' if label[:1].lower() in 'aeiou' else 'a'} {label}"


def coerce_form_value(raw: Any, step: FormStep) -> Any:
    """Handle coerce form value."""
    return step.coerce(raw)


def history_tool_calls_from_content(content: str) -> dict | None:
    """Handle history tool calls from content."""
    try:
        parsed = json.loads(content or "")
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) and "tool_calls" in parsed else None
