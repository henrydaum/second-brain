from __future__ import annotations

import json
from typing import Any

from state_machine.conversationClass import FormStep


def schema_to_form_steps(schema: dict | None) -> list[FormStep]:
    props = (schema or {}).get("properties", {})
    required = set((schema or {}).get("required", []))
    return [
        FormStep(name, info.get("description", ""), name in required, info.get("type", "string"), info.get("enum"), info.get("default"))
        for name, info in props.items()
    ]


def coerce_form_value(raw: Any, step: FormStep) -> Any:
    return step.coerce(raw)


def history_tool_calls_from_content(content: str) -> dict | None:
    try:
        parsed = json.loads(content or "")
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) and "tool_calls" in parsed else None
