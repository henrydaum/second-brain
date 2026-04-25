from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class FormParam:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    enum: list | None = None
    default: object = ""


@dataclass
class PendingParamForm:
    subject: str = ""
    params: list[FormParam] = field(default_factory=list)
    collected: dict[str, object] = field(default_factory=dict)
    current_idx: int = 0
    awaiting_name: bool = False

    @property
    def current_param(self) -> FormParam | None:
        return self.params[self.current_idx] if self.current_idx < len(self.params) else None

    def skip_current(self) -> bool:
        if self.current_param is None or self.current_param.required:
            return False
        self.current_idx += 1
        return True

    def store(self, name: str, value: object):
        self.collected[name] = value
        self.current_idx += 1


@dataclass
class PendingScheduleCreate:
    step: int = 0
    collected: dict[str, str] = field(default_factory=dict)

    def current_field(self, steps: list[str]) -> str | None:
        return steps[self.step] if self.step < len(steps) else None


def schema_to_params(schema: dict | None) -> list[FormParam]:
    props = schema.get("properties", {}) if schema else {}
    required = set(schema.get("required", [])) if schema else set()
    return [
        FormParam(
            name=name,
            type=info.get("type", "string"),
            description=info.get("description", ""),
            required=name in required,
            enum=info.get("enum"),
        )
        for name, info in props.items()
    ]


def coerce_param_value(raw: str, param_type: str):
    if param_type == "integer":
        return int(raw)
    if param_type == "number":
        return float(raw)
    if param_type == "boolean":
        return raw.lower() in ("true", "yes", "1")
    if param_type == "array":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return [line.strip() for line in (raw.splitlines() if "\n" in raw else raw.split(",")) if line.strip()]
    if param_type == "object":
        return json.loads(raw)
    return raw


LLM_ADD_PARAMS = [
    FormParam("llm_endpoint", description="Custom API endpoint URL. Leave blank for the default OpenAI endpoint. For LM Studio, see developer tab."),
    FormParam("llm_api_key", description="API key or environment variable name (e.g. OPENAI_API_KEY). Leave blank for local models."),
    FormParam("llm_context_size", type="integer", description="Max context window in tokens. Set 0 for reactive-only compaction."),
    FormParam("llm_service_class", description="Which LLM backend to use.", required=True, enum=["OpenAILLM", "LMStudioLLM"]),
]


def _format_name_hint(title: str, names: list[str] | None, note: str = "") -> str:
    if not names:
        return ""
    lines = []
    current = ""
    for name in names:
        item = f"`{name}`"
        next_line = item if not current else f"{current}, {item}"
        if len(next_line) > 72 and current:
            lines.append(current)
            current = item
        else:
            current = next_line
    if current:
        lines.append(current)
    note_line = f"\n{note}" if note else ""
    return f"\n\n{title}:{note_line}\n" + "\n".join(lines)


def agent_add_params(llm_choices: list[str], tool_names: list[str] | None = None,
                     table_names: list[str] | None = None) -> list[FormParam]:
    """Build the agent-profile creation form. ``llm_choices`` is the live list
    of model names from llm_profiles, plus the literal 'default' sentinel —
    it must be passed in at form-start time so the dropdown reflects what's
    actually configured."""
    tool_hint = _format_name_hint(
        "Agent-visible tool names",
        tool_names,
        "Use these registry names, not filenames like `tool_run_command.py`.",
    )
    table_hint = _format_name_hint("Database table names", table_names)
    return [
        FormParam("llm", description="LLM to use. 'default' follows whatever LLM is currently the default.",
                  required=True, enum=llm_choices),
        FormParam("prompt_suffix", description="Extra text appended to the system prompt. Leave blank for none."),
        FormParam("tools_allow", type="array", description=f"Whitelist of tool names. Skip for no restriction.{tool_hint}", default=None),
        FormParam("tools_deny", type="array", description=f"Blacklist of tool names. Skip for no restriction.{tool_hint}", default=None),
        FormParam("tables_allow", type="array", description=f"Whitelist of database tables. Skip for no restriction.{table_hint}", default=None),
        FormParam("tables_deny", type="array", description=f"Blacklist of database tables. Skip for no restriction.{table_hint}", default=None),
    ]


SCHEDULE_CREATE_STEPS = [
    "job_name", "schedule_type", "schedule_value",
    "channel", "prompt", "title", "description",
]
