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


MODEL_ADD_PARAMS = [
    FormParam("llm_model_name", description="The model identifier sent to the API (e.g. gpt-4, llama-3.1-8b, gemini-2.5-flash).", required=True),
    FormParam("llm_endpoint", description="Custom API endpoint URL. Leave blank for the default OpenAI endpoint. For LM Studio, see developer tab."),
    FormParam("llm_api_key", description="API key or environment variable name (e.g. OPENAI_API_KEY). Leave blank for local models."),
    FormParam("llm_context_size", type="integer", description="Max context window in tokens. Set 0 for reactive-only compaction."),
    FormParam("llm_service_class", description="Which LLM backend to use.", required=True, enum=["OpenAILLM", "LMStudioLLM"]),
]


SCHEDULE_CREATE_STEPS = [
    "job_name", "schedule_type", "schedule_value",
    "channel", "prompt", "title", "description",
]
