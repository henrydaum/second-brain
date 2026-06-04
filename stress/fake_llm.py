"""Network-free LLM backends for stress testing.

These duck-type the kernel's ``BaseLLM`` surface that the conversation loop
actually touches (``loaded``, ``context_size``, ``capabilities``,
``chat_with_tools``, ``load``/``unload``) without inheriting ``BaseService`` —
we want them dead simple and always "loaded", with no real model behind them.

Two flavours:

- :class:`ScriptedLLM` replays a fixed queue of responses (deterministic tests).
- :class:`MonkeyLLM` is the fuzzing brain: given the tool schemas offered on a
  turn, it randomly decides to answer with text or to call a tool with
  plausibly-shaped (and sometimes deliberately malformed) arguments. It is
  seeded, so a failing fuzz example replays exactly.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from plugins.services.service_llm import LLMResponse


class _BaseFake:
    """Minimal stand-in for the bits of BaseLLM the loop calls."""

    is_llm_backend = False  # not discoverable; injected directly into services

    def __init__(self, *, context_size: int = 0):
        # context_size 0 disables proactive compaction in the loop, which is what
        # we want by default — compaction is exercised explicitly elsewhere.
        self.context_size = context_size
        self.loaded = True
        self.model_name = "fake"
        self.capabilities = {"image": None, "audio": None, "video": None}
        self.last_prompt_tokens = 0
        self.last_cached_prompt_tokens = 0
        self.calls: list[list[dict]] = []

    def load(self):
        self.loaded = True
        return True

    def unload(self):
        self.loaded = False

    def has_capability(self, modality: str) -> bool:
        return bool(self.capabilities.get(modality))

    def invoke(self, messages, attachments=None, **kwargs):
        return self.chat_with_tools(messages, None, attachments=attachments, **kwargs)

    def chat_with_tools(self, messages, tools=None, attachments=None, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError


class ScriptedLLM(_BaseFake):
    """Replays a fixed list of :class:`LLMResponse` objects, one per call."""

    def __init__(self, responses, *, context_size: int = 0):
        super().__init__(context_size=context_size)
        self.model_name = "scripted-fake"
        self._responses = list(responses)

    def chat_with_tools(self, messages, tools=None, attachments=None, **kwargs):
        self.calls.append(messages)
        if not self._responses:
            return LLMResponse(content="(scripted LLM exhausted)")
        return self._responses.pop(0)


def text(content: str) -> LLMResponse:
    """Helper: a plain text reply."""
    return LLMResponse(content=content)


def tool_call(name: str, arguments: dict | str, call_id: str = "call_1") -> LLMResponse:
    """Helper: a single tool call."""
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return LLMResponse(content="", tool_calls=[{"id": call_id, "name": name, "arguments": args}])


@dataclass
class MonkeyConfig:
    """Knobs for the fuzzing LLM brain."""
    tool_call_prob: float = 0.55      # chance of calling a tool vs. answering
    malformed_arg_prob: float = 0.15  # chance of emitting junk JSON arguments
    max_tool_calls_per_turn: int = 1  # keep turns bounded
    end_phrases: tuple[str, ...] = ("done", "ok", "here you go", "finished", "no problem")


class MonkeyLLM(_BaseFake):
    """A seeded, schema-aware random agent brain.

    On each turn it inspects the offered tool schemas and either replies with
    text or fabricates a tool call. Arguments are shaped from the schema's
    declared properties (so most calls are *valid* — the interesting fuzzing
    surface is valid sequences, syzkaller-style), with an occasional malformed
    payload to exercise the loop's error handling.
    """

    def __init__(self, seed: int = 0, config: MonkeyConfig | None = None, *, context_size: int = 0):
        super().__init__(context_size=context_size)
        self.model_name = f"monkey-{seed}"
        self.rng = random.Random(seed)
        self.cfg = config or MonkeyConfig()

    def _fabricate_args(self, schema: dict) -> str:
        params = (schema.get("function", schema).get("parameters") or {})
        props = params.get("properties") or {}
        required = params.get("required") or list(props)
        if self.rng.random() < self.cfg.malformed_arg_prob:
            return self.rng.choice(['{not json', '{"x": }', '', '[]', 'null'])
        out: dict = {}
        for key in props:
            if key not in required and self.rng.random() < 0.5:
                continue
            out[key] = self._fabricate_value(props[key])
        return json.dumps(out)

    def _fabricate_value(self, spec: dict):
        t = spec.get("type", "string")
        if "enum" in spec and spec["enum"]:
            return self.rng.choice(spec["enum"])
        if t == "integer":
            return self.rng.randint(-3, 9)
        if t == "number":
            return round(self.rng.uniform(-3, 9), 2)
        if t == "boolean":
            return self.rng.random() < 0.5
        if t == "array":
            return []
        if t == "object":
            return {}
        return self.rng.choice(["test", "hello world", "", "../etc", "🙂", "a" * 200])

    def chat_with_tools(self, messages, tools=None, attachments=None, **kwargs):
        self.calls.append(messages)
        tools = tools or []
        # If the last message is already a tool result, lean toward wrapping up
        # so turns terminate (mirrors a real agent finishing after a tool runs).
        last = messages[-1] if messages else {}
        just_ran_tool = isinstance(last, dict) and last.get("role") == "tool"
        wants_tool = (
            tools
            and not just_ran_tool
            and self.rng.random() < self.cfg.tool_call_prob
        )
        if wants_tool:
            schema = self.rng.choice(tools)
            name = schema.get("function", schema).get("name")
            return tool_call(name, self._fabricate_args(schema), call_id=f"mc_{self.rng.randint(0, 10**6)}")
        return text(self.rng.choice(self.cfg.end_phrases))
