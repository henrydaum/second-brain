"""The LLM backend contract — what a ``llm_*.py`` helper is handed and returns.

This lives in the guest for the same reason :mod:`guest.parsing` does: a
backend is *guest code*, running with a foreign SDK and a network connection,
which is precisely what is least worth trusting in-process. A backend
importing a kernel module would load in-process and die in a subprocess.

A backend is a class::

    class MyBackend(BaseLLMBackend):
        def chat(self, sdk, request: LLMRequest) -> LLMResponse

A class rather than a function because a backend is a *residency*: the
imported provider library living on the instance is what "loaded" means, and
loading it is the expensive part nobody wants to repeat per call. But it is
stateless with respect to the *profile* — every model name, key and endpoint
arrives on the request — which is what lets the kernel keep a pool of
interchangeable boxes for one model.

Three things cross, and all three are plain data:

- :class:`LLMRequest` in,
- :class:`LLMResponse` out,
- text deltas, pushed one at a time through ``sdk.llm.delta``.

Nothing else. In particular there is no ``on_delta`` callback, and the abort
boolean it used to return is gone rather than ported — see CLAUDE.md,
"Streaming inverted, and lost a feature on purpose".

Stdlib only, like everything else in here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Error classification.
#
# Needed on *both* sides of the boundary, which is the whole argument for
# putting it here: the backend classifies its own provider exception (only it
# knows what ``ContextWindowExceededError`` is), and the kernel's compaction
# layer classifies the error it was handed back. One implementation, or the
# two drift and a context overflow stops triggering a compaction retry.
# ──────────────────────────────────────────────────────────────────────

_CONTEXT_LIMIT_HINTS = (
    "context window", "context length", "context_length", "maximum context",
    "max context", "too many tokens", "too long", "max_tokens",
    "prompt is too long", "prompt tokens", "exceeds limit", "exceeds limits",
    "exceeded limit", "token limit", "request too large",
)

_CONTEXT_RELATED_TERMS = ("context", "token", "prompt", "input")
_LIMIT_RELATED_TERMS = ("limit", "limits", "length", "maximum", "max",
                        "too long", "too many", "exceed", "exceeds",
                        "exceeded")
# Phrases that look like a limit but are a *billing* or *availability* answer.
# Retrying these after compaction would compact for nothing and fail again.
_NON_CONTEXT_LIMIT_HINTS = (
    "not support model", "not supported model", "current token plan",
    "token plan not support",
)


def _stringify_error_detail(value) -> str:
    """Best-effort conversion of SDK error payloads into searchable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def extract_llm_error_text(error) -> str:
    """Flatten a provider exception and its payload into one text blob.

    Providers hide the useful sentence in a different attribute each — the
    string form, ``message``, a nested ``body`` dict — so all of them are
    searched rather than guessing which one this provider used.
    """
    if error is None:
        return ""
    parts = [str(error)]
    for attr in ("message", "body", "response", "error", "errors"):
        text = _stringify_error_detail(getattr(error, attr, None))
        if text:
            parts.append(text)
    return " | ".join(part for part in parts if part)


def is_context_limit_error(error) -> bool:
    """Whether this failure means "the prompt was too big".

    A heuristic, and deliberately so: every provider spells it differently and
    a new one appears every few months. Guessing wrong in the *false* direction
    costs a failed turn; guessing wrong in the *true* direction costs one
    wasted compaction. So the bar is low, with an explicit exclusion list for
    the phrases that read like a limit but are really about billing.
    """
    text = extract_llm_error_text(error).lower()
    if not text:
        return False
    if any(hint in text for hint in _NON_CONTEXT_LIMIT_HINTS):
        return False
    if any(hint in text for hint in _CONTEXT_LIMIT_HINTS):
        return True
    if (any(term in text for term in _CONTEXT_RELATED_TERMS)
            and any(term in text for term in _LIMIT_RELATED_TERMS)):
        return True
    return "invalid params" in text and "window" in text and "limit" in text


# ──────────────────────────────────────────────────────────────────────
# What crosses.
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LLMRequest:
    """One call to one model.

    Everything the old ``BaseLLM`` instance held as attributes is here
    instead, because a pooled box serves whichever profile asks: two boxes for
    the same backend must be interchangeable, and they only are if none of
    them remembers a model name.

    ``attachments`` arrives already routed. The kernel has run
    ``AttachmentBundle.split_for_llm`` against this model's declared
    capabilities and appended the text fallback to the last user message, so
    what lands here is only what the backend should send natively — a list of
    ``{path, modality, file_name}`` dicts, whose bytes the backend reads with
    ``sdk.fs.read_bytes``.
    """

    model_name: str = ""
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    attachments: list[dict] = field(default_factory=list)
    # Extra provider kwargs (temperature, tool_choice, ...). Forwarded as-is.
    params: dict = field(default_factory=dict)
    # Connection. ``api_key`` is plaintext: a provider library does its own
    # I/O, so there is no outbound Request for the kernel to substitute a
    # ``<secret:...>`` handle into. See docs/SECURITY_CONTRACT_APPENDIX.md.
    api_key: str = ""
    base_url: str = ""
    # Whether the caller wants deltas pushed through ``sdk.llm.delta``. A
    # backend that cannot stream ignores it; the response shape is identical
    # either way, which is what lets the kernel decide per call.
    stream: bool = False

    def to_dict(self) -> dict:
        """Serialize for the wire."""
        return {
            "model_name": self.model_name, "messages": self.messages,
            "tools": self.tools, "attachments": self.attachments,
            "params": self.params, "api_key": self.api_key,
            "base_url": self.base_url, "stream": self.stream,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMRequest":
        """Rebuild from the wire, tolerating a sender that omitted fields."""
        data = data or {}
        known = {f: data.get(f) for f in (
            "model_name", "messages", "tools", "attachments", "params",
            "api_key", "base_url", "stream")}
        return cls(
            model_name=known["model_name"] or "",
            messages=known["messages"] or [],
            tools=known["tools"],
            attachments=known["attachments"] or [],
            params=known["params"] or {},
            api_key=known["api_key"] or "",
            base_url=known["base_url"] or "",
            stream=bool(known["stream"]),
        )


@dataclass
class LLMResponse:
    """What a model said, however it was asked.

    One shape for the blocking and streaming paths alike — the streaming call
    accumulates deltas and returns the same thing — so no caller has to branch
    on how the call was made.
    """

    content: str = ""
    # Each: {"id": str, "name": str, "arguments": str (JSON)}
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    error: str | None = None
    error_code: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model wants to call something."""
        return len(self.tool_calls) > 0

    @property
    def is_error(self) -> bool:
        """Whether this response reports a failure."""
        return bool(self.error)

    @property
    def is_context_limit_error(self) -> bool:
        """Whether the failure was an oversized prompt.

        An explicit ``error_code`` wins: a backend that recognised its own
        provider's exception knows better than the text heuristic.
        """
        if self.error_code == "context_limit":
            return True
        return is_context_limit_error(self.error or self.content)

    def to_dict(self) -> dict:
        """Serialize for the wire."""
        return {
            "content": self.content, "tool_calls": self.tool_calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "error": self.error, "error_code": self.error_code,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMResponse":
        """Rebuild from the wire.

        Tolerant on purpose: this runs on whatever a box handed back, and a
        malformed response should surface as an error response rather than as
        a ``TypeError`` inside the conversation loop.
        """
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return cls(content="", error=f"malformed LLM response: {data!r}",
                       error_code="provider_error")
        calls = data.get("tool_calls")
        return cls(
            content=data.get("content") or "",
            tool_calls=list(calls) if isinstance(calls, list) else [],
            prompt_tokens=data.get("prompt_tokens"),
            cached_prompt_tokens=data.get("cached_prompt_tokens"),
            error=data.get("error"),
            error_code=data.get("error_code"),
        )

    @classmethod
    def failure(cls, message: str, code: str = "provider_error") -> "LLMResponse":
        """The error shape, spelled once.

        ``content`` carries the message too because some callers only ever
        look at the text.
        """
        return cls(content=f"Error: {message}", error=message, error_code=code)


class LLMProviderError(RuntimeError):
    """A provider failed in a way the kernel should react to, not just log.

    Crosses the boundary as *data* — an :class:`LLMResponse` with an
    ``error_code`` — and is re-raised host-side by the kernel. Marshalling a
    live exception would need pickle, and pickle is exactly what a boundary
    exists to avoid.
    """

    def __init__(self, message: str, code: str = "provider_error"):
        super().__init__(message)
        self.code = code


class BaseLLMBackend:
    """What a sandboxed LLM backend subclasses.

    Not a plugin: no family, no entry point, nothing discovery registers —
    exactly like a parser. It lives in the ``llm/`` tree root as
    ``llm_*.py``, and the LLM registry finds it by declaration.

    Declare capabilities at *module* level, so the kernel can read them
    without importing the file::

        supports_streaming = True
        supports_tool_choice = True
        display_name = "LiteLLM"
        dependencies_pip = ["litellm"]

    Isolation is not among them. A backend importing a provider library is
    subprocessed because the kernel can see that import, not because the file
    asked — see ``sandbox/isolation.py``.
    """

    def start(self, sdk):
        """Import the provider library and stand by. Called once per box.

        Anything expensive belongs here rather than in ``chat``: the box is
        resident precisely so this cost is paid once.
        """
        return True

    def chat(self, sdk, request):
        """Answer one :class:`LLMRequest` with one :class:`LLMResponse`.

        When ``request.stream`` is set and this backend declared
        ``supports_streaming``, push text through ``sdk.llm.delta(text)`` as
        it arrives *and* return the accumulated response — the deltas are for
        the user's eyes, the response is what the kernel records. Nothing
        needs to check whether the user cancelled: they cannot, without the
        kernel cancelling this execution, at which point the next Request
        raises ``Terminated``.
        """
        raise NotImplementedError

    def stop(self, sdk):
        """Release anything ``start`` opened. Called once, as the box closes."""
        return True

    def __chat__(self, sdk, request: dict, token: str = ""):
        """Receive one call. The kernel calls this, never an author.

        The same translation ``__hook__`` does for doorways: the wire carries
        dicts, the author writes against dataclasses, and the rehydration
        happens here on the guest side where those dataclasses live.

        A raised exception is classified and returned as an error *response*
        rather than propagating, so the kernel gets a shape it can reason
        about. ``Terminated`` is the exception — it is a ``BaseException`` and
        is not caught, because a cancelled call must not look like a failed
        one.
        """
        sdk._delta_token = token
        try:
            return self.chat(sdk, LLMRequest.from_dict(request)).to_dict()
        except Exception as exc:                    # noqa: BLE001
            code = ("context_limit" if is_context_limit_error(exc)
                    else "provider_error")
            return LLMResponse.failure(extract_llm_error_text(exc),
                                       code).to_dict()
        finally:
            sdk._delta_token = ""
