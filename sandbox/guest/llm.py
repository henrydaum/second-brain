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
    completion_tokens: int | None = None
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
            "completion_tokens": self.completion_tokens,
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
            completion_tokens=data.get("completion_tokens"),
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

    # ── Describing what can be configured ─────────────────────────────
    #
    # Three optional questions, ordered from least to most specific:
    # which providers exist, which models one endpoint serves, and which
    # parameters one model takes. Each narrows the last, and each is
    # answered by the backend because only it knows what its provider
    # library can tell it.
    #
    # All four default to ``[]``, which means "I cannot say" and is a real
    # answer rather than a failure: a backend implementing none of them
    # leaves the user typing the values by hand, exactly as before. That is
    # also the *common* path — model aggregators appear in no provider
    # table — so nothing above these may treat an empty list as an error.
    #
    # **Every row may carry a ``description``**, and the same "I cannot say"
    # rule governs it: ``""`` is the default and means nothing is rendered.
    # It is prose *about the thing the row names* — what a parameter does,
    # what kind of model this is — and it is deliberately the one field with
    # no structure at all, because it exists for the cases where a fact is
    # worth telling somebody and is not worth a key of its own.
    #
    # It is the backend's own account, never a claim the kernel makes, and a
    # caller should render it as such — ``/llm`` quotes it. That register is
    # the point rather than decoration: a backend reads these out of whatever
    # its provider library documents, so the text may describe the *spec* a
    # parameter belongs to rather than the model in front of you, and a quote
    # says "here is what somebody else says" where a sentence of our own
    # would be asserting it.

    def providers(self, sdk, provider: str = "") -> list:
        """Providers this backend can talk to.

        Each entry is ``{"id": str, "label": str, "endpoint": str}``, plus an
        optional ``description``.

        Two questions share one method because they are the same question at
        two widths. Asked bare, it is the *menu* — every provider, and
        ``endpoint`` may be left ``""`` because filling in a hundred and fifty
        of them is not worth what it costs. Asked with *provider*, it is the
        one row somebody actually chose, and the endpoint is the whole reason
        for asking: resolving it may be slow, may reach the network, and for a
        provider that authenticates by device code may even start a login —
        all of which is fine for a provider a person just selected, and none
        of which is fine a hundred and fifty times over to draw a list.

        ``endpoint`` still comes back ``""`` when there genuinely is none, as
        for a provider reached through its own SDK. Blank is honest; a guessed
        URL is not, because it fails much later and blames the model.

        ``description`` follows the same split, and for the same reason: it
        belongs to the narrowed call. A hundred and fifty rows have nowhere to
        put prose anyway — a menu shows labels — so anything worth saying is
        worth saying about the one provider somebody picked.
        """
        return []

    def models(self, sdk, endpoint: str, api_key: str,
               provider: str = "", live: bool = False) -> list:
        """Models reachable at *endpoint*, as ``{"name", "label"}``.

        A ``description`` here should be the *endpoint's own* words about the
        model, when it volunteers any, rather than something composed from a
        table — that is what ``info`` is for, asked about the one model
        chosen. A row in a picker is a name and a label; only the model
        somebody settled on has a place to be described at length.

        ``name`` is the string **this backend wants to be handed back** in
        ``LLMRequest.model_name`` — any provider prefix already applied. That
        is the whole point of asking: which prefix a backend needs is a fact
        about the backend, and making the user know it is how a working model
        ends up unreachable for want of five characters.

        ``live`` is permission to *ask the endpoint*, and it defaults to off.
        A listing fetched from the server is the better answer — current, and
        it covers gateways no table has heard of — but fetching it is egress,
        and the caller that most wants this is a settings form, which is the
        one place that cannot do egress: a command's approval is evaluated on
        its *completed* arguments, so anything a form does runs ungranted, and
        a dialog raised while building a form deadlocks against the session
        lock it is already holding.

        So the default is whatever the backend knows offline, and ``live`` is
        for a caller that already holds a grant. Answering ``[]`` is fine.
        """
        return []

    def info(self, sdk, model_name: str, endpoint: str = "") -> list:
        """Facts about one model, as a single-row list.

        ``[{"context_size": int, "description": str}]`` today, both optional.
        This is the row that most earns a ``description``: it is asked once,
        about a model somebody has chosen, at the moment they are being asked
        to confirm numbers about it. A list rather than a dict because
        every discovery answer is a list and ``__describe__`` normalizes one
        shape; a lone dict would iterate as its keys and arrive as nonsense.

        A row carrying only a ``description`` is a real answer — knowing what
        kind of model this is while not knowing its window is an ordinary
        state for a lookup table to be in, and the two facts are independent.

        ``context_size`` is the *input* window, which is what the kernel
        budgets against — not the output cap, which is a different and usually
        much smaller number. Answer ``[]`` or omit the key when the model is
        not in whatever table this backend consults; a wrong context size is
        worse than none, because the kernel compacts against it and would
        either waste most of the window or overflow it every turn.
        """
        return []

    def params(self, sdk, model_name: str, endpoint: str) -> list:
        """Extra provider parameters *model_name* accepts.

        Each entry is ``{"name", "label", "kind", "choices", "supported",
        "note"}``, plus an optional ``description``. ``kind`` is ``"choice"``,
        ``"number"``, ``"bool"`` or ``"text"``; ``choices`` matters only for
        ``"choice"``.

        ``description`` and ``note`` answer different questions and both are
        worth having: the description says **what this parameter is**, which
        is true of the parameter wherever it is sent, while the note says
        **what will happen to this value here**, which is a fact about this
        model at this provider. So a supported parameter routinely has a
        description and no note.

        ``supported`` is **a report, never a gate.** A caller shows a false
        one greyed with its ``note`` and still lets it be set, because this
        answer is a lookup in somebody else's table and those tables are
        wrong sometimes — the case that forced this rule was a provider whose
        table omitted the very parameter its API documents. Treating the
        lookup as authoritative would have hidden the setting that works and
        told the user their model could not reason.

        So ``note`` carries the *why*, and should say what will happen to the
        value rather than what the model can do: "this backend drops it; try
        ``thinking``" is actionable, "not supported" is a claim about the
        model that this method is in no position to make.
        """
        return []

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

    def __describe__(self, sdk, question: str, args: dict = None):
        """Answer one of the three discovery questions. The kernel calls this.

        One entry point rather than three, because the three differ only in
        their arguments and every one of them answers a list — a second and
        third wire name would buy nothing and cost three places to keep in
        step.

        Anything raised becomes ``[]``. These questions are asked while
        somebody is filling in a form, and a backend that cannot introspect
        must be no worse than one that never offered to: the form falls back
        to free text, which is where it started.
        """
        args = args or {}
        try:
            if question == "providers":
                answer = self.providers(sdk, args.get("provider") or "")
            elif question == "models":
                answer = self.models(sdk, args.get("endpoint") or "",
                                     args.get("api_key") or "",
                                     args.get("provider") or "",
                                     bool(args.get("live")))
            elif question == "info":
                answer = self.info(sdk, args.get("model_name") or "",
                                   args.get("endpoint") or "")
            elif question == "params":
                answer = self.params(sdk, args.get("model_name") or "",
                                     args.get("endpoint") or "")
            else:
                return []
            return [dict(row) for row in answer if isinstance(row, dict)]
        except Exception as exc:                    # noqa: BLE001
            sdk.log(f"backend could not answer {question!r}: {exc}")
            return []
