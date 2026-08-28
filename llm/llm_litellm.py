"""LiteLLM backend — one client for every provider.

Migrated from ``services/service_litellm.py``, which subclassed ``BaseLLM``
from ``plugins.services.service_llm``. That import is why this file could
never be isolated: a kernel import loads in-process and dies in a subprocess,
and this is the file that most wants the process boundary — a large volatile
third-party SDK, a network socket, and an API key.

What changed, beyond the imports: everything about *which* model is on the
request rather than on the instance, so the kernel can run a pool of these
boxes and serve concurrent calls in parallel. The instance holds only what is
genuinely per-process: the imported library.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT ``llm_extra_params`` ACCEPTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

An LLM profile may carry an ``llm_extra_params`` object. The kernel resolves
it into ``LLMRequest.params`` and ``_provider_kwargs`` hands it to
``litellm.completion`` as keyword arguments. **So the vocabulary is LiteLLM's,
not Second Brain's** — which is the point. Conforming to one normalized
spelling is what lets a profile move between providers without the kernel
learning any of them, and it makes "which providers support this" LiteLLM's
problem to track rather than ours.

Written in ``plugin_config.json`` like this::

    "llm_profiles": {
        "gpt-5.1": {
            "llm_endpoint": "",
            "secret_llm_api_key": "OPENAI_API_KEY",
            "llm_context_size": 400000,
            "llm_service_class": "LiteLLMService",
            "llm_extra_params": {
                "reasoning_effort": "high",
                "temperature": 0.2
            }
        }
    }

``/llm`` -> Edit -> Extra parameters edits that object; ``/llm`` -> Edit ->
Reasoning effort edits the one member with a picker.

TWO RULES BEFORE THE LIST
-------------------------

**A null means "do not send this".** ``{"reasoning_effort": null}`` omits the
param entirely. This is a kernel convention rather than a LiteLLM one, and it
exists because the kernel supplies ``reasoning_effort`` for a profile that
says nothing (``llm.DEFAULT_REASONING_EFFORT``) — without a way to decline, it
would be the one param a profile could not refuse. Note ``null`` and
``"none"`` are different things: null sends nothing and the model does
whatever it natively does, ``"none"`` actively asks the provider not to think.
The literal string ``"off"`` is accepted as an alias for null, because that is
the word ``/llm``'s picker shows and therefore the word people hand-edit into
the file; it is safe to alias because ``off`` is a level at no provider.

**Five keys belong to the profile's own fields, not here.** ``api_key`` and
``api_base`` come from ``secret_llm_api_key`` / ``llm_endpoint``; ``model``,
``messages`` and ``tools`` are the call itself. ``_provider_kwargs`` merges
with ``setdefault``, so a value here *wins* over the profile's — silently, and
in the case of a credential it also lands in plaintext config instead of
behind the ``secret_`` prefix. ``/llm`` refuses all six and names the field
that really sets each; hand-editing the config bypasses that check, so do not
put them here.

WHAT ``drop_params`` COVERS, AND WHAT IT DOES NOT
-------------------------------------------------

``start`` sets ``litellm.drop_params = True``, which means:

* an **unsupported param** for the resolved model is removed before the call.
  Sending ``reasoning_effort`` to a model that has no such setting is free.
* an **unsupported value** for a supported param is *not* touched. It goes to
  the provider, which decides — usually a 400. ``"reasoning_effort": "off"``
  is not a level anywhere; the spelling for that is ``null``.

So the safety net catches the wrong *key*, never the wrong *word*.

THE PARAMS WORTH KNOWING
------------------------

Coverage below is measured with ``get_supported_openai_params`` across six
representative providers (OpenAI, Anthropic, Gemini, DeepSeek, MiniMax, Groq)
and is a rough guide, not a contract — ask LiteLLM about the model in hand
rather than trusting this list to stay current.

*Reasoning*

``reasoning_effort``  (4/6 — OpenAI, Anthropic, Gemini, DeepSeek)
    ``"none" | "minimal" | "low" | "medium" | "high" | "xhigh"``, which is
    LiteLLM's own literal in ``main.completion``. ``"max"`` appears only on
    the Responses-API path for Claude 4.6+, so do not rely on it here.
    The cross-provider standard, and the one to reach for first.

``thinking``  (4/6 — Anthropic, Gemini, DeepSeek, MiniMax)
    Provider-shaped: ``{"type": "enabled", "budget_tokens": 2048}`` for
    Anthropic and Bedrock. Reach for it only when a model takes no
    ``reasoning_effort`` — MiniMax is the example, it accepts ``thinking``
    and ``reasoning_split`` and no effort level at all.

On Anthropic the two are the same dial: LiteLLM turns an effort into a
token budget — minimal 128, low 1024, medium 2048, high 4096, xhigh 8192,
max 16384 — each overridable through a matching
``DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET`` environment variable.

    **The Anthropic caveat, which is this backend's one real gap.** A Claude
    turn with thinking enabled must hand its cryptographically signed
    ``thinking_blocks`` back on the next tool-result turn, or the API refuses
    the call with *"Expected `thinking` or `redacted_thinking`, but found
    `tool_use`"*. LiteLLM does its half in both directions — it puts the
    blocks on the response and reads them straight back off the assistant
    message it is given — but ``LLMResponse`` has no field to carry one and
    the kernel rebuilds assistant messages from ``{role, content,
    tool_calls}``, so they are discarded and cannot be returned. Until that
    round trip exists, a Claude profile that uses tools wants
    ``"reasoning_effort": "none"`` or ``null``.

*Sampling* — all 6/6, the safest things to set

``temperature``   float, usually 0.0-2.0. Lower is more deterministic.
``top_p``         float 0.0-1.0. Nucleus sampling; prefer one or the other.

*Output shape* — all 6/6

``max_tokens`` / ``max_completion_tokens``
    int. Caps the reply, not the context. Note the kernel does its own
    context accounting from ``llm_context_size`` and knows nothing about
    this, so a cap low enough to truncate answers looks like a bad model.
``response_format``   e.g. ``{"type": "json_object"}``. Second Brain drives
    tool calls, so this is rarely what you want.
``parallel_tool_calls``  bool. Whether the model may request several tools in
    one turn.

*Widely but not universally supported*

``seed``               4/6 (not Anthropic, not Gemini). Reproducibility.
``stop``               5/6 (not OpenAI here). Array of stop sequences.
``n``                  5/6. Leave at 1; the loop reads one choice.
``frequency_penalty`` / ``presence_penalty``   4/6.
``logit_bias``         3/6.
``verbosity``          1/6, OpenAI only.

CHECKING A MODEL RATHER THAN GUESSING
--------------------------------------

::

    import litellm
    litellm.get_supported_openai_params(model="gpt-5.1",
                                        custom_llm_provider="openai")
    litellm.supports_reasoning(model="gpt-5.1")

Both are cheap, offline lookups against LiteLLM's model map. When this file's
list and that answer disagree, that answer is right.
"""

dependencies_pip = ["litellm", "Pillow"]
lifetime = "persistent"

# The kernel's default per-call deadline is 60s, and this is the one plugin it
# is wrong for. A deadline measures *running* time, discounting only what the
# guest spends waiting on the kernel — but this box waits on a provider's
# socket inside litellm, which is "something the guest chose to do itself" and
# so is charged in full. Streaming does not help: `sdk.llm.delta` is a one-way
# notice, so a box emitting tokens for two minutes accrues two minutes of
# running time and is killed mid-answer, surfacing as
# `box 'llm_..._0' died during '__chat__'`.
#
# 600 is the ceiling `clamp_timeout` will grant. Note the wall-clock
# `watchdog.HARD_CEILING` is also 600 and is *not* declarable, so ten minutes
# is the real limit on one model call however this is set.
timeout = 600
supports_streaming = True
supports_tool_choice = True
native_modalities = ["image", "audio", "video"]
display_name = "LiteLLM (any provider)"

# The native class this file supersedes. Existing profiles say
# ``llm_service_class: "LiteLLMService"``, and installing this package must not
# orphan them — the registry aliases the old name onto this one.
replaces = ["LiteLLMService"]

import base64
import io
import mimetypes
import time

from guest.llm import BaseLLMBackend, LLMResponse, is_context_limit_error

# Providers whose names LiteLLM already recognises as a prefix. Anything else
# with a custom base_url is addressed through the OpenAI-compatible path.
_KNOWN_PROVIDER_PREFIXES = {
    "anthropic", "azure", "bedrock", "cohere", "deepseek", "gemini", "groq",
    "minimax", "mistral", "ollama", "openai", "openrouter", "vertex_ai", "xai",
}

#: Parameters worth giving a friendlier label and shape than their bare name.
#: Not a list of what is offered - that comes from the provider - and not a
#: source of accepted *values*, which are read from litellm's own signature
#: (see ``_value_choices``). ``instead_of`` names other spellings of the same
#: idea, so a provider whose table omits one can be pointed at the one it does
#: take: MiniMax has no ``reasoning_effort`` and does have ``thinking``.
_TUNABLE_PARAMS = [
    {"name": "reasoning_effort", "label": "Reasoning effort",
     "kind": "choice",
     "instead_of": ("thinking", "reasoning", "reasoning_split")},
    {"name": "temperature", "label": "Temperature", "kind": "number",
     "instead_of": ()},
    {"name": "top_p", "label": "Top-p", "kind": "number", "instead_of": ()},
    {"name": "max_tokens", "label": "Max output tokens", "kind": "number",
     "instead_of": ()},
]

#: Supported params that are not *settings* - the kernel fills each of these
#: from somewhere else, and offering them would invite a profile to fight it.
_NOT_A_SETTING = {
    "stream", "stream_options", "tools", "tool_choice", "functions",
    "function_call", "messages", "model", "api_key", "api_base",
    "max_retries", "extra_headers", "n",
}

# Errors that mean "this will fail again the same way" — no retry, no
# reclassification as a context problem.
_DETERMINISTIC_ERRORS = {
    "RateLimitError", "AuthenticationError", "NotFoundError",
    "PermissionDeniedError", "BadRequestError",
}

MAX_IMAGE_PIXELS = 50_000_000
IMAGE_EDGE = 2048
IMAGE_QUALITY = 80


class LiteLLMBackend(BaseLLMBackend):
    """Unified LLM backend via the litellm SDK."""

    def start(self, sdk):
        """Import litellm once and turn its chattiness down.

        The native version also cleared the ``LiteLLM`` logger's handlers,
        which needed ``import logging`` — refused here, and rightly: reaching
        the logging module is reaching the environment. It is also no longer
        necessary. The child redirects its own stdout to stderr precisely
        because plugin code and its libraries print, so a chatty library
        cannot corrupt the wire; what is left is noise in the app log, and
        litellm's own flags handle that.
        """
        import litellm

        litellm.drop_params = True      # unsupported params degrade, not fail
        litellm.telemetry = False
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        self._litellm = litellm
        sdk.log("litellm ready")
        return True

    def stop(self, sdk):
        """Nothing to close: litellm holds no connection of its own."""
        self._litellm = None
        return True

    # ── the call ──────────────────────────────────────────────────────

    def chat(self, sdk, request):
        """Place one call, streaming or not.

        Both paths build the same response shape, so nothing downstream has to
        know which was taken.
        """
        messages = self._with_attachments(sdk, request)
        kwargs = self._provider_kwargs(request)
        model = self._model_name(request)
        started = time.time()

        try:
            if request.stream:
                response = self._stream(sdk, model, messages, request, kwargs)
            else:
                response = self._blocking(model, messages, request, kwargs)
        except Exception as exc:
            # Classified here rather than left to the wrapper: only this file
            # knows what ``ContextWindowExceededError`` is, and getting that
            # right is what makes the kernel compact and retry instead of
            # failing the turn.
            raise self._classified(exc) from exc

        sdk.log(f"litellm answered in {time.time() - started:.2f}s")
        return response

    @staticmethod
    def _without_duplicated_reasoning(content: str, reasoning: str) -> str:
        """Drop an inline ``<think>`` block that merely repeats *reasoning*.

        Some providers send chain-of-thought **twice**: once in its own
        ``reasoning_content`` field and again wrapped in ``<think>…</think>``
        inside the ordinary content. Measured against MiniMax M3, the block's
        inner text is ``reasoning_content`` exactly, modulo surrounding
        whitespace.

        Removing it *by the text we were given* rather than by scanning for
        tags is the point. A tag scan has to trust a boundary marker the model
        generated, and a model that misplaces one costs real prose — that is
        how replies came to arrive as ``"is a big deal"``. Here the provider
        has already told us what the reasoning was, so the cut needs no guess
        and anything the block holds *beyond* that reasoning is answer text
        that survives.

        Abstains unless the block is recognisably the duplicate: with no
        ``reasoning_content`` (a provider that only ever inlines) or a block
        that does not start with it, this returns *content* untouched and
        ``token_stripper`` remains the backstop it was written to be.
        """
        reasoning = (reasoning or "").strip()
        if not reasoning or not content:
            return content
        start = content.find("<think>")
        if start == -1:
            return content
        inner_at = start + len("<think>")
        end = content.find("</think>", inner_at)
        if end == -1:
            return content
        if not content[inner_at:end].strip().startswith(reasoning):
            return content
        # Everything the block held past the reasoning is answer the model put
        # on the wrong side of its own closer. It comes back.
        extra = content[inner_at:end].strip()[len(reasoning):]
        return (content[:start] + extra
                + content[end + len("</think>"):]).lstrip()

    def _blocking(self, model, messages, request, kwargs):
        """One whole answer, at once."""
        raw = self._litellm.completion(
            model=model, messages=messages,
            tools=request.tools or None, **kwargs)
        choice = raw.choices[0]
        prompt_tokens, cached, completion = self._usage(getattr(raw, "usage", None))
        calls = getattr(choice.message, "tool_calls", None) or []
        return LLMResponse(
            content=self._without_duplicated_reasoning(
                choice.message.content or "",
                getattr(choice.message, "reasoning_content", None) or ""),
            tool_calls=[{"id": call.id, "name": call.function.name,
                         "arguments": call.function.arguments}
                        for call in calls],
            prompt_tokens=prompt_tokens, cached_prompt_tokens=cached,
            completion_tokens=completion)

    def _stream(self, sdk, model, messages, request, kwargs):
        """The same answer, pushed as it arrives *and* returned whole.

        The deltas are for the user's eyes; the returned response is what the
        kernel records. There is nothing here that checks whether the user
        cancelled — ``sdk.llm.delta`` is one-way and answers nothing.

        Which means this loop makes no Request that could ever raise
        ``Terminated``: the usual unwind-at-the-next-Request does not apply,
        and a model that degenerates into repeating a token would otherwise
        run until the provider kills it, unstoppably. The kernel ends this
        call by killing the box instead, so the loop simply stops existing
        mid-iteration. Nothing to handle, and nothing to add.
        """
        # drop_params is on, so a provider that rejects stream_options simply
        # degrades to a stream with no usage chunk (prompt_tokens stays None).
        stream = self._litellm.completion(
            model=model, messages=messages, tools=request.tools or None,
            stream=True, stream_options={"include_usage": True}, **kwargs)

        pieces = []
        thinking = []
        calls_by_index = {}
        prompt_tokens = cached = completion = None

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                seen, seen_cached, seen_completion = self._usage(usage)
                # Kept as ``is not None`` rather than ``or``: a completion of
                # zero tokens is a real answer from the provider, and ``or``
                # would discard it and report "never told us" instead.
                if seen is not None:
                    prompt_tokens = seen
                if seen_cached is not None:
                    cached = seen_cached
                if seen_completion is not None:
                    completion = seen_completion
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            # Never streamed and never appended: this is the field the model
            # thinks in, and the kernel wants the answer. Kept only so the
            # returned content can have its inline duplicate cut by exact
            # match instead of by trusting a tag.
            if (think := getattr(delta, "reasoning_content", None)):
                thinking.append(think)
            text = getattr(delta, "content", None)
            if text:
                pieces.append(text)
                sdk.llm.delta(text)
            for call in (getattr(delta, "tool_calls", None) or []):
                index = getattr(call, "index", 0) or 0
                entry = calls_by_index.setdefault(
                    index, {"id": None, "name": None, "arguments": ""})
                entry["id"] = entry["id"] or getattr(call, "id", None)
                function = getattr(call, "function", None)
                if function is not None:
                    entry["name"] = entry["name"] or getattr(function, "name", None)
                    entry["arguments"] += getattr(function, "arguments", None) or ""

        return LLMResponse(
            content=self._without_duplicated_reasoning(
                "".join(pieces), "".join(thinking)),
            tool_calls=[{"id": entry["id"] or f"call_{index}",
                         "name": entry["name"],
                         "arguments": entry["arguments"] or "{}"}
                        for index, entry in sorted(calls_by_index.items())
                        if entry["name"]],
            prompt_tokens=prompt_tokens, cached_prompt_tokens=cached,
            completion_tokens=completion)

    # ── shaping the request ───────────────────────────────────────────

    def _provider_kwargs(self, request):
        """Connection settings plus whatever the caller forwarded.

        ``request.params`` is where an LLM profile's reasoning effort and its
        extra provider parameters arrive — the kernel merges them in as the
        call is placed, in the OpenAI-compatible spelling, and knows nothing
        about which of them this provider takes. It does not have to:
        ``drop_params`` is set in ``start``, so a model with no
        ``reasoning_effort`` sees the param removed rather than the call
        refused. Nothing here needs to grow a table of what supports what.
        """
        kwargs = dict(request.params or {})
        insist = list(kwargs)
        if insist:
            # ``drop_params`` is set in ``start`` and silently discards
            # anything litellm's table does not list for this provider. Every
            # one of these was configured by hand, so discarding one is a
            # setting that lies; that table also has gaps, omitting
            # ``reasoning_effort`` for a provider whose own API documents it.
            # Naming them here overrides the drop, and if the endpoint then
            # objects, its refusal is the answer - a loud no beats a setting
            # that quietly does nothing.
            kwargs["allowed_openai_params"] = insist
        if request.api_key:
            kwargs.setdefault("api_key", request.api_key)
        if request.base_url:
            kwargs.setdefault("api_base", request.base_url)
        return kwargs

    def _model_name(self, request):
        """What to call the model when talking to LiteLLM.

        A custom endpoint with an unfamiliar prefix is assumed to be
        OpenAI-compatible, which is what almost every self-hosted server is.
        """
        return self._litellm_name(request.model_name, request.base_url)

    @staticmethod
    def _litellm_name(name: str, base_url: str) -> str:
        """The prefix rule, as a plain function of its two inputs.

        Split out of ``_model_name`` because discovery asks the same question
        with no request in hand, and the two answers must not drift: what
        ``models`` offers has to be exactly what ``chat`` will later send.
        """
        provider = ""
        if "/" in name:
            provider = name.split("/", 1)[0].lower().replace("-", "_")
        if base_url and provider not in _KNOWN_PROVIDER_PREFIXES:
            return f"openai/{name}"
        return name

    # -- describing what can be configured -----------------------------

    def providers(self, sdk, provider=""):
        """LiteLLM's provider list, or one provider with its endpoint.

        ``endpoint`` is always ``""`` here, and that is a decision rather than
        a gap. LiteLLM keeps no static table of default base URLs; the only
        way to get one is ``get_llm_provider``, and despite the name that is
        not a lookup. For ``github_copilot`` and at least one other it starts
        an **interactive OAuth device-code login** — printing a sign-in code
        and blocking through three sixty-second waits for a human to
        authorize it. Measured, not inferred.

        So this cannot probe, and the reason is not the delay. Somebody
        configuring a model must never be shown a login code for an unrelated
        service they did not ask about: it is indistinguishable from a
        phishing prompt, which the codes themselves warn about.

        Which leaves guessing, and a guessed endpoint is the worse failure:
        it is wrong silently, fails at the first real call, and the error
        blames the model. So this offers the *names* — which is the half that
        stops somebody having to know that MiniMax is spelled ``minimax`` —
        and the URL stays a question, asked once, where a wrong answer is
        visible and editable.
        """
        litellm = self._litellm
        names = sorted((str(getattr(n, "value", n))
                        for n in getattr(litellm, "provider_list", []) or []))
        if provider:
            names = [n for n in names if n == provider] or [provider]
        return [{"id": n, "label": n.replace("_", " ").title(),
                 "endpoint": self._endpoint_for(n) if provider else ""}
                for n in names]

    def _endpoint_for(self, provider):
        """A provider's default base URL, or ``""``.

        ``get_llm_provider`` is the only accurate source and it is genuinely
        expensive: it resolves credentials, may reach the network, and for a
        device-code provider such as ``github_copilot`` it starts a login and
        waits on it. All acceptable for one provider a person just picked, and
        the reason this is never run across the whole list.

        ``ProviderConfigManager`` looks like a cheap alternative and is not:
        ``get_api_base`` inherits OpenAI's default, so it answers
        ``https://api.openai.com/v1`` for DeepSeek, Groq, Mistral, xAI and
        OpenRouter alike - a confidently wrong URL, which is worse than none.
        """
        try:
            _r, _p, _k, base = self._litellm.get_llm_provider(
                model=f"{provider}/probe")
            if base:
                return base
        except Exception:               # noqa: BLE001 - fall through and try
            pass                        # the other source
        return self._declared_base(provider)

    #: What ``get_api_base`` answers for a provider whose config inherits
    #: OpenAI's and never overrode it. Correct for OpenAI and a confident lie
    #: for everyone else, so it is only trusted when the provider really is
    #: OpenAI.
    _OPENAI_DEFAULT = "https://api.openai.com/v1"

    def _declared_base(self, provider):
        """The base URL a provider's own config class declares, if it declares one.

        The second source, and needed because the first has holes exactly
        where it matters: ``get_llm_provider`` answers nothing for MiniMax,
        Anthropic and OpenAI, which is most of what anybody configures. This
        answers those.

        It has the opposite failure, which is why neither is used alone. A
        provider config that inherits OpenAI's and never sets its own base
        returns OpenAI's - so DeepSeek, Groq, Mistral, xAI and OpenRouter all
        claim ``api.openai.com``. That value is therefore refused unless the
        provider is OpenAI, and the four it would have got wrong are already
        answered correctly by the first source.
        """
        try:
            from litellm.types.utils import LlmProviders
            from litellm.utils import ProviderConfigManager

            config = ProviderConfigManager.get_provider_chat_config(
                model="probe", provider=LlmProviders(provider))
            base = config.get_api_base(None) if config is not None else ""
        except Exception:               # noqa: BLE001 - nothing to offer
            return ""
        if not base:
            return ""
        if base == self._OPENAI_DEFAULT and provider != "openai":
            return ""
        return base

    def models(self, sdk, endpoint, api_key, provider="", live=False):
        """What *endpoint* serves, asked of the endpoint itself where possible.

        With ``live`` set, ``GET /v1/models`` comes first: it is the only
        authoritative answer, it is current, and it covers the gateways that
        appear in no table anywhere - which is most of the endpoints anybody
        actually types. Without it the answer is LiteLLM's own index, which
        needs a provider name and knows nothing about gateways.

        Off by default because fetching is egress and the caller decides
        whether it is in a position to be asked for it.

        What comes back is the name to *store* - the provider prefix restored
        (see :meth:`_prefixed`), and nothing else. Notably **not**
        ``_litellm_name``: the ``openai/`` shim that method adds is a fact
        about how this backend dials a custom endpoint, not part of the
        model's identity, and baking it into the stored name would put it in
        front of the user everywhere the profile is listed. ``chat`` applies
        it per call already, and applying it twice is harmless but pointless.
        """
        rows = self._live_models(sdk, endpoint, api_key) if live else []
        if not rows and provider:
            index = getattr(self._litellm, "models_by_provider", {}) or {}
            rows = sorted(str(m) for m in index.get(provider, []) or [])
        out, seen = [], set()
        for raw in rows:
            name = self._prefixed(raw, provider)
            if name in seen:
                continue
            seen.add(name)
            out.append({"name": name, "label": raw})
        return out

    @staticmethod
    def _prefixed(raw: str, provider: str) -> str:
        """Put the provider back on a bare model id.

        An endpoint's own ``/v1/models`` answers in its local vocabulary, and
        MiniMax's is ``MiniMax-M3`` with no prefix. Handing that straight to
        ``_litellm_name`` produces ``openai/MiniMax-M3`` - which routes to the
        OpenAI config and loses everything litellm knows about MiniMax,
        reasoning included. The working string is ``minimax/MiniMax-M3``, and
        the provider is the piece the listing cannot supply because the server
        has no reason to mention its own name.

        So it is added back here, and only here: a name that already carries a
        slash is left alone, because an aggregator's ids are prefixed *for
        their own catalogue* (``deepseek-ai/deepseek-v4-pro``) and that prefix
        is not litellm's. Getting this right is the whole reason ``models``
        returns a name rather than a list of strings to type.

        Both sources need it and neither reliably has it - litellm's own index
        is prefixed for some providers and bare for others.
        """
        if not provider or "/" in raw:
            return raw
        return f"{provider}/{raw}"

    def _live_models(self, sdk, endpoint, api_key):
        """``GET {endpoint}/models``, or ``[]`` if it does not answer one.

        Deliberately forgiving. This runs while somebody is filling in a form,
        so every way it can fail - no endpoint yet, a wrong key, a server that
        does not implement the route, a body shaped differently - has to come
        back as "I cannot say" and let them type the name. Failing loudly here
        would block setup on a listing that was only ever a convenience.
        """
        if not endpoint:
            return []
        url = endpoint.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            answer = sdk.net.http_json(url, headers=headers)
        except Exception as exc:            # noqa: BLE001 - see docstring
            sdk.log(f"no live model list from {url}: {exc}")
            return []
        if (answer or {}).get("status", 0) >= 400:
            return []
        body = (answer or {}).get("body") or {}
        listing = body.get("data") if isinstance(body, dict) else body
        if not isinstance(listing, list):
            return []
        names = []
        for item in listing:
            name = item.get("id") if isinstance(item, dict) else item
            if isinstance(name, str) and name:
                names.append(name)
        return sorted(names)

    def info(self, sdk, model_name, endpoint=""):
        """One model's facts. Currently just its input context window.

        ``max_input_tokens`` is the right field and the two beside it are
        traps: ``max_output_tokens`` caps a reply, and ``max_tokens`` is
        whichever of the two that entry happens to carry. The kernel budgets
        a conversation against this number, so an output cap here would make
        it compact a million-token model at eight thousand.

        An unknown model raises out of ``get_model_info`` and comes back as
        ``[]`` rather than a guess: the kernel treats 0 as "compact
        reactively", which copes, while a wrong number does not.
        """
        try:
            found = self._litellm.get_model_info(
                model=self._litellm_name(model_name, endpoint))
        except Exception as exc:        # noqa: BLE001 - model not in the map
            sdk.log(f"no model info for {model_name}: {exc}")
            return []
        window = (found or {}).get("max_input_tokens")
        return [{"context_size": int(window)}] if window else []

    def params(self, sdk, model_name, endpoint):
        """Which extra params this model takes, reported and never enforced.

        The supported list is litellm's own, and it is the same list
        ``drop_params`` filters against - so a ``False`` here is an exact
        prediction that the value will be discarded rather than a guess.

        Exact is not the same as right. That table has gaps: a provider can
        document a parameter its litellm config does not list, and then this
        reports unsupported for something that works. So the note says what
        will *happen to the value*, never what the model can do, and it names
        the spelling that does get through when there is one. Whoever reads it
        can then overrule this method, which is the point of it being a
        report.
        """
        litellm = self._litellm
        model = self._litellm_name(model_name, endpoint)
        try:
            resolved, provider, _k, _b = litellm.get_llm_provider(model=model)
            supported = set(litellm.get_supported_openai_params(
                model=resolved, custom_llm_provider=provider) or [])
        except Exception as exc:            # noqa: BLE001 - unknown model
            sdk.log(f"no parameter list for {model}: {exc}")
            return []

        out = []
        for spec in _TUNABLE_PARAMS:
            name = spec["name"]
            ok = name in supported
            note = ""
            if not ok:
                swap = [alt for alt in spec.get("instead_of", ())
                        if alt in supported]
                note = (f"this backend discards it for {provider}; "
                        f"try {swap[0]!r} instead" if swap else
                        f"this backend discards it for {provider}")
            out.append({**{k: v for k, v in spec.items()
                           if k != "instead_of"},
                        "choices": self._value_choices(name),
                        "supported": ok, "note": note})
        # Anything else the provider takes that is worth offering by name.
        known = {spec["name"] for spec in _TUNABLE_PARAMS}
        for name in sorted(supported - known - _NOT_A_SETTING):
            choices = self._value_choices(name)
            out.append({"name": name, "label": name.replace("_", " "),
                        "kind": "choice" if choices else "text",
                        "choices": choices, "supported": True, "note": ""})
        return out

    def _value_choices(self, name):
        """The accepted values for a parameter, read off litellm's signature.

        ``litellm.completion`` annotates a handful of parameters with a
        ``Literal``, which is the only machine-readable statement of accepted
        *values* anywhere in the library - three of forty-odd today
        (``reasoning_effort``, ``verbosity``, ``modalities``). Reading them
        beats a table here for the usual reason: a hardcoded one drifts, and
        this one was already wrong, missing ``none``, ``xhigh`` and
        ``default``.

        It is the union litellm accepts rather than what one model does, so it
        is a picker's contents and not a claim about the provider - the same
        standing every other answer from ``params`` has.
        """
        import inspect
        import typing

        try:
            annotation = inspect.signature(
                self._litellm.completion).parameters[name].annotation
        except (KeyError, TypeError, ValueError):
            return []
        found = []
        # ``Literal['a', 'b'] | None`` and ``List[Literal[...]]`` both appear,
        # so the whole tree is walked rather than the top level matched.
        stack = [annotation]
        while stack:
            node = stack.pop()
            if typing.get_origin(node) is typing.Literal:
                found.extend(str(arg) for arg in typing.get_args(node))
            else:
                stack.extend(typing.get_args(node))
        return sorted(set(found), key=found.index) if found else []

    def _usage(self, usage):
        """``(prompt_tokens, cached_prompt_tokens, completion_tokens)``.

        All three are the provider's own counts, lifted from the ``usage``
        block it returns. Nothing here tokenises anything: only the provider
        knows how it serialised the chat template and the tool schemas, so its
        number is the billable one and a local estimate would merely be a
        second opinion nobody charges by.

        ``cached_prompt_tokens`` is the discounted *share of*
        ``prompt_tokens``, not an addition to it.
        """
        if not usage:
            return None, None, None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        if not details:
            return prompt_tokens, None, completion_tokens
        cached = (details.get("cached_tokens") if isinstance(details, dict)
                  else getattr(details, "cached_tokens", None))
        return prompt_tokens, cached, completion_tokens

    def _classified(self, exc):
        """Re-raise with a code the kernel acts on.

        A context overflow must be recognised: it is the difference between
        the kernel compacting the conversation and retrying, and the turn
        simply failing.
        """
        from guest.llm import LLMProviderError, extract_llm_error_text

        name = type(exc).__name__
        text = extract_llm_error_text(exc)
        if name == "ContextWindowExceededError":
            return LLMProviderError(text, code="context_limit")
        if name in _DETERMINISTIC_ERRORS:
            return LLMProviderError(text, code="provider_error")
        code = "context_limit" if is_context_limit_error(exc) else "provider_error"
        return LLMProviderError(text, code=code)

    # ── attachments ───────────────────────────────────────────────────

    def _with_attachments(self, sdk, request):
        """Inline the media the kernel already decided this model can read.

        Routing happened kernel-side, so everything here is meant to go on the
        wire natively — there is no capability check to repeat. What can still
        fail is the *encoding*: an unreadable file, a mislabelled extension.
        Those degrade to a text pointer rather than losing the attachment
        silently.
        """
        if not request.attachments:
            return request.messages

        blocks, labels, fallbacks = [], [], []
        for item in request.attachments:
            try:
                block = self._block(sdk, item)
            except Exception as exc:
                sdk.log(f"could not inline {item.get('file_name')}: {exc}",
                        level="warning")
                block = None
            if block is None:
                fallbacks.append(self._pointer(item))
                continue
            blocks.append(block)
            labels.append(f"<{item['modality'].title()} {len(labels) + 1}: "
                          f"{item['file_name']}>")

        if not blocks and not fallbacks:
            return request.messages

        parts = []
        if labels:
            parts.append("The following native attachments are provided:\n"
                         + "\n".join(labels))
        if fallbacks:
            parts.append("\n\n".join(fallbacks))
        return self._append(request.messages, "\n\n".join(parts), blocks)

    def _append(self, messages, note, blocks):
        """Add the note and blocks to the last user message."""
        out = [dict(message) for message in messages]
        for index in range(len(out) - 1, -1, -1):
            if out[index].get("role") != "user":
                continue
            content = out[index].get("content")
            if isinstance(content, list):
                out[index]["content"] = [*content,
                                         {"type": "text", "text": note},
                                         *blocks]
            else:
                out[index]["content"] = [
                    {"type": "text",
                     "text": f"{content or ''}\n\n{note}".strip()},
                    *blocks]
            break
        return out

    def _block(self, sdk, item):
        """One provider content block, or None if it cannot be built."""
        modality = item.get("modality")
        path = item.get("path") or ""
        if modality == "image":
            url = self._image_data_url(sdk, path)
            return {"type": "image_url", "image_url": {"url": url}} if url else None
        if modality == "audio":
            if not self._mime(path).startswith("audio/"):
                return None
            suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
            return {"type": "input_audio", "input_audio": {
                "data": self._b64(sdk, path), "format": suffix}}
        if modality == "video":
            if not self._mime(path).startswith("video/"):
                return None
            return {"type": "video_url", "video_url": {
                "url": f"data:{self._mime(path)};base64,{self._b64(sdk, path)}"}}
        return None

    def _pointer(self, item):
        """The text stand-in for an attachment that could not be inlined."""
        parsed = (item.get("parsed_text") or "").strip()
        name = item.get("file_name") or "attachment"
        if parsed:
            return (f"The user attached a {item.get('modality', 'file')} file "
                    f"({name}). Parsed contents:\n{parsed}")
        return (f"The user attached a file: {name}. It has been saved into "
                f"{item.get('path', '')}.")

    def _mime(self, path):
        """Best-guess mime type from the extension."""
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    def _b64(self, sdk, path):
        """File contents, base64 for a data URL.

        ``sdk.fs.read_bytes`` and never ``open``: the bytes have to come
        through a Request, and ``sdk.fs.read`` would decode them as UTF-8 with
        replacement and hand back a mangled file.
        """
        return base64.b64encode(sdk.fs.read_bytes(path)).decode("ascii")

    def _image_data_url(self, sdk, path):
        """Re-encode an image small enough to send.

        Providers reject very large images and bill by pixel, so this shrinks
        to a bounded edge and re-encodes as JPEG rather than forwarding a
        20-megapixel PNG.
        """
        from PIL import Image, ImageFile

        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image = None
        try:
            image = Image.open(io.BytesIO(sdk.fs.read_bytes(path)))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((IMAGE_EDGE, IMAGE_EDGE), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=IMAGE_QUALITY,
                       optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        finally:
            if image is not None:
                image.close()
