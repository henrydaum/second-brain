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
"""

dependencies_pip = ["litellm", "Pillow"]
lifetime = "persistent"
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

    def _blocking(self, model, messages, request, kwargs):
        """One whole answer, at once."""
        raw = self._litellm.completion(
            model=model, messages=messages,
            tools=request.tools or None, **kwargs)
        choice = raw.choices[0]
        prompt_tokens, cached = self._usage(getattr(raw, "usage", None))
        calls = getattr(choice.message, "tool_calls", None) or []
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=[{"id": call.id, "name": call.function.name,
                         "arguments": call.function.arguments}
                        for call in calls],
            prompt_tokens=prompt_tokens, cached_prompt_tokens=cached)

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
        calls_by_index = {}
        prompt_tokens = cached = None

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                seen, seen_cached = self._usage(usage)
                prompt_tokens = seen or prompt_tokens
                cached = seen_cached or cached
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
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
            content="".join(pieces),
            tool_calls=[{"id": entry["id"] or f"call_{index}",
                         "name": entry["name"],
                         "arguments": entry["arguments"] or "{}"}
                        for index, entry in sorted(calls_by_index.items())
                        if entry["name"]],
            prompt_tokens=prompt_tokens, cached_prompt_tokens=cached)

    # ── shaping the request ───────────────────────────────────────────

    def _provider_kwargs(self, request):
        """Connection settings plus whatever the caller forwarded."""
        kwargs = dict(request.params or {})
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
        name = request.model_name
        provider = ""
        if "/" in name:
            provider = name.split("/", 1)[0].lower().replace("-", "_")
        if request.base_url and provider not in _KNOWN_PROVIDER_PREFIXES:
            return f"openai/{name}"
        return name

    def _usage(self, usage):
        """``(prompt_tokens, cached_prompt_tokens)`` from a usage object."""
        if not usage:
            return None, None
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        if not details:
            return prompt_tokens, None
        cached = (details.get("cached_tokens") if isinstance(details, dict)
                  else getattr(details, "cached_tokens", None))
        return prompt_tokens, cached

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
