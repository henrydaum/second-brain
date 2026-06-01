"""LiteLLM backend for the LLM router."""

import logging
import os
import time
from pathlib import Path

from plugins.services.service_llm import BaseLLM, LLMProviderError, LLMResponse, _cached_prompt_tokens, extract_llm_error_text, is_context_limit_error

logger = logging.getLogger("LLMClass")

_DETERMINISTIC_ERRORS = {"RateLimitError", "AuthenticationError", "NotFoundError", "PermissionDeniedError", "BadRequestError"}


def _quiet_litellm():
    """Keep LiteLLM's own diagnostics out of the REPL/app log."""
    llm_logger = logging.getLogger("LiteLLM")
    llm_logger.handlers.clear()
    llm_logger.propagate = False
    llm_logger.setLevel(logging.ERROR)


class LiteLLMService(BaseLLM):
    """Unified LLM backend via the litellm SDK."""
    is_llm_backend = True

    def __init__(self, model_name, api_key=None, base_url=None):
        super().__init__()
        self.model_name, self.api_key, self.base_url, self.loaded = model_name, api_key, base_url, False

    def _load(self):
        try:
            _quiet_litellm()
            import litellm
            litellm.drop_params = True
            litellm.telemetry = False
            litellm.set_verbose = False
            litellm.suppress_debug_info = True
            litellm.logging = False
            _quiet_litellm()
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"LiteLLM Load Error: {e}")
            return False

    def unload(self):
        self.loaded = False
        logger.info("LiteLLM unloaded.")

    def _inject_images(self, messages: list[dict], image_paths: list[str]) -> list[dict]:
        import base64
        image_blocks, valid_names = [], []
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Image not found, skipping: {path}")
                continue
            image_bytes = self.get_image_bytes(path)
            if not image_bytes:
                continue
            image_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}"}})
            valid_names.append(Path(path).name)
        if not image_blocks:
            return messages
        messages = [msg.copy() for msg in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                text = f"{messages[i]['content']}\n\nThe following images are provided:\n" + "\n".join(f"<Image {j+1}: {n}>" for j, n in enumerate(valid_names))
                messages[i]["content"] = [{"type": "text", "text": text}, *image_blocks]
                break
        return messages

    def _provider_kwargs(self, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if self.api_key:
            kwargs.setdefault("api_key", self.api_key)
        if self.base_url:
            kwargs.setdefault("api_base", self.base_url)
        return kwargs

    def _classify_error(self, e) -> str:
        if type(e).__name__ == "ContextWindowExceededError":
            return "context_limit"
        if type(e).__name__ in _DETERMINISTIC_ERRORS:
            return "provider_error"
        return "context_limit" if is_context_limit_error(e) else "provider_error"

    def invoke(self, messages, attachments=None, **kwargs):
        if not self.loaded:
            logger.error("LiteLLM not loaded. Call load() first.")
            return LLMResponse(content="Error: model not loaded", error="model not loaded", error_code="not_loaded")
        try:
            _quiet_litellm()
            import litellm
            _quiet_litellm()
            messages, native_paths = self._resolve_attachments(messages, attachments)
            messages = self._inject_images(messages, native_paths)
            logger.debug(f"LiteLLM invoke: {len(messages)} messages, tools={'yes' if kwargs.get('tools') else 'no'}, model={self.model_name}")
            t0 = time.time()
            response = litellm.completion(model=self.model_name, messages=messages, **self._provider_kwargs(kwargs))
            logger.debug(f"LiteLLM responded in {time.time() - t0:.2f}s")
            choice, usage = response.choices[0], getattr(response, "usage", None)
            prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
            cached_tok = _cached_prompt_tokens(usage)
            self.last_prompt_tokens, self.last_cached_prompt_tokens = prompt_tok, cached_tok
            if cached_tok:
                logger.debug(f"LiteLLM prompt cache hit: {cached_tok}/{prompt_tok} prompt tokens")
            calls = getattr(choice.message, "tool_calls", None) or []
            return LLMResponse(content=choice.message.content or "", tool_calls=[{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments} for tc in calls], prompt_tokens=prompt_tok, cached_prompt_tokens=cached_tok)
        except Exception as e:
            message, code = extract_llm_error_text(e), self._classify_error(e)
            logger.error(f"LiteLLM Invoke Error: {message}")
            if code == "context_limit":
                raise LLMProviderError(message, code=code) from e
            return LLMResponse(content=f"Error: {message}", error=message, error_code=code)

    def stream(self, messages, attachments=None, **kwargs):
        if not self.loaded:
            logger.error("LiteLLM not loaded. Call load() first.")
            return
        try:
            _quiet_litellm()
            import litellm
            _quiet_litellm()
            messages, native_paths = self._resolve_attachments(messages, attachments)
            for chunk in litellm.completion(model=self.model_name, messages=self._inject_images(messages, native_paths), stream=True, **self._provider_kwargs(kwargs)):
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as e:
            logger.error(f"LiteLLM Stream Error: {e}")

    def chat_with_tools(self, messages, tools=None, **kwargs):
        if tools:
            kwargs["tools"] = tools
        return self.invoke(messages, **kwargs)
