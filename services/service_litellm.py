"""LiteLLM backend for the LLM router."""


dependencies_files = []
dependencies_pip = ['litellm', 'Pillow']

import base64
import logging
import mimetypes
import os
import time
from pathlib import Path

from plugins.services.service_llm import BaseLLM, LLMProviderError, LLMResponse, _cached_prompt_tokens, extract_llm_error_text, is_context_limit_error

logger = logging.getLogger("LLMClass")

_DETERMINISTIC_ERRORS = {"RateLimitError", "AuthenticationError", "NotFoundError", "PermissionDeniedError", "BadRequestError"}
_KNOWN_PROVIDER_PREFIXES = {"anthropic", "azure", "bedrock", "cohere", "deepseek", "gemini", "groq", "minimax", "mistral", "ollama", "openai", "openrouter", "vertex_ai", "xai"}


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
        self.native_attachment_modalities = {"image", "audio", "video"}

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

    def _inject_attachments(self, messages: list[dict], attachments) -> list[dict]:
        blocks, labels = [], []
        fallback = []
        for att in attachments or []:
            path = getattr(att, "path", "")
            if not os.path.exists(path):
                logger.warning(f"Attachment not found, skipping: {path}")
                fallback.append(self._attachment_pointer(att))
                continue
            try:
                block = self._attachment_block(att)
            except Exception as e:
                logger.warning(f"Failed to prepare native attachment {path}: {e}")
                fallback.append(self._attachment_pointer(att))
                continue
            if block:
                blocks.append(block)
                labels.append(f"<{att.modality.title()} {len(labels) + 1}: {att.file_name}>")
            else:
                fallback.append(self._attachment_pointer(att))
        if not blocks and not fallback:
            return messages
        messages = [msg.copy() for msg in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                parts = []
                if labels:
                    parts.append("The following native attachments are provided:\n" + "\n".join(labels))
                if fallback:
                    parts.append("\n\n".join(fallback))
                note = "\n\n".join(parts)
                content = messages[i].get("content")
                messages[i]["content"] = [*content, {"type": "text", "text": note}, *blocks] if isinstance(content, list) else [{"type": "text", "text": f"{content or ''}\n\n{note}".strip()}, *blocks]
                break
        return messages

    def _attachment_pointer(self, att) -> str:
        parsed = (getattr(att, "parsed_text", None) or "").strip()
        if parsed:
            return f"The user attached a {getattr(att, 'modality', 'file')} file ({getattr(att, 'file_name', 'attachment')}). Parsed contents:\n{parsed}"
        return f"The user attached a file: {getattr(att, 'file_name', 'attachment')}. It has been saved into {getattr(att, 'path', '')}."

    def _attachment_block(self, att):
        if att.modality == "image":
            url = self._image_data_url(att.path)
            return {"type": "image_url", "image_url": {"url": url}} if url else None
        if att.modality == "audio":
            if not (mimetypes.guess_type(att.path)[0] or "").startswith("audio/"):
                return None
            return {"type": "input_audio", "input_audio": {"data": base64.b64encode(Path(att.path).read_bytes()).decode("utf-8"), "format": Path(att.path).suffix.lower().lstrip(".")}}
        if att.modality == "video":
            if not (mimetypes.guess_type(att.path)[0] or "").startswith("video/"):
                return None
            return {"type": "video_url", "video_url": {"url": self._data_url(att.path)}}
        return None

    def _image_data_url(self, path: str) -> str | None:
        from PIL import Image, ImageFile
        import io
        Image.MAX_IMAGE_PIXELS = 50_000_000
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = None
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=80, optimize=True)
            return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Failed to process image {path}: {e}")
            return None
        finally:
            if img:
                img.close()

    def _data_url(self, path: str) -> str:
        return f"data:{mimetypes.guess_type(path)[0] or 'application/octet-stream'};base64,{base64.b64encode(Path(path).read_bytes()).decode('utf-8')}"

    def _provider_kwargs(self, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if self.api_key:
            kwargs.setdefault("api_key", self.api_key)
        if self.base_url:
            kwargs.setdefault("api_base", self.base_url)
        return kwargs

    def _litellm_model_name(self) -> str:
        provider = self.model_name.split("/", 1)[0].lower().replace("-", "_") if "/" in self.model_name else ""
        return f"openai/{self.model_name}" if self.base_url and provider not in _KNOWN_PROVIDER_PREFIXES else self.model_name

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
            messages, native_attachments = self._prepare_attachments(messages, attachments)
            messages = self._inject_attachments(messages, native_attachments)
            logger.debug(f"LiteLLM invoke: {len(messages)} messages, tools={'yes' if kwargs.get('tools') else 'no'}, model={self.model_name}")
            t0 = time.time()
            response = litellm.completion(model=self._litellm_model_name(), messages=messages, **self._provider_kwargs(kwargs))
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
            messages, native_attachments = self._prepare_attachments(messages, attachments)
            for chunk in litellm.completion(model=self._litellm_model_name(), messages=self._inject_attachments(messages, native_attachments), stream=True, **self._provider_kwargs(kwargs)):
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as e:
            logger.error(f"LiteLLM Stream Error: {e}")

    def chat_with_tools(self, messages, tools=None, **kwargs):
        if tools:
            kwargs["tools"] = tools
        return self.invoke(messages, **kwargs)
