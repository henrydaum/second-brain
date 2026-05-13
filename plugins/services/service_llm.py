"""Service plugin for LLM."""

from pathlib import Path
from dataclasses import dataclass, field
import os
import logging
import time
import json

from plugins.BaseService import BaseService

logger = logging.getLogger("LLMClass")

"""This is the one plugin which is truly required for the application to run. All other plugins are technically optional."""


_CONTEXT_LIMIT_HINTS = (
    "context window",
    "context length",
    "context_length",
    "maximum context",
    "max context",
    "too many tokens",
    "too long",
    "max_tokens",
    "prompt is too long",
    "prompt tokens",
    "exceeds limit",
    "exceeds limits",
    "exceeded limit",
    "token limit",
    "request too large",
)

_CONTEXT_RELATED_TERMS = ("context", "token", "prompt", "input")
_LIMIT_RELATED_TERMS = ("limit", "limits", "length", "maximum", "max", "too long", "too many", "exceed", "exceeds", "exceeded")
_NON_CONTEXT_LIMIT_HINTS = (
    "not support model",
    "not supported model",
    "current token plan",
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
    """Flatten provider exceptions and response payloads into one text blob."""
    if error is None:
        return ""

    parts = [str(error)]
    for attr in ("message", "body", "response", "error", "errors"):
        value = getattr(error, attr, None)
        text = _stringify_error_detail(value)
        if text:
            parts.append(text)

    return " | ".join(part for part in parts if part)


def is_context_limit_error(error) -> bool:
    """Heuristic classifier for provider-specific context overflow failures."""
    text = extract_llm_error_text(error).lower()
    if not text:
        return False

    if any(hint in text for hint in _NON_CONTEXT_LIMIT_HINTS):
        return False

    if any(hint in text for hint in _CONTEXT_LIMIT_HINTS):
        return True

    has_context_signal = any(term in text for term in _CONTEXT_RELATED_TERMS)
    has_limit_signal = any(term in text for term in _LIMIT_RELATED_TERMS)
    if has_context_signal and has_limit_signal:
        return True

    if "invalid params" in text and "window" in text and "limit" in text:
        return True

    return False


class LLMProviderError(RuntimeError):
    """Structured provider error surfaced from a model backend."""

    def __init__(self, message: str, code: str = "provider_error"):
        """Initialize the llmprovider error."""
        super().__init__(message)
        self.code = code


@dataclass
class LLMResponse:
    """
    Standardized response from invoke() and chat_with_tools().
    content is always populated. tool_calls is populated when
    the model wants to call tools instead of (or before) answering.
    """
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    # Each tool_call dict: {"id": str, "name": str, "arguments": str (JSON)}
    prompt_tokens: int | None = None   # tokens used by the prompt in this call
    cached_prompt_tokens: int | None = None
    error: str | None = None
    error_code: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Return whether tool calls."""
        return len(self.tool_calls) > 0

    @property
    def is_error(self) -> bool:
        """Return whether error."""
        return bool(self.error)

    @property
    def is_context_limit_error(self) -> bool:
        """Return whether context limit error."""
        if self.error_code == "context_limit":
            return True
        return is_context_limit_error(self.error or self.content)


class BaseLLM(BaseService):
    """
    Abstract base class for Large Language Models.

    All methods use the standard OpenAI messages format:
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

    Subclasses convert to their native format internally.
    """

    def __init__(self):
        """Initialize the base LLM."""
        super().__init__()
        self.shared = True  # LLM clients are typically thread-safe
        # Generalized capability dict so new modalities (image, audio,
        # video, ...) can be added without touching call sites. None
        # means "unknown" — treated as False for routing.
        self.capabilities: dict[str, bool | None] = {
            "image": None,
            "audio": None,
            "video": None,
        }
        self.context_size = None  # Max context window in tokens (auto-detected or from config)

    def has_capability(self, modality: str) -> bool:
        """Return whether capability."""
        return bool(self.capabilities.get(modality))

    def _load(self):
        """Internal helper to load base LLM."""
        raise NotImplementedError

    def unload(self):
        """Handle unload."""
        raise NotImplementedError

    def invoke(self, messages: list[dict], attachments=None, **kwargs) -> LLMResponse:
        """Send messages and return a complete response."""
        raise NotImplementedError

    def stream(self, messages: list[dict], attachments=None, **kwargs):
        """Send messages and yield response chunks."""
        raise NotImplementedError

    def chat_with_tools(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> LLMResponse:
        """Send messages with tool schemas. Returns tool calls or text."""
        raise NotImplementedError

    # =================================================================
    # ATTACHMENT ROUTING
    # =================================================================

    def _resolve_attachments(self, messages: list[dict], attachments) -> tuple[list[dict], list[str]]:
        """Apply the 3-tier attachment routing for this LLM's capabilities.

        Returns ``(messages, native_paths)``:
        - ``messages``: a copy of the input with the suffix appended to
          the last user message (only if a suffix was produced).
        - ``native_paths``: file paths the caller should inline using its
          provider-native image/audio/video plumbing.
        """
        if attachments is None:
            return messages, []
        from attachments.attachment import AttachmentBundle
        bundle = attachments if isinstance(attachments, AttachmentBundle) else AttachmentBundle.from_iterable(attachments)
        if not bundle:
            return messages, []
        native_paths, suffix = bundle.for_llm(self.capabilities)
        if not suffix:
            return messages, native_paths
        out = [m.copy() for m in messages]
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                content = out[i].get("content")
                # content can be a str (most cases) or a list of OpenAI
                # content blocks (when the caller pre-built blocks). Both
                # are handled.
                if isinstance(content, list):
                    out[i]["content"] = content + [{"type": "text", "text": "\n\n" + suffix}]
                else:
                    out[i]["content"] = (str(content or "") + "\n\n" + suffix).strip()
                break
        return out, native_paths

    # =================================================================
    # SHARED IMAGE UTILITIES
    # =================================================================

    @staticmethod
    def get_image_bytes(path: str) -> bytes | None:
        """Convert an image to JPEG bytes, resized to a safe max resolution."""
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
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to process image {path}: {e}")
            return None
        finally:
            if img:
                img.close()


# =====================================================================
# LM STUDIO (Native SDK)
#
# Uses the lmstudio Python SDK for model lifecycle and inference.
# Manages VRAM directly — load() loads weights, unload() frees them.
# Does NOT support tool calling (use OpenAILLM pointed at the LM Studio
# OpenAI-compatible endpoint instead — see Developer tab for the URL).
# =====================================================================

class LMStudioLLM(BaseLLM):
    """Lmstudio LLM."""
    def __init__(self, model_name):
        """Initialize the lmstudio LLM."""
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.loaded = False

    def _load(self):
        """Internal helper to load lmstudio LLM."""
        try:
            import lmstudio as lms
            self.model = lms.llm(self.model_name)
            info = self.model.get_info()
            self.capabilities["image"] = bool(getattr(info, "vision", False))
            logger.info(f"Model has vision support: {self.capabilities['image']}")
            # Auto-detect context size
            ctx = getattr(info, "max_context_length", None) or getattr(info, "context_length", None)
            if ctx:
                self.context_size = int(ctx)
                logger.info(f"Context size: {self.context_size} tokens")
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"LM Studio Load Error: {e}")
            return False

    def unload(self):
        """Handle unload."""
        if self.model:
            self.model.unload()
        self.model = None
        self.loaded = False
        logger.info("LM Studio model unloaded.")

    def _messages_to_chat(self, messages: list[dict], image_handles: list = None):
        """
        Convert OpenAI-format messages to an lmstudio Chat object.
        Maps 'system' and 'developer' roles to system messages.
        """
        import lmstudio as lms

        chat = lms.Chat()
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg.get("content", "")

            if role in ("system", "developer"):
                chat.add_system_prompt(content)
            elif role == "assistant":
                chat.add_assistant_response(content)
            elif role == "user":
                # Attach images to the last user message only
                is_last_user = not any(
                    m["role"] == "user" for m in messages[i + 1:]
                )
                if is_last_user and image_handles:
                    chat.add_user_message(content, images=image_handles)
                else:
                    chat.add_user_message(content)

        return chat

    def _prepare_images(self, image_paths: list[str]) -> tuple[list, list[str], list[str]]:
        """
        Process image paths into LM Studio image handles.
        Returns (image_handles, valid_file_names, temp_files_to_delete).
        """
        if not image_paths:
            return [], [], []

        import lmstudio as lms
        import tempfile

        image_handles = []
        valid_file_names = []
        temp_files = []

        for path in image_paths:
            if not os.path.exists(path):
                continue

            image_bytes = self.get_image_bytes(path)
            if not image_bytes:
                continue

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(image_bytes)
                    tmp_path = f.name
                    f.flush()

                image_handles.append(lms.prepare_image(tmp_path))
                valid_file_names.append(os.path.basename(path))
                temp_files.append(tmp_path)
            except Exception as e:
                logger.error(f"Temp file error for {path}: {e}")
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return image_handles, valid_file_names, temp_files

    @staticmethod
    def _annotate_messages_with_images(messages: list[dict], valid_names: list[str]) -> list[dict]:
        """Copy messages and append image references to the last user message."""
        if not valid_names:
            return messages
        messages = [msg.copy() for msg in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                img_ref = "\n".join(f"<Image {j+1}: {n}>" for j, n in enumerate(valid_names))
                messages[i]["content"] += f"\n\nThe following images are provided:\n{img_ref}"
                break
        return messages

    def _cleanup_temp_files(self, temp_files: list[str]):
        """Internal helper to handle cleanup temp files."""
        for f_path in temp_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except Exception as e:
                logger.debug(f"Temp cleanup failed for {f_path}: {e}")

    def invoke(self, messages, attachments=None, **kwargs):
        """Handle invoke."""
        if not self.loaded or not self.model:
            logger.error("Model not loaded. Call load() first.")
            return LLMResponse(
                content="Error: model not loaded",
                error="model not loaded",
                error_code="not_loaded",
            )

        temp_files = []
        try:
            messages, native_paths = self._resolve_attachments(messages, attachments)
            image_handles, valid_names, temp_files = self._prepare_images(native_paths)
            messages = self._annotate_messages_with_images(messages, valid_names)

            chat = self._messages_to_chat(messages, image_handles)
            config = {}
            if "temperature" in kwargs:
                config["temperature"] = kwargs["temperature"]

            logger.debug(
                f"LM Studio invoke: {len(messages)} messages, {len(valid_names)} images"
            )
            t0 = time.time()
            response = self.model.respond(chat, config=config if config else None)
            logger.debug(f"LM Studio responded in {time.time() - t0:.2f}s")
            return LLMResponse(content=response.content)
        except Exception as e:
            message = extract_llm_error_text(e)
            logger.error(f"LM Studio Invoke Error: {message}")
            if is_context_limit_error(e):
                raise LLMProviderError(message, code="context_limit") from e
            return LLMResponse(
                content=f"Error: {message}",
                error=message,
                error_code="provider_error",
            )
        finally:
            if temp_files:
                time.sleep(0.1)
                self._cleanup_temp_files(temp_files)

    def stream(self, messages, attachments=None, **kwargs):
        """Handle stream."""
        if not self.loaded or not self.model:
            logger.error("Model not loaded. Call load() first.")
            return

        temp_files = []
        try:
            messages, native_paths = self._resolve_attachments(messages, attachments)
            image_handles, valid_names, temp_files = self._prepare_images(native_paths)
            messages = self._annotate_messages_with_images(messages, valid_names)

            chat = self._messages_to_chat(messages, image_handles)
            config = {}
            if "temperature" in kwargs:
                config["temperature"] = kwargs["temperature"]

            for fragment in self.model.respond_stream(chat, config=config if config else None):
                yield fragment.content
        except Exception as e:
            logger.error(f"LM Studio Stream Error: {e}")
        finally:
            if temp_files:
                time.sleep(0.1)
                self._cleanup_temp_files(temp_files)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        """
        LM Studio tool calling has not been implemented.
        Falls back to a plain invoke() call, ignoring any tools.
        For tool calling with LM Studio models, use OpenAILLM pointed
        at the LM Studio OpenAI-compatible endpoint (see Developer tab).
        """
        if tools and not getattr(self, "_tools_warning_logged", False):
            logger.warning("LM Studio tool calling has not been implemented — tools will be ignored. For tool support, use OpenAILLM.")
            self._tools_warning_logged = True
        # Strip tool-related kwargs that invoke() doesn't understand
        kwargs.pop("tools", None)
        return self.invoke(messages, **kwargs)


# =====================================================================
# LITELLM (Unified AI gateway — 100+ providers)
#
# Routes through litellm.completion() which normalizes all providers
# (OpenAI, Anthropic, Google, AWS Bedrock, Azure, Groq, Mistral, etc.)
# into a single OpenAI-compatible interface.
# =====================================================================


class LiteLLMLLM(BaseLLM):
    """LLM backed by the LiteLLM SDK, routing to any of 100+ providers."""

    def __init__(self, model_name, api_key=None, base_url=None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.loaded = False
        m = (model_name or "").lower()
        if any(s in m for s in ("gpt-4o", "gpt-4.1", "gpt-5", "claude-3", "claude-4", "gemini")):
            self.capabilities["image"] = True

    def _load(self):
        try:
            import litellm
            self.loaded = True
            return True
        except ImportError:
            logger.error("litellm is not installed. Install with: pip install 'litellm>=1.60,<1.85'")
            return False

    def unload(self):
        self.loaded = False
        logger.info("LiteLLM model unloaded.")

    def invoke(self, messages, attachments=None, **kwargs):
        if not self.loaded:
            return LLMResponse(content="Error: model not loaded", error="model not loaded", error_code="not_loaded")

        try:
            import litellm

            messages, native_paths = self._resolve_attachments(messages, attachments)
            messages = self._inject_images(messages, native_paths)

            has_tools = "tools" in kwargs and kwargs["tools"]
            logger.debug(f"LiteLLM invoke: {len(messages)} messages, tools={'yes' if has_tools else 'no'}, model={self.model_name}")

            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "drop_params": True,
                **kwargs,
            }
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            if self.base_url:
                completion_kwargs["api_base"] = self.base_url

            t0 = time.time()
            response = litellm.completion(**completion_kwargs)
            logger.debug(f"LiteLLM responded in {time.time() - t0:.2f}s")

            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
            cached_tok = _cached_prompt_tokens(usage)

            if choice.message.tool_calls:
                return LLMResponse(
                    content=choice.message.content or "",
                    tool_calls=[
                        {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in choice.message.tool_calls
                    ],
                    prompt_tokens=prompt_tok,
                    cached_prompt_tokens=cached_tok,
                )

            return LLMResponse(content=choice.message.content or "", prompt_tokens=prompt_tok, cached_prompt_tokens=cached_tok)

        except Exception as e:
            message = extract_llm_error_text(e)
            logger.error(f"LiteLLM Invoke Error: {message}")
            # Check for rate limit / auth errors by class name BEFORE the
            # heuristic, because litellm rate-limit messages contain "tokens"
            # + "limit" which would false-positive as context overflow.
            exc_name = type(e).__name__
            if exc_name in ("RateLimitError", "AuthenticationError", "NotFoundError"):
                return LLMResponse(content=f"Error: {message}", error=message, error_code="provider_error")
            if is_context_limit_error(e):
                raise LLMProviderError(message, code="context_limit") from e
            return LLMResponse(content=f"Error: {message}", error=message, error_code="provider_error")

    def stream(self, messages, attachments=None, **kwargs):
        if not self.loaded:
            return

        try:
            import litellm

            messages, native_paths = self._resolve_attachments(messages, attachments)
            messages = self._inject_images(messages, native_paths)

            completion_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "drop_params": True,
                **kwargs,
            }
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            if self.base_url:
                completion_kwargs["api_base"] = self.base_url

            response = litellm.completion(**completion_kwargs)
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error(f"LiteLLM Stream Error: {e}")

    def chat_with_tools(self, messages, tools=None, **kwargs):
        if tools:
            kwargs["tools"] = tools
        return self.invoke(messages, **kwargs)

    def _inject_images(self, messages: list[dict], image_paths: list[str]) -> list[dict]:
        """Inject base64 images into the last user message (reuses OpenAILLM pattern)."""
        import base64

        if not image_paths:
            return messages

        image_blocks = []
        valid_names = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            image_bytes = self.get_image_bytes(path)
            if not image_bytes:
                continue
            try:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                valid_names.append(Path(path).name)
            except Exception as e:
                logger.error(f"Failed to encode image {path}: {e}")

        if not image_blocks:
            return messages

        messages = [msg.copy() for msg in messages]
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                original_content = messages[i]["content"]
                img_ref = "\n".join(f"<Image {j+1}: {n}>" for j, n in enumerate(valid_names))
                text = f"{original_content}\n\nThe following images are provided:\n{img_ref}"
                messages[i]["content"] = [{"type": "text", "text": text}, *image_blocks]
                break

        return messages


# =====================================================================
# OPENAI-COMPATIBLE (OpenAI, LM Studio via endpoint, Ollama, vLLM, etc.)
#
# Uses the OpenAI Python SDK. Lightweight — no model lifecycle management.
# load() creates the client, unload() releases it.
# Supports tool calling natively.
# =====================================================================

def _cached_prompt_tokens(usage) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return (details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)) if details else None


class OpenAILLM(BaseLLM):
    """Open aillm."""
    def __init__(self, model_name, api_key=None, base_url=None, prompt_cache_key=None, prompt_cache_retention=None):
        """Initialize the open aillm."""
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_retention = prompt_cache_retention
        self.client = None
        self.loaded = False
        # Best-effort capability inference from the model name. Users can
        # override by editing ``capabilities`` after construction.
        m = (model_name or "").lower()
        if any(s in m for s in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "claude-3", "claude-4", "gemini")):
            self.capabilities["image"] = True
        if "gpt-4o" in m or "gpt-5" in m or "audio" in m:
            self.capabilities["audio"] = True

    def _load(self):
        """Internal helper to load open aillm."""
        try:
            import openai

            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = openai.OpenAI(**client_kwargs)
            self.loaded = True
            return True
        except Exception as e:
            logger.error(f"OpenAI Load Error: {e}")
            return False

    def unload(self):
        """Handle unload."""
        self.client = None
        self.loaded = False
        logger.info("OpenAI model unloaded.")

    def _inject_images(self, messages: list[dict], image_paths: list[str]) -> list[dict]:
        """Inject base64 images into the last user message."""
        import base64

        if not image_paths:
            return messages

        image_blocks = []
        valid_names = []
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Image not found, skipping: {path}")
                continue

            image_bytes = self.get_image_bytes(path)
            if not image_bytes:
                continue

            try:
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
                valid_names.append(Path(path).name)
            except Exception as e:
                logger.error(f"Failed to encode image {path}: {e}")

        if not image_blocks:
            return messages

        # Copy messages and convert the last user message to content blocks
        messages = [msg.copy() for msg in messages]

        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                original_content = messages[i]["content"]
                img_ref = "\n".join(f"<Image {j+1}: {n}>" for j, n in enumerate(valid_names))
                text = f"{original_content}\n\nThe following images are provided:\n{img_ref}"

                messages[i]["content"] = [
                    {"type": "text", "text": text},
                    *image_blocks,
                ]
                break

        return messages

    def invoke(self, messages, attachments=None, **kwargs):
        """Handle invoke."""
        if not self.loaded or not self.client:
            logger.error("Model not loaded. Call load() first.")
            return LLMResponse(
                content="Error: model not loaded",
                error="model not loaded",
                error_code="not_loaded",
            )

        try:
            messages, native_paths = self._resolve_attachments(messages, attachments)
            messages = self._inject_images(messages, native_paths)

            has_tools = "tools" in kwargs and kwargs["tools"]
            logger.debug(
                f"OpenAI invoke: {len(messages)} messages, "
                f"tools={'yes' if has_tools else 'no'}, model={self.model_name}"
            )
            t0 = time.time()
            kwargs = self._cache_kwargs(kwargs)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            logger.debug(f"OpenAI responded in {time.time() - t0:.2f}s")
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
            cached_tok = _cached_prompt_tokens(usage)
            if cached_tok:
                logger.debug(f"OpenAI prompt cache hit: {cached_tok}/{prompt_tok} prompt tokens")

            if choice.message.tool_calls:
                return LLMResponse(
                    content=choice.message.content or "",
                    tool_calls=[
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        for tc in choice.message.tool_calls
                    ],
                    prompt_tokens=prompt_tok,
                    cached_prompt_tokens=cached_tok,
                )

            return LLMResponse(content=choice.message.content or "", prompt_tokens=prompt_tok, cached_prompt_tokens=cached_tok)

        except Exception as e:
            message = extract_llm_error_text(e)
            logger.error(f"OpenAI Invoke Error: {message}")
            if is_context_limit_error(e):
                raise LLMProviderError(message, code="context_limit") from e
            return LLMResponse(
                content=f"Error: {message}",
                error=message,
                error_code="provider_error",
            )

    def stream(self, messages, attachments=None, **kwargs):
        """Handle stream."""
        if not self.loaded or not self.client:
            logger.error("Model not loaded. Call load() first.")
            return

        try:
            messages, native_paths = self._resolve_attachments(messages, attachments)
            messages = self._inject_images(messages, native_paths)

            kwargs = self._cache_kwargs(kwargs)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI Stream Error: {e}")

    def chat_with_tools(self, messages, tools=None, **kwargs):
        """Convenience alias — invoke() already handles tools via kwargs."""
        if tools:
            kwargs["tools"] = tools
        return self.invoke(messages, **kwargs)

    def _cache_kwargs(self, kwargs: dict) -> dict:
        kwargs = dict(kwargs)
        if self.prompt_cache_key:
            kwargs.setdefault("prompt_cache_key", self.prompt_cache_key)
        if self.prompt_cache_retention:
            kwargs.setdefault("prompt_cache_retention", self.prompt_cache_retention)
        return kwargs


def _build_llm_from_profile(model_name: str, profile: dict) -> BaseLLM:
    """Instantiate an LLM from a profile config dict (does NOT load it).

    The profile dict carries connection metadata only; the model name is
    the dict key in ``llm_profiles`` and is passed in separately.
    """
    cls_name = profile.get("llm_service_class", "OpenAILLM")

    if cls_name == "LMStudioLLM":
        return LMStudioLLM(model_name)

    if cls_name == "LiteLLMLLM":
        api_key = profile.get("llm_api_key", "")
        resolved_key = os.environ.get(api_key, api_key) if api_key else None
        base_url = profile.get("llm_endpoint", "") or None
        llm = LiteLLMLLM(model_name, api_key=resolved_key, base_url=base_url)
        ctx = int(profile.get("llm_context_size", 0))
        if ctx > 0:
            llm.context_size = ctx
        return llm

    api_key = profile.get("llm_api_key", "")
    resolved_key = os.environ.get(api_key, api_key) if api_key else None
    base_url = profile.get("llm_endpoint", "") or None
    llm = OpenAILLM(model_name, api_key=resolved_key, base_url=base_url, prompt_cache_key=profile.get("prompt_cache_key") or None, prompt_cache_retention=profile.get("prompt_cache_retention") or None)
    ctx = int(profile.get("llm_context_size", 0))
    if ctx > 0:
        llm.context_size = ctx
    return llm


# =====================================================================
# LLM ROUTER (Virtual Proxy)
#
# Registered as the "llm" service. Resolves to whichever LLM the user
# has marked as default (config["default_llm_profile"]) and delegates
# all calls to it. Each LLM is registered as its own service keyed by
# model name; this router is just the convenience handle for "the
# default LLM" — the thing tasks and non-agent code talk to.
# =====================================================================


class LLMRouter(BaseLLM):
    """Default-LLM proxy. Resolves and forwards to the LLM marked as
    ``default_llm_profile`` in config; falls back to the first registered
    LLM if the default is missing or unset.
    """

    config_settings = [
        ("LLM Profiles", "llm_profiles",
         "LLM connection configs keyed by model name.",
         {},
         {"type": "json_dict", "hidden": True}),

        ("Default LLM Profile", "default_llm_profile",
         "Model name of the LLM used when an agent profile says 'default'.",
         "",
         {"type": "text", "hidden": True}),
    ]

    def __init__(self, config: dict, services: dict | None = None):
        """Initialize the llmrouter."""
        super().__init__()
        self.config = config
        # ``services`` is the live service registry. Mutations made by
        # ``add_llm``/``remove_llm`` flow straight into the service dict.
        self.services: dict = services if services is not None else {}
        self.model_name = "LLM Router"

    # --- Resolution ---

    def _llm_keys(self) -> list[str]:
        """Service keys that correspond to LLMs in llm_profiles."""
        profiles = self.config.get("llm_profiles", {}) or {}
        return [name for name in profiles if name in self.services]

    def _resolve_default_name(self) -> str | None:
        """Internal helper to resolve default name."""
        configured = self.config.get("default_llm_profile") or ""
        if configured and configured in self.services:
            return configured
        keys = self._llm_keys()
        if configured and keys:
            logger.warning(
                f"default_llm_profile {configured!r} not registered — "
                f"falling back to {keys[0]!r}"
            )
        return keys[0] if keys else None

    @property
    def active(self) -> BaseLLM | None:
        """Return active."""
        name = self._resolve_default_name()
        return self.services.get(name) if name else None

    # --- LLM management ---

    def add_llm(self, model_name: str, profile_config: dict):
        """Register an LLM in the live service registry."""
        self.services[model_name] = _build_llm_from_profile(model_name, profile_config)

    def remove_llm(self, model_name: str) -> str:
        """Remove LLM."""
        llm = self.services.pop(model_name, None)
        if llm and getattr(llm, "loaded", False):
            llm.unload()
        return f"LLM '{model_name}' removed."

    def list_llms(self) -> list[dict]:
        """List llms."""
        profiles = self.config.get("llm_profiles", {}) or {}
        default_name = self.config.get("default_llm_profile") or ""
        result = []
        for model_name, pconf in profiles.items():
            llm = self.services.get(model_name)
            result.append({
                "model_name": model_name,
                "class": pconf.get("llm_service_class", "OpenAILLM"),
                "endpoint": pconf.get("llm_endpoint", ""),
                "context_size": pconf.get("llm_context_size", 0),
                "default": model_name == default_name,
                "loaded": llm.loaded if llm else False,
            })
        return result

    def _mirror_active(self):
        """Copy key attributes from the resolved default LLM."""
        a = self.active
        if a:
            self.capabilities = dict(a.capabilities)
            self.context_size = a.context_size
            self.loaded = a.loaded
            name = self._resolve_default_name() or "?"
            self.model_name = f"{name} ({a.model_name})"
        else:
            self.loaded = False
            self.model_name = "LLM Router (no LLM configured)"

    # --- BaseLLM interface (delegate to default) ---

    def _load(self):
        """Internal helper to load llmrouter."""
        a = self.active
        if a is None:
            logger.warning("No LLMs configured.")
            return False
        result = a.load()
        self._mirror_active()
        return result

    def unload(self):
        """Handle unload."""
        for model_name in self._llm_keys():
            svc = self.services.get(model_name)
            if getattr(svc, "loaded", False):
                svc.unload()
        self.loaded = False
        self.model_name = "LLM Router"
        logger.info("All LLMs unloaded.")

    def invoke(self, messages, attachments=None, **kwargs):
        """Handle invoke."""
        a = self.active
        if not a or not a.loaded:
            return LLMResponse(
                content="Error: no LLM loaded",
                error="no LLM loaded",
                error_code="not_loaded",
            )
        return a.invoke(messages, attachments, **kwargs)

    def stream(self, messages, attachments=None, **kwargs):
        """Handle stream."""
        a = self.active
        if not a or not a.loaded:
            return
        yield from a.stream(messages, attachments, **kwargs)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        """Handle chat with tools."""
        a = self.active
        if not a or not a.loaded:
            return LLMResponse(
                content="Error: no LLM loaded",
                error="no LLM loaded",
                error_code="not_loaded",
            )
        return a.chat_with_tools(messages, tools, **kwargs)


def build_services(config: dict) -> dict:
    """Register one service per LLM (keyed by model name) plus the ``llm``
    router that resolves to the default LLM.
    """

    services: dict = {}
    profiles = config.get("llm_profiles", {}) or {}

    for model_name, pconf in profiles.items():
        services[model_name] = _build_llm_from_profile(model_name, pconf)

    # Pick a default LLM if none is set.
    if not config.get("default_llm_profile") and profiles:
        config["default_llm_profile"] = next(iter(profiles))

    services["llm"] = LLMRouter(config, services)
    return services
