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
    error: str | None = None
    error_code: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_error(self) -> bool:
        return bool(self.error)

    @property
    def is_context_limit_error(self) -> bool:
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

    @property
    def vision(self) -> bool | None:
        """Back-compat shim. Reads ``capabilities['image']``."""
        return self.capabilities.get("image")

    @vision.setter
    def vision(self, value: bool | None) -> None:
        self.capabilities["image"] = value

    def has_capability(self, modality: str) -> bool:
        return bool(self.capabilities.get(modality))

    def _load(self):
        raise NotImplementedError

    def unload(self):
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
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.loaded = False

    def _load(self):
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
        for f_path in temp_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except Exception as e:
                logger.debug(f"Temp cleanup failed for {f_path}: {e}")

    def invoke(self, messages, attachments=None, **kwargs):
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
# OPENAI-COMPATIBLE (OpenAI, LM Studio via endpoint, Ollama, vLLM, etc.)
#
# Uses the OpenAI Python SDK. Lightweight — no model lifecycle management.
# load() creates the client, unload() releases it.
# Supports tool calling natively.
# =====================================================================

class OpenAILLM(BaseLLM):
    def __init__(self, model_name, api_key=None, base_url=None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            logger.debug(f"OpenAI responded in {time.time() - t0:.2f}s")
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None

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
                )

            return LLMResponse(content=choice.message.content or "", prompt_tokens=prompt_tok)

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
        if not self.loaded or not self.client:
            logger.error("Model not loaded. Call load() first.")
            return

        try:
            messages, native_paths = self._resolve_attachments(messages, attachments)
            messages = self._inject_images(messages, native_paths)

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


def _build_llm_from_profile(model_name: str, profile: dict) -> BaseLLM:
    """Instantiate an LLM from a profile config dict (does NOT load it).

    The profile dict carries connection metadata only; the model name is
    the dict key in ``llm_profiles`` and is passed in separately.
    """
    cls_name = profile.get("llm_service_class", "OpenAILLM")

    if cls_name == "LMStudioLLM":
        return LMStudioLLM(model_name)

    api_key = profile.get("llm_api_key", "")
    resolved_key = os.environ.get(api_key, api_key) if api_key else None
    base_url = profile.get("llm_endpoint", "") or None
    llm = OpenAILLM(model_name, api_key=resolved_key, base_url=base_url)
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


def _migrate_legacy_llm_config(config: dict) -> bool:
    """One-time migration of the old conflated ``llm_profiles`` shape into
    the new split (LLM connection configs + agent profiles).

    Old shape (per entry):
        {llm_model_name, llm_endpoint, llm_api_key, llm_context_size,
         llm_service_class, [prompt_suffix, whitelist_or_blacklist_tools,
         tools_list]}

    New shape:
        llm_profiles[model_name] = {llm_endpoint, llm_api_key,
                                    llm_context_size, llm_service_class}
        default_llm_profile = "<model_name>"
        agent_profiles[name]  = {llm, prompt_suffix,
                                 whitelist_or_blacklist_tools, tools_list}
        active_agent_profile  = "<name>"

    Returns True if any migration ran. Idempotent — entries already in the
    new shape are left alone.
    """
    profiles = config.get("llm_profiles", {})
    if not isinstance(profiles, dict):
        return False

    legacy_entries: list[tuple[str, dict]] = [
        (name, pconf) for name, pconf in profiles.items()
        if isinstance(pconf, dict) and "llm_model_name" in pconf
    ]

    # Even older flat-keys form — pre-profile migration. Wrap into a single
    # legacy entry first so the rest of the migration handles it uniformly.
    if not profiles and config.get("llm_model_name"):
        legacy_entries = [(
            "default",
            {
                "llm_model_name": config.get("llm_model_name", ""),
                "llm_endpoint": config.get("llm_endpoint", ""),
                "llm_api_key": config.get("llm_api_key", "OPENAI_API_KEY"),
                "llm_context_size": config.get("llm_context_size", 0),
                "llm_service_class": "OpenAILLM",
            },
        )]

    if not legacy_entries:
        return False

    new_llms: dict = {}
    new_agents: dict = config.get("agent_profiles", {}) or {}
    scope_keys = ("prompt_suffix", "whitelist_or_blacklist_tools", "tools_list")
    old_active = config.get("active_llm_profile", "")
    new_default_llm = ""
    new_active_agent = ""

    for name, pconf in legacy_entries:
        model = pconf.get("llm_model_name") or name
        new_llms[model] = {
            "llm_endpoint": pconf.get("llm_endpoint", ""),
            "llm_api_key": pconf.get("llm_api_key", ""),
            "llm_context_size": pconf.get("llm_context_size", 0),
            "llm_service_class": pconf.get("llm_service_class", "OpenAILLM"),
        }
        if name == old_active or not new_default_llm:
            new_default_llm = model

        # If the legacy profile carried scope, materialize an agent profile
        # by the same name pointing at this model.
        if any(pconf.get(k) for k in scope_keys):
            new_agents[name] = {
                "llm": model,
                "prompt_suffix": pconf.get("prompt_suffix", "") or "",
                "whitelist_or_blacklist_tools": pconf.get("whitelist_or_blacklist_tools", "blacklist"),
                "tools_list": pconf.get("tools_list") or [],
            }
            if name == old_active:
                new_active_agent = name

    # Carry over any non-legacy entries verbatim (already in the new shape).
    for name, pconf in profiles.items():
        if isinstance(pconf, dict) and "llm_model_name" not in pconf and name not in new_llms:
            new_llms[name] = pconf

    # Always ensure a default agent profile exists.
    if "default" not in new_agents:
        new_agents["default"] = {
            "llm": "default",
            "prompt_suffix": "",
            "whitelist_or_blacklist_tools": "blacklist",
            "tools_list": [],
        }

    config["llm_profiles"] = new_llms
    config["agent_profiles"] = new_agents
    if new_default_llm:
        config["default_llm_profile"] = new_default_llm
    if not config.get("active_agent_profile"):
        config["active_agent_profile"] = new_active_agent or "default"

    # Drop superseded keys so they don't drift.
    config.pop("active_llm_profile", None)
    for k in ("llm_model_name", "llm_endpoint", "llm_api_key", "llm_context_size"):
        config.pop(k, None)

    logger.info(
        f"Migrated legacy llm_profiles: {len(new_llms)} LLM(s), "
        f"{len(new_agents)} agent profile(s), default LLM = {new_default_llm!r}"
    )
    return True


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
        name = self._resolve_default_name()
        return self.services.get(name) if name else None

    # --- LLM management ---

    def add_llm(self, model_name: str, profile_config: dict):
        """Register an LLM in the live service registry."""
        self.services[model_name] = _build_llm_from_profile(model_name, profile_config)

    def remove_llm(self, model_name: str) -> str:
        llm = self.services.pop(model_name, None)
        if llm and getattr(llm, "loaded", False):
            llm.unload()
        return f"LLM '{model_name}' removed."

    def list_llms(self) -> list[dict]:
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
        a = self.active
        if a is None:
            logger.warning("No LLMs configured.")
            return False
        result = a.load()
        self._mirror_active()
        return result

    def unload(self):
        for model_name in self._llm_keys():
            svc = self.services.get(model_name)
            if getattr(svc, "loaded", False):
                svc.unload()
        self.loaded = False
        self.model_name = "LLM Router"
        logger.info("All LLMs unloaded.")

    def invoke(self, messages, attachments=None, **kwargs):
        a = self.active
        if not a or not a.loaded:
            return LLMResponse(
                content="Error: no LLM loaded",
                error="no LLM loaded",
                error_code="not_loaded",
            )
        return a.invoke(messages, attachments, **kwargs)

    def stream(self, messages, attachments=None, **kwargs):
        a = self.active
        if not a or not a.loaded:
            return
        yield from a.stream(messages, attachments, **kwargs)

    def chat_with_tools(self, messages, tools=None, **kwargs):
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
    _migrate_legacy_llm_config(config)

    services: dict = {}
    profiles = config.get("llm_profiles", {}) or {}

    for model_name, pconf in profiles.items():
        services[model_name] = _build_llm_from_profile(model_name, pconf)

    # Pick a default LLM if none is set.
    if not config.get("default_llm_profile") and profiles:
        config["default_llm_profile"] = next(iter(profiles))

    services["llm"] = LLMRouter(config, services)
    return services
