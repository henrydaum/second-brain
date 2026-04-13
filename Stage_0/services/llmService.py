from pathlib import Path
from dataclasses import dataclass, field
import os
import logging
import time

from Stage_0.BaseService import BaseService

logger = logging.getLogger("LLMClass")


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

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


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
        
    config_settings = [
        ("LLM Model Name", "llm_model_name",
         "Model name for the language model API.",
         "gpt-5-mini",
         {"type": "text"}),

        ("LLM Endpoint", "llm_endpoint",
         "Custom API endpoint URL. Leave blank for the default OpenAI endpoint.",
         "",
         {"type": "text"}),

        ("LLM API Key", "llm_api_key",
         "API key or environment variable name for the LLM.",
         "OPENAI_API_KEY",
         {"type": "text"}),

        ("LLM Context Size", "llm_context_size",
         "Max context window in tokens. Auto-detected for LM Studio models. "
         "Set manually for OpenAI-compatible endpoints. When set, the agent "
         "proactively compacts the conversation at 80% usage. When 0, the agent "
         "still compacts reactively when a context-limit error is hit.",
         0,
         {"type": "text"}),
    ]
    
    def __init__(self):
        super().__init__()
        self.shared = True  # LLM clients are typically thread-safe
        self.vision = None  # True/False/None (None = unknown)
        self.context_size = None  # Max context window in tokens (auto-detected or from config)

    def _load(self):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def invoke(self, messages: list[dict], image_paths: list[str] = None, **kwargs) -> LLMResponse:
        """Send messages and return a complete response."""
        raise NotImplementedError

    def stream(self, messages: list[dict], image_paths: list[str] = None, **kwargs):
        """Send messages and yield response chunks."""
        raise NotImplementedError

    def chat_with_tools(self, messages: list[dict], tools: list[dict] = None, **kwargs) -> LLMResponse:
        """Send messages with tool schemas. Returns tool calls or text."""
        raise NotImplementedError

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
        self.vision = None
        self.loaded = False

    def _load(self):
        try:
            import lmstudio as lms
            self.model = lms.llm(self.model_name)
            info = self.model.get_info()
            self.vision = info.vision
            logger.info(f"Model has vision support: {self.vision}")
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

    def invoke(self, messages, image_paths=None, **kwargs):
        if not self.loaded or not self.model:
            logger.error("Model not loaded. Call load() first.")
            return LLMResponse(content="Error: model not loaded")

        temp_files = []
        try:
            image_handles, valid_names, temp_files = self._prepare_images(image_paths)
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
            logger.error(f"LM Studio Invoke Error: {e}")
            return LLMResponse(content=f"Error: {e}")
        finally:
            if temp_files:
                time.sleep(0.1)
                self._cleanup_temp_files(temp_files)

    def stream(self, messages, image_paths=None, **kwargs):
        if not self.loaded or not self.model:
            logger.error("Model not loaded. Call load() first.")
            return

        temp_files = []
        try:
            image_handles, valid_names, temp_files = self._prepare_images(image_paths)
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
        LM Studio native SDK does not support function calling.
        Falls back to a plain invoke() call, ignoring any tools.
        For tool calling with LM Studio models, use OpenAILLM pointed
        at the LM Studio OpenAI-compatible endpoint (see Developer tab).
        """
        if tools and not getattr(self, "_tools_warning_logged", False):
            logger.warning("LM Studio native SDK does not support tool calling — "
                           "tools will be ignored. For tool support, use OpenAILLM "
                           "pointed at the LM Studio OpenAI-compatible endpoint.")
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

    def invoke(self, messages, image_paths=None, **kwargs):
        if not self.loaded or not self.client:
            logger.error("Model not loaded. Call load() first.")
            return LLMResponse(content="Error: model not loaded")

        try:
            messages = self._inject_images(messages, image_paths)

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
            logger.error(f"OpenAI Invoke Error: {e}")
            return LLMResponse(content=f"Error: {e}")

    def stream(self, messages, image_paths=None, **kwargs):
        if not self.loaded or not self.client:
            logger.error("Model not loaded. Call load() first.")
            return

        try:
            messages = self._inject_images(messages, image_paths)

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


def _build_llm_from_profile(profile: dict) -> BaseLLM:
    """Instantiate an LLM from a profile config dict (does NOT load it)."""
    cls_name = profile.get("llm_service_class", "OpenAILLM")
    model = profile.get("llm_model_name", "")

    if cls_name == "LMStudioLLM":
        return LMStudioLLM(model)

    api_key = profile.get("llm_api_key", "")
    resolved_key = os.environ.get(api_key, api_key) if api_key else None
    base_url = profile.get("llm_endpoint", "") or None
    llm = OpenAILLM(model, api_key=resolved_key, base_url=base_url)
    ctx = int(profile.get("llm_context_size", 0))
    if ctx > 0:
        llm.context_size = ctx
    return llm


# =====================================================================
# LLM ROUTER (Virtual Proxy)
#
# Registered as the "llm" service. Delegates all calls to whichever
# underlying LLM profile is currently active. Manages named profiles
# internally so the agent and service registry stay unchanged.
# =====================================================================

class LLMRouter(BaseLLM):

    config_settings = BaseLLM.config_settings + [
        ("LLM Profiles", "llm_profiles",
         "Named LLM configurations. Managed via /model command.",
         {},
         {"type": "json_dict", "hidden": True}),

        ("Active LLM Profile", "active_llm_profile",
         "Name of the currently active LLM profile.",
         "",
         {"type": "text", "hidden": True}),
    ]

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._profiles: dict[str, BaseLLM] = {}
        self._active_name: str | None = None
        self.model_name = "LLM Router"

    @property
    def active(self) -> BaseLLM | None:
        return self._profiles.get(self._active_name)

    # --- Profile management ---

    def add_profile(self, name: str, profile_config: dict):
        """Instantiate an LLM from profile config and store it."""
        self._profiles[name] = _build_llm_from_profile(profile_config)

    def remove_profile(self, name: str) -> str:
        """Unload and remove a profile."""
        llm = self._profiles.pop(name, None)
        if llm and llm.loaded:
            llm.unload()
        if self._active_name == name:
            self._active_name = next(iter(self._profiles), None)
            self._mirror_active()
        return f"Profile '{name}' removed."

    def switch(self, name: str) -> str:
        """Switch the active profile. Loads the new one if not already loaded."""
        if name not in self._profiles:
            return f"Unknown profile: '{name}'"
        self._active_name = name
        llm = self.active
        if not llm.loaded:
            llm.load()
        self._mirror_active()
        return f"Switched to '{name}' ({llm.model_name})."

    def list_profiles(self) -> list[dict]:
        """Return profile info for display."""
        profiles = self.config.get("llm_profiles", {})
        result = []
        for name, pconf in profiles.items():
            llm = self._profiles.get(name)
            result.append({
                "name": name,
                "model": pconf.get("llm_model_name", "?"),
                "class": pconf.get("llm_service_class", "OpenAILLM"),
                "active": name == self._active_name,
                "loaded": llm.loaded if llm else False,
            })
        return result

    def _sync_from_config(self):
        """Read profiles from config, instantiate missing ones, handle migration."""
        profiles = self.config.get("llm_profiles", {})

        # Migration: old flat keys -> single "default" profile
        if not profiles and self.config.get("llm_model_name"):
            profiles = {
                "default": {
                    "llm_model_name": self.config.get("llm_model_name", ""),
                    "llm_endpoint": self.config.get("llm_endpoint", ""),
                    "llm_api_key": self.config.get("llm_api_key", "OPENAI_API_KEY"),
                    "llm_context_size": self.config.get("llm_context_size", 0),
                    "llm_service_class": "OpenAILLM",
                }
            }
            self.config["llm_profiles"] = profiles
            self.config["active_llm_profile"] = "default"

        # Instantiate LLM objects for profiles not yet created
        for name, pconf in profiles.items():
            if name not in self._profiles:
                self._profiles[name] = _build_llm_from_profile(pconf)

        # Remove profiles deleted from config
        for name in list(self._profiles):
            if name not in profiles:
                if self._profiles[name].loaded:
                    self._profiles[name].unload()
                del self._profiles[name]

        # Set active
        active = self.config.get("active_llm_profile", "")
        if active and active in self._profiles:
            self._active_name = active
        elif self._profiles:
            self._active_name = next(iter(self._profiles))

    def _mirror_active(self):
        """Copy key attributes from the active LLM to the router."""
        a = self.active
        if a:
            self.vision = a.vision
            self.context_size = a.context_size
            self.loaded = a.loaded
            self.model_name = f"{self._active_name} ({a.model_name})"
        else:
            self.loaded = False
            self.model_name = "LLM Router (no active profile)"

    # --- BaseLLM interface (delegate to active) ---

    def _load(self):
        self._sync_from_config()
        if self._active_name and self.active:
            result = self.active.load()
            self._mirror_active()
            return result
        logger.warning("No LLM profiles configured.")
        return False

    def unload(self):
        for llm in self._profiles.values():
            if llm.loaded:
                llm.unload()
        self.loaded = False
        self.model_name = "LLM Router"
        logger.info("All LLM profiles unloaded.")

    def invoke(self, messages, image_paths=None, **kwargs):
        if not self.active or not self.active.loaded:
            return LLMResponse(content="Error: no active LLM profile loaded")
        return self.active.invoke(messages, image_paths, **kwargs)

    def stream(self, messages, image_paths=None, **kwargs):
        if not self.active or not self.active.loaded:
            return
        yield from self.active.stream(messages, image_paths, **kwargs)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        if not self.active or not self.active.loaded:
            return LLMResponse(content="Error: no active LLM profile loaded")
        return self.active.chat_with_tools(messages, tools, **kwargs)


def build_services(config: dict) -> dict:
    router = LLMRouter(config)
    return {"llm": router}