from pathlib import Path
from dataclasses import dataclass, field
import os
import logging

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
    def __init__(self):
        super().__init__()
        self.shared = True  # LLM clients are typically thread-safe

    def load(self):
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
# Does NOT support tool calling (use OpenAILLM with base_url for that).
# =====================================================================

class LMStudioLLM(BaseLLM):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.vision = None
        self.loaded = False

    def load(self):
        logger.info(f"Loading LM Studio model: {self.model_name}")
        try:
            import lmstudio as lms
            self.model = lms.llm(self.model_name)
            self.vision = self.model.get_info().vision
            logger.info(f"Model has vision support: {self.vision}")
            self.loaded = True
            logger.info("LM Studio model loaded.")
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
                chat.add_system_message(content)
            elif role == "assistant":
                chat.add_assistant_message(content)
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

    def _cleanup_temp_files(self, temp_files: list[str]):
        for f_path in temp_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except Exception:
                logger.debug(f"Temp cleanup failed for {f_path}: {e}")

    def invoke(self, messages, image_paths=None, **kwargs):
        if not self.loaded or not self.model:
            logger.error("Model not loaded. Call load() first.")
            return LLMResponse(content="Error: model not loaded")

        temp_files = []
        try:
            image_handles, valid_names, temp_files = self._prepare_images(image_paths)

            # Append image references to the last user message
            if valid_names:
                messages = [msg.copy() for msg in messages]
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        img_ref = "\n".join(f"<Image {j+1}: {n}>" for j, n in enumerate(valid_names))
                        messages[i]["content"] += f"\n\nThe following images are provided:\n{img_ref}"
                        break

            chat = self._messages_to_chat(messages, image_handles)
            config = {}
            if "temperature" in kwargs:
                config["temperature"] = kwargs["temperature"]

            response = self.model.respond(chat, config=config if config else None)
            return LLMResponse(content=response.content)
        except Exception as e:
            logger.error(f"LM Studio Invoke Error: {e}")
            return LLMResponse(content=f"Error: {e}")
        finally:
            if temp_files:
                import time
                time.sleep(0.1)
                self._cleanup_temp_files(temp_files)

    def stream(self, messages, image_paths=None, **kwargs):
        if not self.loaded or not self.model:
            logger.error("Model not loaded. Call load() first.")
            return

        temp_files = []
        try:
            image_handles, valid_names, temp_files = self._prepare_images(image_paths)

            if valid_names:
                messages = [msg.copy() for msg in messages]
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        img_ref = "\n".join(f"<Image {j+1}: {n}>" for j, n in enumerate(valid_names))
                        messages[i]["content"] += f"\n\nThe following images are provided:\n{img_ref}"
                        break

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
                import time
                time.sleep(0.1)
                self._cleanup_temp_files(temp_files)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        """
        LM Studio native SDK does not support function calling.
        For tool calling with LM Studio models, use OpenAILLM with
        base_url="http://localhost:1234/v1".
        """
        logger.error("LM Studio native SDK does not support tool calling. "
                      "Use OpenAILLM with base_url='http://localhost:1234/v1' instead.")
        return LLMResponse(content="Error: tool calling not supported via LM Studio native SDK")


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

    def load(self):
        try:
            logger.info(f"Loading OpenAI model: {self.model_name}")
            import openai

            client_kwargs = {}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self.client = openai.OpenAI(**client_kwargs)
            self.loaded = True
            logger.info(f"OpenAI model loaded: {self.model_name}" +
                        (f" @ {self.base_url}" if self.base_url else ""))
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

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            choice = response.choices[0]

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
                )

            return LLMResponse(content=choice.message.content or "")

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


def build_services(config: dict) -> dict:
    return {
        "llm": OpenAILLM(
            model_name=config.get("llm_model_name", "gemma-3-4b-it"),
            base_url=config.get("llm_endpoint", "http://localhost:1234/v1"),
        ),
    }