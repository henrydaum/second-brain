"""Service plugin for LLM."""

from dataclasses import dataclass, field
import importlib
import inspect
import os
import logging
import json
import sys

from plugins.BaseService import BaseService
from plugins.helpers.plugin_paths import PLUGIN_CONFIG

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
        self.last_prompt_tokens = None
        self.last_cached_prompt_tokens = None

    def has_capability(self, modality: str) -> bool:
        """Return whether capability."""
        return bool(self.capabilities.get(modality))

    @classmethod
    def suggested_api_key_env(cls, model_name: str) -> str | None:
        """Return the likely API-key env var for setup prompts, if known."""
        return None

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


def _cached_prompt_tokens(usage) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    return (details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)) if details else None


def _llm_backend_classes() -> dict[str, type[BaseLLM]]:
    backends = {}
    built_dir, sandbox_dir, prefix, namespaces = PLUGIN_CONFIG["service"]
    for directory, namespace in ((built_dir, namespaces[0]), (sandbox_dir, namespaces[1])):
        if not directory.exists():
            continue
        for py_file in sorted(directory.glob(f"{prefix}*.py")):
            if py_file.stem in {"service_llm"} or py_file.stem.startswith("_"):
                continue
            try:
                module = importlib.import_module(namespace.format(stem=py_file.stem)) if namespace.startswith("plugins.") else _load_sandbox_backend(py_file, namespace.format(stem=py_file.stem))
            except Exception as e:
                logger.warning(f"Could not inspect LLM backend {py_file.name}: {e}")
                continue
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module.__name__ and issubclass(cls, BaseLLM) and getattr(cls, "is_llm_backend", False):
                    backends[cls.__name__] = cls
    return backends


def llm_backend_names() -> list[str]:
    return sorted(_llm_backend_classes())


def llm_backend_api_key_hint(class_name: str, model_name: str) -> str:
    cls = _llm_backend_classes().get(class_name or "LiteLLMService")
    env = cls.suggested_api_key_env(model_name) if cls else None
    return f" Suggested env var: `{env}`." if env else ""


def _load_sandbox_backend(path, module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_llm_from_profile(model_name: str, profile: dict) -> BaseLLM:
    """Instantiate an LLM from a profile config dict (does NOT load it).

    The profile dict carries connection metadata only; the model name is
    the dict key in ``llm_profiles`` and is passed in separately.
    """
    cls_name = profile.get("llm_service_class") or "LiteLLMService"
    if cls_name in {"OpenAILLM", "LMStudioLLM"}:
        cls_name = "LiteLLMService"

    api_key = profile.get("llm_api_key", "")
    resolved_key = os.environ.get(api_key, api_key) if api_key else None
    base_url = profile.get("llm_endpoint", "") or None

    backends = _llm_backend_classes()
    cls = backends.get(cls_name) or backends.get("LiteLLMService")
    if cls is None:
        raise RuntimeError(f"No LLM backend named {cls_name!r} is installed.")
    llm = cls(model_name, api_key=resolved_key, base_url=base_url)
    llm.capabilities.update({k: v for k, v in (profile.get("llm_capabilities") or {}).items() if k in llm.capabilities})

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
                "class": pconf.get("llm_service_class", "LiteLLMService"),
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
